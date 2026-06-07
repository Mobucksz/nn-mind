// nn-mind — native Electron app. No Flask, no HTTP server.
// The window loads the UI from a local file and talks to a Python compute
// sidecar (backend/sidecar.py) over newline-delimited JSON on stdin/stdout.

const { app, BrowserWindow, ipcMain } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const readline = require('readline');
const fs = require('fs');

const ROOT = path.join(__dirname, '..');

// Resolve a Python interpreter: env override → common local paths → PATH.
function resolvePython() {
  const candidates = [
    process.env.NN_MIND_PYTHON,
    process.platform === 'win32'
      ? path.join(process.env.LOCALAPPDATA || '', 'Python', 'bin', 'python.exe')
      : null,
    path.join(ROOT, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python'),
    process.platform === 'win32' ? 'python.exe' : 'python3',
    'python',
  ].filter(Boolean);
  for (const c of candidates) {
    if (c.includes(path.sep) && !fs.existsSync(c)) continue;
    return c;
  }
  return process.platform === 'win32' ? 'python' : 'python3';
}

let mainWindow = null;
let sidecar = null;
let nextId = 1;
const pending = new Map();        // id -> {resolve, reject}

function startSidecar() {
  const py = resolvePython();
  sidecar = spawn(py, ['backend/sidecar.py'], {
    cwd: ROOT,
    stdio: ['pipe', 'pipe', 'pipe'],
    env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1' },
  });

  sidecar.on('error', (err) => {
    if (mainWindow) mainWindow.webContents.send('event', {
      event: 'sidecar.error', data: { message: `Failed to start Python (${py}): ${err.message}` },
    });
  });
  sidecar.on('exit', (code) => {
    if (mainWindow) mainWindow.webContents.send('event', {
      event: 'sidecar.exit', data: { code },
    });
  });

  const rl = readline.createInterface({ input: sidecar.stdout });
  rl.on('line', (line) => {
    line = line.trim();
    if (!line) return;
    let msg;
    try { msg = JSON.parse(line); } catch { return; }
    if (msg.event) {
      if (mainWindow) mainWindow.webContents.send('event', msg);
      return;
    }
    const p = pending.get(msg.id);
    if (!p) return;
    pending.delete(msg.id);
    if (msg.ok) p.resolve(msg.result);
    else p.reject(new Error(msg.error || 'sidecar error'));
  });

  // Surface Python tracebacks/logs to the Electron console for debugging.
  readline.createInterface({ input: sidecar.stderr })
    .on('line', (l) => console.error('[sidecar]', l));
}

function callSidecar(channel, payload) {
  return new Promise((resolve, reject) => {
    if (!sidecar || sidecar.killed) return reject(new Error('sidecar not running'));
    const id = nextId++;
    pending.set(id, { resolve, reject });
    sidecar.stdin.write(JSON.stringify({ id, channel, payload: payload || {} }) + '\n');
    // Safety timeout so a wedged channel doesn't hang the UI forever.
    setTimeout(() => {
      if (pending.has(id)) { pending.delete(id); reject(new Error(`'${channel}' timed out`)); }
    }, 30000);
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440, height: 900, minWidth: 1024, minHeight: 680,
    backgroundColor: '#07080c',
    title: 'nn-mind',
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  mainWindow.loadFile(path.join(ROOT, 'web', 'index.html'));
  mainWindow.on('closed', () => { mainWindow = null; });
}

app.on('ready', () => {
  startSidecar();
  ipcMain.handle('api', (_evt, { channel, payload }) => callSidecar(channel, payload));
  createWindow();
});

function shutdown() {
  if (sidecar && !sidecar.killed) {
    try { callSidecar('ibkr.disconnect', {}); } catch (_) {}
    sidecar.kill();
  }
}
app.on('window-all-closed', () => { shutdown(); app.quit(); });
app.on('before-quit', shutdown);
