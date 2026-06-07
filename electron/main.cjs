// Electron wrapper for nn-mind — loads Flask in a native window.
const { app, BrowserWindow } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const http = require('http');

const PORT = 7891;
const FLASK_DIR = path.join(__dirname, '..');
const PYTHON = 'C:\\Users\\Bryce\\AppData\\Local\\Python\\bin\\python.exe';
const MAX_RETRIES = 60; // 30 seconds

let mainWindow = null;
let flask = null;

function startFlask() {
  flask = spawn(PYTHON, ['app.py'], {
    cwd: FLASK_DIR,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
  });
  flask.stdout.on('data', (d) => process.stdout.write('[flask] ' + d.toString()));
  flask.stderr.on('data', (d) => process.stderr.write('[flask] ' + d.toString()));
  flask.on('exit', (code) => {
    if (code !== 0 && code !== null) console.error('Flask exited with code:', code);
  });
}

function waitForFlask(retries) {
  const req = http.get('http://127.0.0.1:' + PORT, (res) => {
    res.resume(); // drain response
    console.log('Flask ready on port', PORT);
    createWindow();
  });
  req.on('error', () => {
    if (retries > 0) {
      setTimeout(() => waitForFlask(retries - 1), 500);
    } else {
      console.error('Flask did not start in time');
      app.quit();
    }
  });
  req.setTimeout(1000, () => { req.destroy(); });
  // Do NOT call req.end() — http.get does it automatically
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440, height: 900,
    minWidth: 1024, minHeight: 680,
    backgroundColor: '#07080c',
    title: 'nn-mind',
    autoHideMenuBar: true,
    webPreferences: { contextIsolation: true, nodeIntegration: false },
  });
  mainWindow.loadURL('http://127.0.0.1:' + PORT);
  mainWindow.on('closed', () => { mainWindow = null; });
}

app.on('ready', () => {
  startFlask();
  setTimeout(() => waitForFlask(MAX_RETRIES), 1500);
});

app.on('window-all-closed', () => {
  if (flask) { flask.kill(); flask = null; }
  app.quit();
});

app.on('before-quit', () => {
  if (flask) { flask.kill(); flask = null; }
});
