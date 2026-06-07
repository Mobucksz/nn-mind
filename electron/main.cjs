// Electron wrapper for nn-mind - loads the Flask server in a native window.
const { app, BrowserWindow } = require('electron');
const { spawn } = require('child_process');
const path = require('path');

const PORT = 7891;
const FLASK_DIR = path.join(__dirname, '..');
const PYTHON = 'C:\\Users\\Bryce\\AppData\\Local\\Python\\bin\\python.exe';

let mainWindow = null;
let flask = null;

function startFlask() {
  flask = spawn(PYTHON, ['app.py'], {
    cwd: FLASK_DIR,
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
  });
  flask.stdout.on('data', (d) => console.log(d.toString()));
  flask.stderr.on('data', (d) => console.error(d.toString()));
  flask.on('exit', (code) => { if (code !== 0) console.error('Flask exited:', code); });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440, height: 900, minWidth: 1024, minHeight: 680,
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
  // Wait for Flask to be ready, then open
  const check = () => {
    const http = require('http');
    const req = http.get('http://127.0.0.1:' + PORT, () => { createWindow(); });
    req.on('error', () => setTimeout(check, 500));
    req.end();
  };
  setTimeout(check, 1500);
});

app.on('window-all-closed', () => {
  if (flask) flask.kill();
  app.quit();
});

app.on('before-quit', () => {
  if (flask) flask.kill();
});
