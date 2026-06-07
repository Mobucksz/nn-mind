// Electron entry point for nn-mind. Pure Electron - no Python, no Flask.
// All option-pricing and NN training runs in this (Node) main process via
// TensorFlow.js; the renderer talks to it over an IPC bridge (see preload.cjs).

const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');

const { handle } = require('../src/handlers');
const worker = require('../src/worker');

let mainWindow = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1024,
    minHeight: 680,
    backgroundColor: '#07080c',
    title: 'nn-mind',
    autoHideMenuBar: true,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, 'preload.cjs'),
    },
  });
  mainWindow.loadFile(path.join(__dirname, '..', 'web', 'index.html'));

  // Surface renderer-side logs and load failures in the main process console.
  const wc = mainWindow.webContents;
  wc.on('console-message', (_e, level, message) => console.log(`[renderer] ${message}`));
  wc.on('did-fail-load', (_e, code, desc) => console.error(`[renderer] load failed ${code}: ${desc}`));

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// Single IPC channel; the renderer calls window.nn.invoke(action, payload).
ipcMain.handle('nn:invoke', async (_event, action, payload) => {
  return handle(action, payload);
});

app.on('ready', async () => {
  // Restore any previously trained model checkpoints before opening the window.
  try {
    await worker.loadSavedModels();
  } catch (e) {
    console.error('Failed to load saved models:', e);
  }
  createWindow();
});

app.on('window-all-closed', () => {
  app.quit();
});

app.on('activate', () => {
  if (mainWindow === null) createWindow();
});
