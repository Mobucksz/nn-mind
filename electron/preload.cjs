// Safe bridge between the renderer (web/index.html) and the Electron main
// process. The UI never touches Node or the sidecar directly — it calls
// window.api.invoke(channel, payload) and subscribes to live events.

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  // Request/response to the Python sidecar (replaces the old fetch('/api/...')).
  invoke: (channel, payload) => ipcRenderer.invoke('api', { channel, payload }),

  // Unsolicited events (ibkr.snapshot, ibkr.status, sidecar.*).
  onEvent: (handler) => {
    const listener = (_evt, msg) => handler(msg);
    ipcRenderer.on('event', listener);
    return () => ipcRenderer.removeListener('event', listener);
  },
});
