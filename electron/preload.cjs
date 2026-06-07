// Preload bridge: exposes a minimal, safe API to the renderer.
// window.nn.invoke(action, payload) mirrors the old fetch('/api/...') contract -
// it resolves with the response body, or throws Error(message) on a non-2xx
// status, so the frontend's error handling works unchanged.

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('nn', {
  invoke: async (action, payload) => {
    const { status, body } = await ipcRenderer.invoke('nn:invoke', action, payload);
    if (status >= 400) {
      throw new Error((body && body.error) || `error ${status}`);
    }
    return body;
  },
});
