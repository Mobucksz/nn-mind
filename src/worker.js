// Background training worker with an in-memory job queue and status reporting.
// Port of src/worker.py. Training runs as a yielding async task (not a thread),
// so status polling over IPC stays responsive in the Electron main process.

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const { createPricer } = require('./models');
const { trainModel, trainModelMulti } = require('./trainer');
const { latestSession, loadSession, listSessions } = require('./loader');

// In-memory job store and live model cache.
const _jobs = {};
const _modelCache = {};

// All trained model data lives under data/.
const DATA_DIR = path.join(__dirname, '..', 'data');
const MODEL_DIR = path.join(DATA_DIR, 'models');
const SESSION_DIR = path.join(DATA_DIR, 'sessions');
const LOG_DIR = path.join(DATA_DIR, 'logs');
for (const d of [MODEL_DIR, SESSION_DIR, LOG_DIR]) fs.mkdirSync(d, { recursive: true });

const nowIso = () => new Date().toISOString();

// Load all saved checkpoints from data/models/ into the cache.
async function loadSavedModels() {
  if (!fs.existsSync(MODEL_DIR)) return;
  for (const fname of fs.readdirSync(MODEL_DIR)) {
    if (fname.endsWith('.latest.json')) {
      const modelType = fname.slice(0, -'.latest.json'.length);
      try {
        const pricer = createPricer(modelType);
        await pricer.load(path.join(MODEL_DIR, fname));
        _modelCache[modelType] = pricer;
      } catch (e) {
        console.error(`Failed to load ${fname}: ${e.message}`);
      }
    }
  }
}

function _getOrCreateModel(modelType) {
  if (!_modelCache[modelType]) _modelCache[modelType] = createPricer(modelType);
  return _modelCache[modelType];
}

// Start a training job. Returns a job id immediately; training runs async.
function startTraining(modelType, { epochs = 50, lr = 0.001, sessionPath = null, multi = false } = {}) {
  const jobId = crypto.randomUUID();
  const job = {
    id: jobId,
    model_type: modelType,
    status: 'queued',
    progress: 0,
    epochs,
    current_epoch: 0,
    loss: null,
    val_loss: null,
    lr,
    multi: !!multi,
    nsessions: 1,
    history: [],
    error: null,
    created_at: nowIso(),
    completed_at: null,
  };
  _jobs[jobId] = job;

  const statusCb = (epoch, loss, valLoss, curLr) => {
    job.status = 'running';
    job.progress = (epoch / epochs) * 100;
    job.current_epoch = epoch;
    job.loss = loss;
    job.val_loss = valLoss;
    job.lr = curLr;
  };

  (async () => {
    try {
      job.status = 'loading_data';
      const pricer = _getOrCreateModel(modelType);
      let history;

      if (multi) {
        // Train across every available recorded session (shape-grouped).
        const sds = listSessions().map((s) => loadSession(s.path)).filter(Boolean);
        if (!sds.length) throw new Error('No session data available');
        job.nsessions = sds.length;
        job.status = 'training';
        history = await trainModelMulti(pricer, sds, { epochs, lr, statusCallback: statusCb });
      } else {
        const sd = sessionPath ? loadSession(sessionPath) : latestSession();
        if (!sd) throw new Error('No session data available');
        job.status = 'training';
        history = await trainModel(pricer, sd, { epochs, lr, statusCallback: statusCb });
      }

      // Save checkpoint, rotating any previous .latest into .v{N}.
      const base = path.join(MODEL_DIR, modelType);
      const latestPath = base + '.latest.json';
      if (fs.existsSync(latestPath)) {
        let v = 1;
        while (fs.existsSync(`${base}.v${v}.json`)) v++;
        fs.renameSync(latestPath, `${base}.v${v}.json`);
      }
      await pricer.save(latestPath);

      job.status = 'completed';
      job.progress = 100;
      job.current_epoch = epochs;
      job.history = history;
      job.completed_at = nowIso();
      job.checkpoint = latestPath;
    } catch (e) {
      job.status = 'failed';
      job.error = e.message;
      job.completed_at = nowIso();
      console.error(e);
    }
  })();

  return jobId;
}

function getJobStatus(jobId) {
  const job = _jobs[jobId];
  return job ? { ...job } : null;
}

function listJobs() {
  return Object.values(_jobs).map((j) => ({ ...j }));
}

function getModel(modelType) {
  return _modelCache[modelType] || null;
}

function listAvailableModels() {
  return Object.entries(_modelCache).map(([name, model]) => ({
    name,
    type: name,
    trained: !!model.isTrained,
    has_attention: typeof model.attention === 'function',
  }));
}

function computeSurface(modelType, strikes, expiries, S) {
  const model = _modelCache[modelType];
  if (!model) return null;
  return model.surface(strikes, expiries, S);
}

function computeCompareSurfaces(modelTypes, strikes, expiries, S) {
  const surfaces = {};
  for (const mt of modelTypes) {
    const surf = computeSurface(mt, strikes, expiries, S);
    if (surf !== null) surfaces[mt] = surf;
  }
  return surfaces;
}

module.exports = {
  startTraining,
  getJobStatus,
  listJobs,
  getModel,
  listAvailableModels,
  computeSurface,
  computeCompareSurfaces,
  loadSavedModels,
  MODEL_DIR,
};
