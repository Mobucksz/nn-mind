// Request handlers - the Node/IPC equivalent of the Flask routes in app.py.
// Each handler returns { status, body }; the IPC bridge turns a non-2xx status
// into a thrown Error(body.error) on the renderer side, mirroring fetch().

const worker = require('./worker');
const { latestSession, listSessions } = require('./loader');
const { createPricer } = require('./models');
const { median, min, max, mean, expiryT } = require('./util');

const ok = (body) => ({ status: 200, body });
const err = (status, message) => ({ status, body: { error: message } });

// Normalize an expiry entry to { date, days, T }.
function expiryInfo(e) {
  if (e && typeof e === 'object') {
    return { date: e.date, days: e.days, T: e.T };
  }
  return { date: '?', days: 0, T: Number(e) };
}

// --- Training ---------------------------------------------------------------

function train(payload = {}) {
  const modelType = payload.model_type || 'mlp';
  const epochs = parseInt(payload.epochs ?? 50, 10);
  const lr = parseFloat(payload.lr ?? 0.001);

  if (modelType === 'bs' || modelType === 'binomial') {
    return err(400, `'${modelType}' is a closed-form model, no training needed`);
  }
  const jobId = worker.startTraining(modelType, { epochs, lr });
  return { status: 202, body: { job_id: jobId, status: 'queued' } };
}

function trainStatus(payload = {}) {
  const job = worker.getJobStatus(payload.job_id);
  if (!job) return err(404, 'Job not found');
  return ok(job);
}

function trainJobs() {
  return ok(worker.listJobs());
}

// --- Inference --------------------------------------------------------------

function infer(payload = {}) {
  const S = Number(payload.S);
  const K = Number(payload.K);
  const T = Number(payload.T);
  const modelType = payload.model_type || 'mlp';
  const r = payload.r != null ? Number(payload.r) : 0.045;

  let price;
  let greeks;
  if (modelType === 'bs' || modelType === 'binomial') {
    const pricer = createPricer(modelType);
    price = pricer.price(S, K, T, r);
    greeks = pricer.greeks(S, K, T, r);
  } else {
    const model = worker.getModel(modelType);
    if (!model || !model.isTrained) {
      return err(400, `Model '${modelType}' not trained yet`);
    }
    price = model.price(S, K, T, r);
    greeks = model.greeks(S, K, T, r);
  }
  return ok({ price, model: modelType, ...greeks });
}

// --- Surface ----------------------------------------------------------------

function surface(payload = {}) {
  const modelType = payload.model || 'mlp';
  const sd = latestSession();
  if (!sd) return err(400, 'No session data');

  const strikes = sd.strikes;
  const expiries = sd.expiries.map((e) => (e && typeof e === 'object' ? expiryInfo(e) : e));
  const S0 = median(sd.spot);

  let pricer;
  if (modelType === 'bs' || modelType === 'binomial') {
    pricer = createPricer(modelType);
  } else {
    const model = worker.getModel(modelType);
    if (!model || !model.isTrained) return err(400, `Model '${modelType}' not trained`);
    pricer = model;
  }

  const prices = pricer.surface(sd.strikes, sd.expiries, S0);
  return ok({ model: modelType, strikes, expiries, prices, S: S0 });
}

// --- Compare ----------------------------------------------------------------

function compare(payload = {}) {
  const modelStr = payload.models || 'mlp,bs';
  const modelTypes = modelStr.split(',').map((m) => m.trim());

  const sd = latestSession();
  if (!sd) return err(400, 'No session data');
  const S0 = median(sd.spot);

  const surfaces = worker.computeCompareSurfaces(modelTypes, sd.strikes, sd.expiries, S0);

  // Add closed-form models if requested but not cached.
  for (const mt of modelTypes) {
    if (!(mt in surfaces) && (mt === 'bs' || mt === 'binomial')) {
      surfaces[mt] = createPricer(mt).surface(sd.strikes, sd.expiries, S0);
    }
  }

  // Pairwise differences (A - B).
  const differences = {};
  const keys = Object.keys(surfaces);
  for (let i = 0; i < keys.length; i++) {
    for (let j = i + 1; j < keys.length; j++) {
      const a = surfaces[keys[i]];
      const b = surfaces[keys[j]];
      differences[`${keys[i]}-${keys[j]}`] = a.map((row, ri) => row.map((v, ci) => v - b[ri][ci]));
    }
  }

  return ok({
    models: keys,
    surfaces,
    differences,
    strikes: sd.strikes,
    expiries: sd.expiries.map((e) => (e && typeof e === 'object' ? e.date : String(e))),
    S: S0,
  });
}

// --- Models -----------------------------------------------------------------

function models() {
  const trained = worker.listAvailableModels();

  const closedForm = [
    { name: 'bs', type: 'bs', trained: true, has_attention: false, description: 'Black-Scholes closed-form' },
    { name: 'binomial', type: 'binomial', trained: true, has_attention: false, description: 'Binomial tree (American)' },
  ];

  const all = closedForm.concat(trained.filter((m) => m.name !== 'bs' && m.name !== 'binomial'));

  const descriptions = {
    mlp: 'Feedforward MLP (3 layers)',
    lstm: 'LSTM for vol term structure',
    transformer: 'Attention-based surface encoder',
    ensemble: 'Stacked ensemble with uncertainty',
  };
  const nnTypes = ['mlp', 'lstm', 'transformer', 'ensemble'];
  const existing = new Set(all.map((m) => m.name));
  for (const nt of nnTypes) {
    if (!existing.has(nt)) {
      all.push({
        name: nt,
        type: nt,
        trained: false,
        has_attention: nt === 'transformer',
        description: descriptions[nt],
      });
    }
  }
  return ok(all);
}

// --- Session info -----------------------------------------------------------

function latestSessionInfo() {
  const sd = latestSession();
  if (!sd) return err(404, 'No sessions available');
  return ok({
    meta: sd.meta,
    strikes: sd.strikes,
    expiries: sd.expiries.map(expiryInfo),
    spot_min: min(sd.spot),
    spot_max: max(sd.spot),
    spot_avg: mean(sd.spot),
    nframes: sd.nframes,
    shape: [sd.nframes, sd.nexp, sd.nstrikes, 7],
  });
}

function sessions() {
  return ok(
    listSessions().map((s) => ({
      date: s.date,
      symbol: s.symbol,
      source: s.source,
      path: String(s.path),
    }))
  );
}

// --- Attention --------------------------------------------------------------

function attention(payload = {}) {
  const S = Number(payload.S);
  const K = Number(payload.K);
  const T = Number(payload.T);
  const model = worker.getModel('transformer');
  if (!model || !model.isTrained) return err(400, 'Transformer not trained yet');
  return ok({ attention: model.attention(S, K, T) });
}

// --- Dispatch ---------------------------------------------------------------

const ROUTES = {
  'latest-session': latestSessionInfo,
  sessions,
  models,
  surface,
  compare,
  infer,
  train,
  'train-status': trainStatus,
  'train-jobs': trainJobs,
  attention,
};

async function handle(action, payload) {
  const fn = ROUTES[action];
  if (!fn) return err(404, `Unknown action '${action}'`);
  try {
    return await fn(payload);
  } catch (e) {
    console.error(`handler '${action}' failed:`, e);
    return err(500, e.message || String(e));
  }
}

module.exports = { handle };
