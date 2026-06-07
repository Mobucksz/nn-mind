// Training loop for neural network option pricers (tfjs port of models/trainer.py).
// Loads recorded session data, computes features, trains models with Adam +
// cosine-annealed LR, and reports per-epoch history for live status updates.

const tf = require('@tensorflow/tfjs');
const { TransformerPricer } = require('./models/transformer');
const { expiryT } = require('./util');

// --- feature/target normalization helpers ---------------------------------

function colMeanStd(rows) {
  const n = rows.length;
  const d = rows[0].length;
  const mean = new Array(d).fill(0);
  const std = new Array(d).fill(0);
  for (const r of rows) for (let i = 0; i < d; i++) mean[i] += r[i];
  for (let i = 0; i < d; i++) mean[i] /= n;
  for (const r of rows) for (let i = 0; i < d; i++) std[i] += (r[i] - mean[i]) ** 2;
  for (let i = 0; i < d; i++) std[i] = Math.sqrt(std[i] / n);
  return { mean, std };
}

function scalarMeanStd(values) {
  const n = values.length;
  let m = 0;
  for (const v of values) m += v;
  m /= n;
  let s = 0;
  for (const v of values) s += (v - m) ** 2;
  return { mean: m, std: Math.sqrt(s / n) };
}

function normalizeRows(rows, scaler) {
  return rows.map((r) => r.map((v, i) => (v - scaler.mean[i]) / (scaler.std[i] + 1e-8)));
}

// Convert a recorded session into features/targets for a given model type.
function prepareFeatures(sd, mode = 'mlp') {
  const { data, strikes, expiries, spot } = sd;
  const NF = data.length;
  const NE = expiries.length;
  const NS = strikes.length;
  const S0 = median(spot);
  const ivAt = (fi, ei, si) => data[fi][ei][si][0];

  if (mode === 'mlp') {
    const rows = [];
    const targets = [];
    for (let fi = 0; fi < NF; fi++) {
      for (let ei = 0; ei < NE; ei++) {
        const T = expiryT(expiries[ei]);
        for (let si = 0; si < NS; si++) {
          const K = strikes[si];
          rows.push([S0, K, T, 0.045, 0.0, K / S0, T]);
          targets.push([
            ivAt(fi, ei, si), // price proxy (IV)
            data[fi][ei][si][1], // delta
            data[fi][ei][si][2], // gamma
            data[fi][ei][si][3], // vega
            data[fi][ei][si][4], // theta
          ]);
        }
      }
    }
    const scaler = colMeanStd(rows);
    const targetScaler = colMeanStd(targets);
    const Xn = normalizeRows(rows, scaler);
    const yn = normalizeRows(targets, targetScaler);
    const nTrain = (NF - 1) * NE * NS;
    return {
      Xtrain: Xn.slice(0, nTrain),
      ytrain: yn.slice(0, nTrain),
      Xval: Xn.slice(nTrain),
      yval: yn.slice(nTrain),
      scaler,
      targetScaler,
    };
  }

  if (mode === 'transformer') {
    const frames = [];
    const targets = [];
    for (let fi = 0; fi < NF; fi++) {
      const gf = [];
      const gt = [];
      for (let ei = 0; ei < NE; ei++) {
        const T = expiryT(expiries[ei]);
        for (let si = 0; si < NS; si++) {
          gf.push([strikes[si] / S0, T, ei / Math.max(NE - 1, 1), si / Math.max(NS - 1, 1)]);
          gt.push(ivAt(fi, ei, si));
        }
      }
      frames.push(gf);
      targets.push(gt);
    }
    const flat = frames.flat();
    const scaler = colMeanStd(flat);
    const targetScaler = scalarMeanStd(targets.flat());
    const Xn = frames.map((f) => normalizeRows(f, scaler));
    const yn = targets.map((row) => row.map((v) => (v - targetScaler.mean) / (targetScaler.std + 1e-8)));
    const nTrain = NF - 1;
    return {
      Xtrain: Xn.slice(0, nTrain),
      ytrain: yn.slice(0, nTrain),
      Xval: Xn.slice(nTrain),
      yval: yn.slice(nTrain),
      scaler,
      targetScaler,
      seqLen: NE * NS,
    };
  }

  if (mode === 'lstm') {
    const seqs = [];
    const targets = [];
    for (let fi = 0; fi < NF; fi++) {
      for (let si = 0; si < NS; si++) {
        const seq = [];
        for (let ei = 0; ei < NE; ei++) seq.push(ivAt(fi, ei, si));
        seqs.push(seq.map((v) => [v])); // [NE, 1]
        targets.push(seq); // [NE]
      }
    }
    const allVals = seqs.flat().map((x) => x[0]);
    const scaler = scalarMeanStd(allVals);
    const targetScaler = scalarMeanStd(targets.flat());
    const Xn = seqs.map((s) => s.map(([v]) => [(v - scaler.mean) / (scaler.std + 1e-8)]));
    const yn = targets.map((row) => row.map((v) => (v - targetScaler.mean) / (targetScaler.std + 1e-8)));
    const nTrain = (NF - 1) * NS;
    return {
      Xtrain: Xn.slice(0, nTrain),
      ytrain: yn.slice(0, nTrain),
      Xval: Xn.slice(nTrain),
      yval: yn.slice(nTrain),
      scaler,
      targetScaler,
      seqLen: NE,
    };
  }

  throw new Error(`Unknown mode '${mode}'`);
}

function median(a) {
  const b = [...a].sort((x, y) => x - y);
  const m = Math.floor(b.length / 2);
  return b.length % 2 ? b[m] : (b[m - 1] + b[m]) / 2;
}

// Build the input/label tensors for a mode from prepared arrays.
function toTensors(mode, data) {
  if (mode === 'mlp') {
    return {
      Xtrain: tf.tensor2d(data.Xtrain),
      ytrain: tf.tensor2d(data.ytrain),
      Xval: tf.tensor2d(data.Xval),
      yval: tf.tensor2d(data.yval),
    };
  }
  // transformer + lstm: 3D inputs, labels reshaped to [..., seq, 1].
  const lab = (rows) => tf.tensor3d(rows.map((r) => r.map((v) => [v])));
  return {
    Xtrain: tf.tensor3d(data.Xtrain),
    ytrain: lab(data.ytrain),
    Xval: tf.tensor3d(data.Xval),
    yval: lab(data.yval),
  };
}

// Cosine-annealed LR (T_max = epochs, eta_min = 0); lr at epoch e (0-indexed).
function cosineLr(base, e, epochs) {
  return (base * (1 + Math.cos((Math.PI * e) / epochs))) / 2;
}

// Train an NN pricer on recorded session data. Async so it can yield to the
// event loop between epochs (keeps IPC/status responsive in the main process).
async function trainModel(pricer, sd, { epochs = 50, lr = 0.001, statusCallback = null } = {}) {
  const mode = type(pricer);
  const prepared = prepareFeatures(sd, mode);
  const t = toTensors(mode, prepared);

  const optimizer = tf.train.adam(lr);
  const history = [];

  try {
    for (let epoch = 0; epoch < epochs; epoch++) {
      const curLr = cosineLr(lr, epoch, epochs);
      try {
        optimizer.learningRate = curLr;
      } catch {
        /* some optimizer builds expose lr read-only; base lr still trains */
      }

      const lossScalar = optimizer.minimize(
        () => {
          const pred = pricer.model.apply(t.Xtrain, { training: true });
          return tf.losses.meanSquaredError(t.ytrain, pred);
        },
        true
      );
      const loss = lossScalar.dataSync()[0];
      lossScalar.dispose();

      const valLoss = tf.tidy(() => {
        const pred = pricer.model.predict(t.Xval);
        return tf.losses.meanSquaredError(t.yval, pred).dataSync()[0];
      });

      history.push({ epoch: epoch + 1, loss, val_loss: valLoss, lr: curLr });
      if (statusCallback) statusCallback(epoch + 1, loss, valLoss, curLr);

      // Yield so status polling / other IPC can run between epochs.
      await new Promise((r) => setImmediate(r));
    }
  } finally {
    t.Xtrain.dispose();
    t.ytrain.dispose();
    t.Xval.dispose();
    t.yval.dispose();
    optimizer.dispose();
  }

  pricer._trained = true;
  pricer._trainHistory = history;
  pricer.setNormalization(prepared.scaler, prepared.targetScaler);
  if (pricer instanceof TransformerPricer) {
    pricer.setGrid(sd.strikes, sd.expiries);
  }
  return history;
}

// Map a pricer instance to its training mode.
function type(pricer) {
  const n = pricer.constructor.name.toLowerCase().replace('pricer', '');
  return n;
}

module.exports = { prepareFeatures, trainModel };
