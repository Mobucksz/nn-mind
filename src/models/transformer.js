// Attention-based Transformer for surface encoding.
// TensorFlow.js port of models/transformer.py.
// Input:  per-node features [moneyness, T, norm_expiry_idx, norm_strike_idx]
// Output: price per (strike, expiry). 4 heads, 2 layers, d_model=64.

const tf = require('@tensorflow/tfjs');
const { OptionPricer } = require('./base');
const { PositionalEncoding, MultiHeadSelfAttention } = require('./layers');
const { expiryT, zeros2d } = require('../util');
const { saveModel, loadModel } = require('./fileio');

// Post-norm transformer encoder, matching torch nn.TransformerEncoderLayer
// defaults (dim_feedforward=2048, relu, dropout, norm after residual).
// dff was 2048 (a torch default) which is wildly oversized for a 4-feature,
// ~150-node IV-surface problem and made CPU training ~20s/epoch. 256 keeps the
// model expressive while cutting per-epoch cost by ~8x. Saved checkpoints are
// self-describing, so older 2048-wide models still load fine.
function buildTransformer(inputDim = 4, dModel = 64, nHeads = 4, numLayers = 2, dropout = 0.1, dff = 256) {
  const inp = tf.input({ shape: [null, inputDim] });
  let x = tf.layers.dense({ units: dModel }).apply(inp); // input projection
  x = new PositionalEncoding({ dModel }).apply(x);

  for (let l = 0; l < numLayers; l++) {
    const attn = new MultiHeadSelfAttention({ dModel, nHeads }).apply(x);
    const attnD = tf.layers.dropout({ rate: dropout }).apply(attn);
    let x1 = tf.layers.add().apply([x, attnD]);
    x1 = tf.layers.layerNormalization({ axis: -1 }).apply(x1);

    let ff = tf.layers.dense({ units: dff, activation: 'relu' }).apply(x1);
    ff = tf.layers.dense({ units: dModel }).apply(ff);
    ff = tf.layers.dropout({ rate: dropout }).apply(ff);
    let x2 = tf.layers.add().apply([x1, ff]);
    x2 = tf.layers.layerNormalization({ axis: -1 }).apply(x2);
    x = x2;
  }

  const out = tf.layers.dense({ units: 1 }).apply(x); // [B, S, 1]
  return tf.model({ inputs: inp, outputs: out });
}

class TransformerPricer extends OptionPricer {
  constructor(name = 'transformer') {
    super(name);
    this.model = buildTransformer();
    this._trained = false;
    this._strikes = null;
    this._expiries = null;
    this._scaler = null; // { mean: [4], std: [4] }
    this._targetScaler = null; // { mean: scalar, std: scalar }
    this._trainHistory = [];
  }

  get isTrained() {
    return this._trained;
  }

  _buildGridFeatures(S, strikes, expiries) {
    const NE = expiries.length;
    const NS = strikes.length;
    const rows = [];
    for (let ei = 0; ei < NE; ei++) {
      const T = expiryT(expiries[ei]);
      for (let si = 0; si < NS; si++) {
        rows.push([
          strikes[si] / S, // moneyness
          T, // time to expiry
          ei / Math.max(NE - 1, 1), // norm expiry index
          si / Math.max(NS - 1, 1), // norm strike index
        ]);
      }
    }
    return rows;
  }

  _normalizeRows(rows) {
    if (!this._scaler) return rows;
    const { mean, std } = this._scaler;
    return rows.map((r) => r.map((v, i) => (v - mean[i]) / (std[i] + 1e-8)));
  }

  surface(strikes, expiries, S, r = 0.045, q = 0) {
    const NE = expiries.length;
    const NS = strikes.length;
    if (!this._trained) return zeros2d(NE, NS);
    const rows = this._normalizeRows(this._buildGridFeatures(S, strikes, expiries));
    const out = tf.tidy(() => {
      const x = tf.tensor3d([rows]); // [1, N, 4]
      const y = this.model.predict(x); // [1, N, 1]
      return y.reshape([rows.length]).arraySync();
    });
    const tmean = this._targetScaler ? this._targetScaler.mean : 0;
    const tstd = this._targetScaler ? this._targetScaler.std : 1;
    const result = zeros2d(NE, NS);
    let kk = 0;
    for (let i = 0; i < NE; i++) {
      for (let j = 0; j < NS; j++) {
        result[i][j] = Math.max(out[kk++] * tstd + tmean, 0);
      }
    }
    return result;
  }

  price(S, K, T, r = 0.045, q = 0) {
    if (!this._trained || !this._strikes || !this._expiries) return 0;
    const strikes = this._strikes;
    const expT = this._expiries.map(expiryT);
    let si = 0;
    let best = Infinity;
    for (let j = 0; j < strikes.length; j++) {
      const d = Math.abs(strikes[j] - K);
      if (d < best) {
        best = d;
        si = j;
      }
    }
    let ei = 0;
    best = Infinity;
    for (let i = 0; i < expT.length; i++) {
      const d = Math.abs(expT[i] - T);
      if (d < best) {
        best = d;
        ei = i;
      }
    }
    const result = this.surface(strikes, this._expiries, S, r, q);
    return result[ei][si];
  }

  // Real attention: how strongly the encoder attends from the queried (K, T)
  // node to every (strike, expiry) node on the grid. We run one forward pass
  // with weight-capture enabled on each self-attention layer, average the
  // softmax maps across layers (and heads, done in the layer), then return the
  // query node's row reshaped to the NE x NS grid.
  attention(S, K, T) {
    if (!this._trained || !this._strikes || !this._expiries) return null;
    const strikes = this._strikes;
    const expiries = this._expiries;
    const NE = expiries.length;
    const NS = strikes.length;

    // Nearest grid node to the query (K, T).
    const nearest = (arr, target) => {
      let idx = 0, best = Infinity;
      for (let i = 0; i < arr.length; i++) {
        const d = Math.abs(arr[i] - target);
        if (d < best) { best = d; idx = i; }
      }
      return idx;
    };
    const si = nearest(strikes, K);
    const ei = nearest(expiries.map(expiryT), T);
    const queryIdx = ei * NS + si;

    // Find the self-attention layers (works for both fresh and loaded models).
    const attnLayers = this.model.layers.filter(
      (l) => l instanceof MultiHeadSelfAttention || (l.getClassName && l.getClassName() === 'MultiHeadSelfAttention')
    );
    if (attnLayers.length === 0) {
      const v = 1 / (NE * NS);
      return zeros2d(NE, NS).map((row) => row.map(() => v));
    }

    attnLayers.forEach((l) => { l.collectAttention = true; });
    try {
      const rows = this._normalizeRows(this._buildGridFeatures(S, strikes, expiries));
      tf.tidy(() => { this.model.predict(tf.tensor3d([rows])); }); // populates lastAttention

      // Average the [S,S] maps across layers, then read the query row.
      const row = tf.tidy(() => {
        let acc = attnLayers[0].lastAttention.clone();
        for (let i = 1; i < attnLayers.length; i++) acc = tf.add(acc, attnLayers[i].lastAttention);
        const avg = tf.div(acc, attnLayers.length);   // [S,S]
        return avg.slice([queryIdx, 0], [1, NE * NS]).reshape([NE * NS]).arraySync();
      });

      const out = zeros2d(NE, NS);
      let kk = 0;
      for (let i = 0; i < NE; i++) for (let j = 0; j < NS; j++) out[i][j] = row[kk++];
      return out;
    } finally {
      attnLayers.forEach((l) => {
        l.collectAttention = false;
        if (l.lastAttention) { l.lastAttention.dispose(); l.lastAttention = null; }
      });
    }
  }

  setNormalization(scaler, targetScaler) {
    this._scaler = scaler;
    this._targetScaler = targetScaler;
  }

  setGrid(strikes, expiries) {
    this._strikes = strikes;
    this._expiries = expiries;
  }

  async save(p) {
    await saveModel(this.model, p, {
      scaler: this._scaler,
      targetScaler: this._targetScaler,
      trained: this._trained,
      strikes: this._strikes,
      expiries: this._expiries,
      history: this._trainHistory,
      name: this.name,
    });
  }

  async load(p) {
    const { model, extra } = await loadModel(p);
    this.model = model;
    this._scaler = extra.scaler;
    this._targetScaler = extra.targetScaler;
    this._trained = extra.trained || false;
    this._strikes = extra.strikes;
    this._expiries = extra.expiries;
    this._trainHistory = extra.history || [];
  }
}

module.exports = { TransformerPricer, buildTransformer };
