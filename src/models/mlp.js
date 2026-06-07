// Feedforward MLP for option pricing (TensorFlow.js port of models/mlp.py).
// Input:  [S, K, T, r, q, moneyness, time_to_expiry]
// Output: [price, delta, gamma, vega, theta]
// 3 hidden layers (128, 64, 32) with ReLU + dropout.

const tf = require('@tensorflow/tfjs');
const { OptionPricer } = require('./base');
const { expiryT, zeros2d } = require('../util');
const { saveModel, loadModel } = require('./fileio');

function buildMLP(inputDim = 7, hidden = [128, 64, 32], dropout = 0.2) {
  const m = tf.sequential();
  m.add(tf.layers.dense({ inputShape: [inputDim], units: hidden[0], activation: 'relu' }));
  m.add(tf.layers.dropout({ rate: dropout }));
  for (let i = 1; i < hidden.length; i++) {
    m.add(tf.layers.dense({ units: hidden[i], activation: 'relu' }));
    m.add(tf.layers.dropout({ rate: dropout }));
  }
  m.add(tf.layers.dense({ units: 5 })); // price, delta, gamma, vega, theta
  return m;
}

class MLPPricer extends OptionPricer {
  constructor(name = 'mlp', inputDim = 7) {
    super(name);
    this.inputDim = inputDim;
    this.model = buildMLP(inputDim);
    this._trained = false;
    this._scaler = null; // { mean: [7], std: [7] }
    this._targetScaler = null; // { mean: [5], std: [5] }
    this._trainHistory = [];
  }

  get isTrained() {
    return this._trained;
  }

  _prepareInput(S, K, T, r = 0.045, q = 0) {
    return [S, K, T, r, q, K / S, T];
  }

  _normalize(x) {
    if (!this._scaler) return x;
    return x.map((v, i) => (v - this._scaler.mean[i]) / (this._scaler.std[i] + 1e-8));
  }

  // Run a batch of feature rows through the net -> [[5], ...] raw outputs.
  _forward(rows) {
    return tf.tidy(() => {
      const x = tf.tensor2d(rows);
      const out = this.model.predict(x);
      return out.arraySync();
    });
  }

  price(S, K, T, r = 0.045, q = 0) {
    if (!this._trained) return 0;
    const feat = this._normalize(this._prepareInput(S, K, T, r, q));
    const out = this._forward([feat])[0];
    // Denormalize price using the channel-0 target stats (matches mlp.py).
    let price = out[0];
    if (this._targetScaler) price = price * this._targetScaler.std[0] + this._targetScaler.mean[0];
    return Math.max(price, 0);
  }

  greeks(S, K, T, r = 0.045, q = 0) {
    if (!this._trained) return { delta: 0, gamma: 0, vega: 0, theta: 0 };
    const feat = this._normalize(this._prepareInput(S, K, T, r, q));
    const out = this._forward([feat])[0];
    // Greeks are returned as the raw network outputs, matching mlp.py.
    return { delta: out[1], gamma: out[2], vega: out[3], theta: out[4] };
  }

  surface(strikes, expiries, S, r = 0.045, q = 0) {
    const exp = expiries.map(expiryT);
    const NE = exp.length;
    const NS = strikes.length;
    if (!this._trained) return zeros2d(NE, NS);
    // Batch every (expiry, strike) node through the net in one predict.
    const rows = [];
    for (let i = 0; i < NE; i++) {
      for (let j = 0; j < NS; j++) {
        rows.push(this._normalize(this._prepareInput(S, strikes[j], exp[i], r, q)));
      }
    }
    const out = this._forward(rows);
    const m0 = this._targetScaler ? this._targetScaler.mean[0] : 0;
    const s0 = this._targetScaler ? this._targetScaler.std[0] : 1;
    const result = zeros2d(NE, NS);
    let k = 0;
    for (let i = 0; i < NE; i++) {
      for (let j = 0; j < NS; j++) {
        result[i][j] = Math.max(out[k++][0] * s0 + m0, 0);
      }
    }
    return result;
  }

  setNormalization(scaler, targetScaler) {
    this._scaler = scaler;
    this._targetScaler = targetScaler;
  }

  async save(p) {
    await saveModel(this.model, p, {
      scaler: this._scaler,
      targetScaler: this._targetScaler,
      trained: this._trained,
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
    this._trainHistory = extra.history || [];
  }
}

module.exports = { MLPPricer, buildMLP };
