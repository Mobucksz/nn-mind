// LSTM for volatility term structure (TensorFlow.js port of models/lstm.py).
// Input:  sequence of IV across expiries for a given strike  [seq, 1]
// Output: predicted IV per expiry                            [seq, 1]
// 2-layer LSTM, hidden size 64.
//
// As in the original, the LSTM does not price single options directly:
// price() returns 0 and surface() returns zeros. It still trains (so the UI
// shows a real loss curve); its outputs simply are not used for pricing.

const tf = require('@tensorflow/tfjs');
const { OptionPricer } = require('./base');
const { zeros2d } = require('../util');
const { saveModel, loadModel } = require('./fileio');

function buildLSTM(inputDim = 1, hiddenSize = 64, numLayers = 2) {
  const m = tf.sequential();
  m.add(
    tf.layers.lstm({
      inputShape: [null, inputDim],
      units: hiddenSize,
      returnSequences: true,
    })
  );
  for (let i = 1; i < numLayers; i++) {
    m.add(tf.layers.lstm({ units: hiddenSize, returnSequences: true }));
  }
  m.add(tf.layers.timeDistributed({ layer: tf.layers.dense({ units: inputDim }) }));
  return m;
}

class LSTMPricer extends OptionPricer {
  constructor(name = 'lstm') {
    super(name);
    this.model = buildLSTM();
    this._trained = false;
    this._scaler = null;
    this._targetScaler = null;
    this._trainHistory = [];
  }

  get isTrained() {
    return this._trained;
  }

  price() {
    return 0;
  }

  surface(strikes, expiries) {
    return zeros2d(expiries.length, strikes.length);
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

module.exports = { LSTMPricer, buildLSTM };
