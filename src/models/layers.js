// Custom TensorFlow.js layers for the transformer encoder.
// tfjs core has no MultiHeadAttention / positional-encoding layers, so we
// implement them here and register them for model save/load.

const tf = require('@tensorflow/tfjs');

// Sinusoidal positional encoding, precomputed as a constant [1, maxLen, dModel].
function computePE(maxLen, dModel) {
  const pe = [];
  for (let pos = 0; pos < maxLen; pos++) {
    const row = new Array(dModel).fill(0);
    for (let i = 0; i < dModel; i += 2) {
      const divTerm = Math.exp((i * -Math.log(10000.0)) / dModel);
      row[i] = Math.sin(pos * divTerm);
      if (i + 1 < dModel) row[i + 1] = Math.cos(pos * divTerm);
    }
    pe.push(row);
  }
  return tf.tensor3d([pe]); // [1, maxLen, dModel]
}

class PositionalEncoding extends tf.layers.Layer {
  constructor(config) {
    super(config);
    this.dModel = config.dModel;
    this.maxLen = config.maxLen || 5000;
  }

  build() {
    this.pe = tf.keep(computePE(this.maxLen, this.dModel));
    this.built = true;
  }

  call(inputs) {
    const x = Array.isArray(inputs) ? inputs[0] : inputs;
    const seq = x.shape[1];
    const slice = this.pe.slice([0, 0, 0], [1, seq, this.dModel]);
    return tf.add(x, slice);
  }

  computeOutputShape(inputShape) {
    return inputShape;
  }

  getConfig() {
    return Object.assign(super.getConfig(), { dModel: this.dModel, maxLen: this.maxLen });
  }

  static get className() {
    return 'PositionalEncoding';
  }
}
tf.serialization.registerClass(PositionalEncoding);

// Multi-head self-attention with separate Q/K/V/O projections.
class MultiHeadSelfAttention extends tf.layers.Layer {
  constructor(config) {
    super(config);
    this.dModel = config.dModel;
    this.nHeads = config.nHeads;
    this.dHead = Math.floor(this.dModel / this.nHeads);
  }

  build() {
    const w = (n) =>
      this.addWeight(n, [this.dModel, this.dModel], 'float32', tf.initializers.glorotUniform({}));
    const b = (n) => this.addWeight(n, [this.dModel], 'float32', tf.initializers.zeros());
    this.wq = w('wq');
    this.wk = w('wk');
    this.wv = w('wv');
    this.wo = w('wo');
    this.bq = b('bq');
    this.bk = b('bk');
    this.bv = b('bv');
    this.bo = b('bo');
    this.built = true;
  }

  call(inputs) {
    const x = Array.isArray(inputs) ? inputs[0] : inputs;
    const batch = x.shape[0];
    const seq = x.shape[1];

    // x[B,S,D] . W[D,D] + b  -> [B,S,D]
    const proj = (wv, bv) => {
      const xt = tf.reshape(x, [-1, this.dModel]);
      const r = tf.add(tf.matMul(xt, wv.read()), bv.read());
      return tf.reshape(r, [batch, seq, this.dModel]);
    };

    const split = (t) =>
      tf.transpose(tf.reshape(t, [batch, seq, this.nHeads, this.dHead]), [0, 2, 1, 3]); // [B,H,S,dH]

    const q = split(proj(this.wq, this.bq));
    const k = split(proj(this.wk, this.bk));
    const v = split(proj(this.wv, this.bv));

    let scores = tf.matMul(q, k, false, true); // [B,H,S,S]
    scores = tf.div(scores, Math.sqrt(this.dHead));
    const attn = tf.softmax(scores, -1);

    // Opt-in: stash the head-averaged attention weights [S,S] for inspection.
    // Off by default so normal predict() stays allocation-free; the pricer's
    // attention() method flips it on around a single forward pass.
    if (this.collectAttention) {
      if (this.lastAttention) { this.lastAttention.dispose(); this.lastAttention = null; }
      this.lastAttention = tf.keep(tf.tidy(() => tf.mean(attn, 1).squeeze([0]))); // [S,S]
    }

    let ctx = tf.matMul(attn, v); // [B,H,S,dH]
    ctx = tf.reshape(tf.transpose(ctx, [0, 2, 1, 3]), [batch, seq, this.dModel]); // [B,S,D]

    const ctxt = tf.reshape(ctx, [-1, this.dModel]);
    const out = tf.add(tf.matMul(ctxt, this.wo.read()), this.bo.read());
    return tf.reshape(out, [batch, seq, this.dModel]);
  }

  computeOutputShape(inputShape) {
    return inputShape;
  }

  getConfig() {
    return Object.assign(super.getConfig(), { dModel: this.dModel, nHeads: this.nHeads });
  }

  static get className() {
    return 'MultiHeadSelfAttention';
  }
}
tf.serialization.registerClass(MultiHeadSelfAttention);

module.exports = { PositionalEncoding, MultiHeadSelfAttention };
