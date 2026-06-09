// Ensemble: stack multiple models, return mean + uncertainty.
// Port of models/ensemble.py.

const { OptionPricer } = require('./base');
const { mean, std, zeros2d } = require('../util');

function trained(m) {
  return m && m.isTrained;
}

class EnsemblePricer extends OptionPricer {
  constructor(name = 'ensemble', models = null) {
    super(name);
    this.models = models || [];
  }

  addModel(model) {
    this.models.push(model);
  }

  get isTrained() {
    return this.models.length > 0;
  }

  price(S, K, T, r = 0.045, q = 0) {
    const prices = this.models.filter(trained).map((m) => m.price(S, K, T, r, q));
    if (!prices.length) return 0;
    return mean(prices);
  }

  priceWithUncertainty(S, K, T, r = 0.045, q = 0) {
    const prices = [];
    const details = [];
    for (const m of this.models) {
      if (trained(m)) {
        const p = m.price(S, K, T, r, q);
        prices.push(p);
        details.push({ model: m.name, price: p });
      }
    }
    if (!prices.length) return { mean: 0, std: 0, details: [] };
    return { mean: mean(prices), std: std(prices), details };
  }

  surface(strikes, expiries, S, r = 0.045, q = 0) {
    const surfaces = this.models.filter(trained).map((m) => m.surface(strikes, expiries, S, r, q));
    const NE = expiries.length;
    const NS = strikes.length;
    if (!surfaces.length) return zeros2d(NE, NS);
    return meanSurfaces(surfaces, NE, NS);
  }

  greeks(S, K, T, r = 0.045, q = 0) {
    const list = this.models.filter(trained).map((m) => m.greeks(S, K, T, r, q));
    if (!list.length) return { delta: 0, gamma: 0, vega: 0, theta: 0 };
    const keys = Object.keys(list[0]);
    const out = {};
    for (const key of keys) out[key] = mean(list.map((g) => g[key]));
    return out;
  }
}

function meanSurfaces(surfaces, NE, NS) {
  const out = zeros2d(NE, NS);
  for (let i = 0; i < NE; i++) {
    for (let j = 0; j < NS; j++) {
      let s = 0;
      for (const surf of surfaces) s += surf[i][j];
      out[i][j] = s / surfaces.length;
    }
  }
  return out;
}

module.exports = { EnsemblePricer };
