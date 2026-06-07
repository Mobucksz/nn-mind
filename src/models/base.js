// Abstract base class for all option pricers. Port of models/base.py.

const { zeros2d } = require('../util');

class OptionPricer {
  constructor(name) {
    this.name = name;
  }

  // Single option price.
  price(S, K, T, r = 0.045, q = 0) {
    throw new Error('not implemented');
  }

  // [NE][NS] price matrix given arrays of strikes and expiries.
  surface(strikes, expiries, S, r = 0.045, q = 0) {
    throw new Error('not implemented');
  }

  // { delta, gamma, vega, theta }.
  greeks(S, K, T, r = 0.045, q = 0) {
    return { delta: 0, gamma: 0, vega: 0, theta: 0 };
  }

  async save(p) {}
  async load(p) {}

  // Attention map for explainability, if applicable.
  attention(S, K, T) {
    return null;
  }
}

module.exports = { OptionPricer, zeros2d };
