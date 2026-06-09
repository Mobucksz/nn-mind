// Black-Scholes and Binomial option pricers (closed-form, no training).
// Port of models/traditional.py.

const { OptionPricer } = require('./base');
const { normCdf, normPdf, brentq } = require('../stats');
const { expiryT, zeros2d } = require('../util');

class BlackScholesPricer extends OptionPricer {
  constructor(name = 'bs') {
    super(name);
  }

  _d1(S, K, T, r, q, sigma = 0.25) {
    return (Math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
  }

  _d2(d1, T, sigma = 0.25) {
    return d1 - sigma * Math.sqrt(T);
  }

  price(S, K, T, r = 0.045, q = 0, sigma = 0.25) {
    if (T <= 0) return Math.max(S - K, 0);
    const d1 = this._d1(S, K, T, r, q, sigma);
    const d2 = this._d2(d1, T, sigma);
    return S * Math.exp(-q * T) * normCdf(d1) - K * Math.exp(-r * T) * normCdf(d2);
  }

  greeks(S, K, T, r = 0.045, q = 0, sigma = 0.25) {
    if (T <= 0) return { delta: S > K ? 1 : 0, gamma: 0, vega: 0, theta: 0 };
    const d1 = this._d1(S, K, T, r, q, sigma);
    const d2 = this._d2(d1, T, sigma);
    const delta = Math.exp(-q * T) * normCdf(d1);
    const gamma = (Math.exp(-q * T) * normPdf(d1)) / (S * sigma * Math.sqrt(T));
    const vega = (S * Math.exp(-q * T) * normPdf(d1) * Math.sqrt(T)) / 100; // per 1% vol move
    const theta =
      (-(S * Math.exp(-q * T) * normPdf(d1) * sigma) / (2 * Math.sqrt(T)) -
        r * K * Math.exp(-r * T) * normCdf(d2) +
        q * S * Math.exp(-q * T) * normCdf(d1)) /
      365;
    return { delta, gamma, vega, theta };
  }

  surface(strikes, expiries, S, r = 0.045, q = 0, sigma = 0.25) {
    const exp = expiries.map(expiryT);
    const NE = exp.length;
    const NS = strikes.length;
    const result = zeros2d(NE, NS);
    for (let i = 0; i < NE; i++) {
      for (let j = 0; j < NS; j++) {
        result[i][j] = this.price(S, strikes[j], exp[i], r, q, sigma);
      }
    }
    return result;
  }

  impliedVol(S, K, T, marketPrice, r = 0.045, q = 0) {
    if (T <= 0) return 0;
    try {
      return brentq((sigma) => this.price(S, K, T, r, q, sigma) - marketPrice, 0.001, 2.0);
    } catch {
      return 0.25; // fallback to ATM guess
    }
  }

  attention() {
    return null;
  }
}

class BinomialPricer extends OptionPricer {
  constructor(name = 'binomial', steps = 100) {
    super(name);
    this.steps = steps;
  }

  price(S, K, T, r = 0.045, q = 0, sigma = 0.25, isCall = true) {
    if (T <= 0) return Math.max(isCall ? S - K : K - S, 0);
    const n = this.steps;
    const dt = T / n;
    const u = Math.exp(sigma * Math.sqrt(dt));
    const d = 1 / u;
    const p = (Math.exp((r - q) * dt) - d) / (u - d);
    const disc = Math.exp(-r * dt);

    // Terminal node values.
    let V = new Array(n + 1);
    for (let i = 0; i <= n; i++) {
      const ST = S * Math.pow(u, i) * Math.pow(d, n - i);
      V[i] = Math.max(isCall ? ST - K : K - ST, 0);
    }

    // Backward induction with early-exercise (American). Indexing mirrors
    // models/traditional.py exactly (V_new[j] = disc*(p*V[j] + (1-p)*V[j+1]))
    // so JS output is numerically identical to the original.
    for (let step = n - 1; step >= 0; step--) {
      const Vn = new Array(step + 1);
      for (let j = 0; j <= step; j++) {
        const cont = disc * (p * V[j] + (1 - p) * V[j + 1]);
        const St = S * Math.pow(u, j) * Math.pow(d, step - j);
        const exercise = isCall ? St - K : K - St;
        Vn[j] = Math.max(cont, exercise);
      }
      V = Vn;
    }
    return V[0];
  }

  surface(strikes, expiries, S, r = 0.045, q = 0, sigma = 0.25) {
    const exp = expiries.map(expiryT);
    const NE = exp.length;
    const NS = strikes.length;
    const result = zeros2d(NE, NS);
    for (let i = 0; i < NE; i++) {
      for (let j = 0; j < NS; j++) {
        result[i][j] = this.price(S, strikes[j], exp[i], r, q, sigma);
      }
    }
    return result;
  }
}

module.exports = { BlackScholesPricer, BinomialPricer };
