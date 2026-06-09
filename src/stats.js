// Statistics helpers - replaces scipy.stats.norm / scipy.optimize.brentq.
// Pure JS, no dependencies. Used by the closed-form pricers.

// Standard normal PDF.
function normPdf(x) {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

// erf via Abramowitz & Stegun 7.1.26 (max abs error ~1.5e-7).
function erf(x) {
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
      0.254829592) *
      t *
      Math.exp(-x * x);
  return sign * y;
}

// Standard normal CDF.
function normCdf(x) {
  return 0.5 * (1 + erf(x / Math.SQRT2));
}

// Brent's method root finder - replaces scipy.optimize.brentq.
// Returns the root of f in [a, b], or throws if not bracketed / no convergence.
function brentq(f, a, b, tol = 1e-12, maxIter = 100) {
  let fa = f(a);
  let fb = f(b);
  if (fa * fb > 0) throw new Error('brentq: root not bracketed');
  if (Math.abs(fa) < Math.abs(fb)) {
    [a, b] = [b, a];
    [fa, fb] = [fb, fa];
  }
  let c = a;
  let fc = fa;
  let d = b - a;
  let mflag = true;
  for (let i = 0; i < maxIter; i++) {
    if (fb === 0 || Math.abs(b - a) < tol) return b;
    let s;
    if (fa !== fc && fb !== fc) {
      // Inverse quadratic interpolation.
      s =
        (a * fb * fc) / ((fa - fb) * (fa - fc)) +
        (b * fa * fc) / ((fb - fa) * (fb - fc)) +
        (c * fa * fb) / ((fc - fa) * (fc - fb));
    } else {
      // Secant.
      s = b - (fb * (b - a)) / (fb - fa);
    }
    const cond =
      s < (3 * a + b) / 4 ||
      s > b ||
      (mflag && Math.abs(s - b) >= Math.abs(b - c) / 2) ||
      (!mflag && Math.abs(s - b) >= Math.abs(c - d) / 2) ||
      (mflag && Math.abs(b - c) < tol) ||
      (!mflag && Math.abs(c - d) < tol);
    if (cond) {
      s = (a + b) / 2;
      mflag = true;
    } else {
      mflag = false;
    }
    const fs = f(s);
    d = c;
    c = b;
    fc = fb;
    if (fa * fs < 0) {
      b = s;
      fb = fs;
    } else {
      a = s;
      fa = fs;
    }
    if (Math.abs(fa) < Math.abs(fb)) {
      [a, b] = [b, a];
      [fa, fb] = [fb, fa];
    }
  }
  return b;
}

module.exports = { normPdf, normCdf, erf, brentq };
