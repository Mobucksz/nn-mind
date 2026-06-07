// Small numeric helpers - the bits of numpy the port actually uses.

function mean(a) {
  if (!a.length) return 0;
  let s = 0;
  for (const x of a) s += x;
  return s / a.length;
}

function median(a) {
  if (!a.length) return 0;
  const b = [...a].sort((x, y) => x - y);
  const m = Math.floor(b.length / 2);
  return b.length % 2 ? b[m] : (b[m - 1] + b[m]) / 2;
}

function std(a) {
  if (!a.length) return 0;
  const m = mean(a);
  let s = 0;
  for (const x of a) s += (x - m) * (x - m);
  return Math.sqrt(s / a.length); // population std, matches numpy default
}

function min(a) {
  let m = Infinity;
  for (const x of a) if (x < m) m = x;
  return m;
}

function max(a) {
  let m = -Infinity;
  for (const x of a) if (x > m) m = x;
  return m;
}

// T value for an expiry entry (dict {T:...} or bare number).
function expiryT(e) {
  return typeof e === 'object' && e !== null ? e.T : e;
}

// Zero matrix [rows][cols].
function zeros2d(rows, cols) {
  return Array.from({ length: rows }, () => new Array(cols).fill(0));
}

module.exports = { mean, median, std, min, max, expiryT, zeros2d };
