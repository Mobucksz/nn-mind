// Load IBKR-recorded option chain sessions for NN training.
// Port of data/loader.py. Reads the same session files OpVis records
// (shared data pipeline), plus nn-mind's own local recordings.

const fs = require('fs');
const path = require('path');

// OpVis records sessions here.
const OPVIS_SESSIONS_DIR = 'C:\\Users\\Bryce\\3d-option-application\\public\\data\\sessions';
const OPVIS_MANIFEST = 'C:\\Users\\Bryce\\3d-option-application\\public\\data\\manifest.json';

// nn-mind keeps its own recordings here (data/recordings, sibling of src/).
const LOCAL_RECORDINGS = path.join(__dirname, '..', 'data', 'recordings');

function exists(p) {
  try {
    fs.accessSync(p);
    return true;
  } catch {
    return false;
  }
}

// Strip a UTF-8 BOM if present (Python read this with utf-8-sig).
function readJson(p) {
  let text = fs.readFileSync(p, 'utf8');
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
  return JSON.parse(text);
}

function listSessions() {
  const sessions = [];

  // From OpVis manifest.
  if (exists(OPVIS_MANIFEST)) {
    try {
      const man = readJson(OPVIS_MANIFEST);
      for (const s of man.sessions || []) {
        let fp = path.join(OPVIS_SESSIONS_DIR, s.file || '');
        if ((s.file || '').includes('sessions/')) {
          fp = path.join(OPVIS_SESSIONS_DIR, s.file.replace('sessions/', ''));
        } else if (exists(path.join(path.dirname(OPVIS_SESSIONS_DIR), s.file))) {
          fp = path.join(path.dirname(OPVIS_SESSIONS_DIR), s.file);
        }
        if (exists(fp)) {
          sessions.push({
            date: s.date,
            symbol: s.symbol,
            snapshots: s.snapshots,
            path: fp,
            source: 'opvis',
          });
        }
      }
    } catch {
      // Ignore a malformed manifest; fall through to local recordings.
    }
  }

  // From local recordings.
  if (exists(LOCAL_RECORDINGS)) {
    const files = fs
      .readdirSync(LOCAL_RECORDINGS)
      .filter((f) => f.startsWith('session-') && f.endsWith('.json'))
      .sort();
    for (const f of files) {
      sessions.push({
        date: f.replace(/^session-/, '').replace(/\.json$/, ''),
        symbol: 'SPY',
        snapshots: null,
        path: path.join(LOCAL_RECORDINGS, f),
        source: 'local',
      });
    }
  }

  // Most recent first.
  return sessions.sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : 0));
}

// Reshape a flat array into [NE][NS]; missing entries become 0.
function reshape2d(flat, NE, NS) {
  const out = [];
  for (let i = 0; i < NE; i++) {
    const row = [];
    for (let j = 0; j < NS; j++) {
      const idx = i * NS + j;
      row.push(idx < flat.length ? Number(flat[idx]) || 0 : 0);
    }
    out.push(row);
  }
  return out;
}

// Load a full recorded session into nested arrays.
// Returns null if the session has no usable data.
function loadSession(p) {
  const raw = readJson(p);

  const meta = raw.meta || {};
  const strikes = (raw.strikes || []).map(Number);
  const expiries = raw.expiries || [];
  const snapshots = raw.snapshots || [];

  if (!snapshots.length || !strikes.length) return null;

  const NE = expiries.length;
  const NS = strikes.length;
  const NF = snapshots.length;

  // data[fi][ei][si][feat]; features: iv, delta, gamma, vega, theta, oi, volume.
  const data = [];
  const spot = [];
  const timestamps = [];

  for (let fi = 0; fi < NF; fi++) {
    const snap = snapshots[fi];
    timestamps.push(snap.t || '');
    spot.push(Number(snap.underlying) || 0);

    const iv = reshape2d(snap.iv || [], NE, NS);
    const delta = reshape2d(snap.callDelta || [], NE, NS);
    const gamma = reshape2d(snap.gamma || [], NE, NS);
    const vega = reshape2d(snap.vega || [], NE, NS);
    const theta = reshape2d(snap.callTheta || [], NE, NS);
    const callOI = reshape2d(snap.callOI || [], NE, NS);
    const putOI = reshape2d(snap.putOI || [], NE, NS);
    const callVol = reshape2d(snap.callVol || [], NE, NS);
    const putVol = reshape2d(snap.putVol || [], NE, NS);

    const frame = [];
    for (let ei = 0; ei < NE; ei++) {
      const erow = [];
      for (let si = 0; si < NS; si++) {
        erow.push([
          iv[ei][si],
          delta[ei][si],
          gamma[ei][si],
          vega[ei][si],
          theta[ei][si],
          callOI[ei][si] + putOI[ei][si],
          callVol[ei][si] + putVol[ei][si],
        ]);
      }
      frame.push(erow);
    }
    data.push(frame);
  }

  return {
    meta,
    strikes,
    expiries,
    spot,
    timestamps,
    data, // [NF][NE][NS][7]
    nframes: NF,
    nexp: NE,
    nstrikes: NS,
  };
}

function latestSession() {
  const sessions = listSessions();
  if (!sessions.length) return null;
  return loadSession(sessions[0].path);
}

module.exports = { listSessions, loadSession, latestSession };
