// app.js - nn-mind controller. Talks to the Electron main process over the
// window.nn IPC bridge (same contract as the old fetch('/api/...') calls) and
// drives the Surface3D engine. All option pricing / training stays in the
// Node/TF.js backend; this file only orchestrates the UI.

import { Surface3D, viridis, diverging } from './surface3d.js';

const $ = (s) => document.querySelector(s);
const api = (action, payload) => window.nn.invoke(action, payload);
const fmt = (x, d = 4) => (x == null || Number.isNaN(x)) ? '-' : Number(x).toFixed(d);

// ---- state -----------------------------------------------------------------
let MODELS = [];
let SESSION = null;
let MODE = 'price';            // 'price' | 'diff' | 'attention'
let surf = null;
let lastPrice = null;          // {K, T, price, model} for the surface marker

// ---- colorbar legend -------------------------------------------------------
function paintColorbar(canvas, fn, diverge) {
  const ctx = canvas.getContext('2d');
  const h = canvas.height, w = canvas.width;
  for (let y = 0; y < h; y++) {
    const t = 1 - y / (h - 1);
    const [r, g, b] = diverge ? fn(t * 2 - 1) : fn(t);
    ctx.fillStyle = `rgb(${r | 0},${g | 0},${b | 0})`;
    ctx.fillRect(0, y, w, 1);
  }
}

// ---- model selects ---------------------------------------------------------
function modelOptions() {
  return MODELS.map((m) => `<option value="${m.name}">${m.name}</option>`).join('');
}
function fillSelects() {
  const opts = modelOptions();
  for (const id of ['#vModel', '#cmpA', '#cmpB', '#inModel']) {
    const prev = $(id).value;
    $(id).innerHTML = opts;
    if (prev) $(id).value = prev;
  }
  if (!$('#vModel').value) $('#vModel').value = 'bs';
  if (!$('#cmpA').value) $('#cmpA').value = 'binomial';
  if (!$('#cmpB').value) $('#cmpB').value = 'bs';
  if (!$('#inModel').value) $('#inModel').value = 'bs';
}

function modelBadge(m) {
  if (m.type === 'bs' || m.type === 'binomial') return '<span class="badge cf">closed-form</span>';
  return m.trained ? '<span class="badge trained">trained</span>' : '<span class="badge untrained">untrained</span>';
}

function renderModelList() {
  const host = $('#modelList'); host.innerHTML = '';
  for (const m of MODELS) {
    const isNN = !(m.type === 'bs' || m.type === 'binomial');
    const el = document.createElement('div'); el.className = 'model';
    el.innerHTML = `
      <div class="top">
        <span class="name">${m.name}</span>
        ${modelBadge(m)}
        ${m.has_attention ? '<span class="badge attn">attention</span>' : ''}
      </div>
      <div class="desc">${m.description || ''}</div>
      ${isNN ? `
        <div class="row">
          <input class="mono" type="number" value="60" min="1" max="500" id="ep-${m.name}">
          <span class="hint">epochs</span><span style="flex:1"></span>
          <button data-train="${m.name}">${m.trained ? 'Retrain' : 'Train'}</button>
        </div>
        <div class="progress" style="display:none"><i></i></div>
        <div class="hint mono" data-log="${m.name}"></div>` : ''}`;
    host.appendChild(el);
  }
  host.querySelectorAll('[data-train]').forEach((b) =>
    b.addEventListener('click', () => trainModel(b.dataset.train)));
}

async function trainModel(name) {
  const card = $(`[data-train="${name}"]`).closest('.model');
  const bar = card.querySelector('.progress'), fill = bar.querySelector('i');
  const log = card.querySelector('[data-log]'), btn = card.querySelector('[data-train]');
  const epochs = parseInt($(`#ep-${name}`).value, 10) || 60;
  const multi = !!($('#multiTrain') && $('#multiTrain').checked);
  btn.disabled = true; bar.style.display = 'block'; log.textContent = multi ? 'queued (all sessions)...' : 'queued...';
  try {
    const { job_id } = await api('train', { model_type: name, epochs, multi });
    await new Promise((resolve, reject) => {
      const tick = async () => {
        try {
          const j = await api('train-status', { job_id });
          fill.style.width = (j.progress || 0) + '%';
          const tag = j.multi ? `[${j.nsessions} sessions] ` : '';
          log.textContent = `${tag}${j.status} - epoch ${j.current_epoch}/${j.epochs}` +
            (j.loss != null ? ` - loss ${fmt(j.loss, 4)} - val ${fmt(j.val_loss, 4)}` : '');
          if (j.status === 'completed') return resolve();
          if (j.status === 'failed') return reject(new Error(j.error || 'training failed'));
          setTimeout(tick, 400);
        } catch (e) { reject(e); }
      };
      tick();
    });
    log.textContent = 'done - ' + log.textContent;
    await loadModels();
    if ((MODE === 'price' && $('#vModel').value === name) || MODE === 'diff' || MODE === 'attention') renderView();
  } catch (e) {
    log.innerHTML = `<span class="err">${e.message}</span>`;
  } finally { btn.disabled = false; }
}

// ---- session ---------------------------------------------------------------
async function loadSession() {
  try {
    SESSION = await api('latest-session');
    $('#sbSymbol').textContent = SESSION.meta.symbol || '-';
    $('#sbDate').textContent = SESSION.meta.sessionDate || SESSION.meta.date || '-';
    $('#sbSpot').textContent = fmt(SESSION.spot_avg, 2);
    $('#sessionSub').textContent = `${SESSION.meta.source || 'recording'} - ${SESSION.nframes} frames`;
    const e = SESSION.expiries;
    $('#sessionStats').innerHTML = `
      <div class="stat"><div class="v">${SESSION.strikes.length}</div><div class="l">strikes</div></div>
      <div class="stat"><div class="v">${e.length}</div><div class="l">expiries</div></div>
      <div class="stat"><div class="v mono">${fmt(SESSION.spot_min, 1)}-${fmt(SESSION.spot_max, 1)}</div><div class="l">spot range</div></div>
      <div class="stat"><div class="v mono">${e.length ? e[0].days + '-' + e[e.length - 1].days + 'd' : '-'}</div><div class="l">expiry span</div></div>`;
  } catch (e) {
    $('#sessionSub').innerHTML = `<span class="err">${e.message}</span>`;
  }
}

async function loadModels() {
  MODELS = await api('models');
  renderModelList();
  fillSelects();
}

// ---- the 3D view -----------------------------------------------------------
function setLegend(label, lo, hi, diverge) {
  $('#legendLabel').textContent = label;
  paintColorbar($('#legendBar'), diverge ? diverging : viridis, diverge);
  if (diverge) {
    const m = Math.max(Math.abs(lo), Math.abs(hi));
    $('#legHi').textContent = '+' + fmt(m, 1); $('#legMid').textContent = '0'; $('#legLo').textContent = '-' + fmt(m, 1);
  } else {
    $('#legHi').textContent = fmt(hi, 1); $('#legMid').textContent = fmt((lo + hi) / 2, 1); $('#legLo').textContent = fmt(lo, 1);
  }
}

async function renderView() {
  $('#viewErr').textContent = '';
  $('#viewTitle').textContent = 'rendering...';
  // Show only the selector group that belongs to the active mode.
  $('#priceCtl').style.display = MODE === 'price' ? 'flex' : 'none';
  $('#diffCtl').style.display = MODE === 'diff' ? 'flex' : 'none';
  $('#attnCtl').style.display = MODE === 'attention' ? 'flex' : 'none';
  try {
    if (MODE === 'price') {
      const model = $('#vModel').value;
      const s = await api('surface', { model });
      const { lo, hi } = surf.setData({
        matrix: s.prices, strikes: s.strikes, expiries: s.expiries,
        valueLabel: 'CALL $', mode: 'sequential', S: s.S,
      });
      setLegend('call price ($)', lo, hi, false);
      $('#viewTitle').textContent = `${model} - call-price surface - S=${fmt(s.S, 2)}`;
      if (lastPrice && lastPrice.model === model) surf.setMarker(lastPrice.K, lastPrice.T, lastPrice.price);
    } else if (MODE === 'diff') {
      const a = $('#cmpA').value, b = $('#cmpB').value;
      if (a === b) { $('#viewErr').textContent = 'Pick two different models.'; $('#viewTitle').textContent = 'model comparison'; return; }
      const r = await api('compare', { models: `${a},${b}` });
      let diff = r.differences[`${a}-${b}`];
      if (!diff && r.differences[`${b}-${a}`]) diff = r.differences[`${b}-${a}`].map((row) => row.map((v) => -v));
      if (!diff) throw new Error('no difference available for this pair');
      const exp = SESSION ? SESSION.expiries : r.expiries.map((d) => ({ date: d }));
      const { lo, hi, maxAbs } = surf.setData({
        matrix: diff, strikes: r.strikes, expiries: exp,
        valueLabel: `${a}-${b}`, mode: 'diverging', S: r.S,
      });
      setLegend(`${a} - ${b}  ($)`, lo, hi, true);
      $('#viewTitle').textContent = `${a} - ${b} - price divergence - max|diff| ${fmt(maxAbs, 2)}`;
    } else if (MODE === 'attention') {
      if (!SESSION) throw new Error('no session data');
      const S = SESSION.spot_avg;
      const mid = Math.floor(SESSION.strikes.length / 2);
      const K = lastPrice ? lastPrice.K : SESSION.strikes[mid];
      const T = lastPrice ? lastPrice.T : (SESSION.expiries[0] ? SESSION.expiries[0].T : 0.08);
      const r = await api('attention', { S, K, T });
      // Real attention is an NE x NS map over the actual strike/expiry grid.
      const m = r.attention;
      const { lo, hi } = surf.setData({
        matrix: m, strikes: SESSION.strikes, expiries: SESSION.expiries,
        valueLabel: 'attn', mode: 'sequential', S,
      });
      setLegend('attention weight', lo, hi, false);
      $('#viewTitle').textContent = `transformer attention from K=${fmt(K, 0)} T=${fmt(T, 3)} - where the model looks`;
      // Mark the query node so you can see what it attends away from.
      const nearest = (arr, t) => arr.reduce((b, v, i) => Math.abs(v - t) < Math.abs(arr[b] - t) ? i : b, 0);
      const si = nearest(SESSION.strikes, K);
      const ei = nearest(SESSION.expiries.map((e) => e.T != null ? e.T : (e.days || 0) / 365), T);
      surf.setMarker(K, T, m[ei][si]);
    }
  } catch (e) {
    $('#viewErr').textContent = e.message;
    $('#viewTitle').textContent = MODE === 'price' ? 'call-price surface' : MODE === 'diff' ? 'model comparison' : 'attention map';
  }
}

// ---- pricer ----------------------------------------------------------------
async function priceOption() {
  $('#priceErr').textContent = '';
  const K = +$('#inK').value, T = +$('#inT').value, model = $('#inModel').value;
  const body = { S: +$('#inS').value, K, T, model_type: model };
  try {
    const r = await api('infer', body);
    $('#priceVal').textContent = fmt(r.price, 3);
    $('#priceModel').textContent = '- ' + r.model;
    $('#gDelta').textContent = fmt(r.delta, 4);
    $('#gGamma').textContent = fmt(r.gamma, 5);
    $('#gVega').textContent = fmt(r.vega, 4);
    $('#gTheta').textContent = fmt(r.theta, 4);
    lastPrice = { K, T, price: r.price, model };
    // Drop a marker on the surface when the priced model is the one on screen.
    if (MODE === 'price' && $('#vModel').value === model) surf.setMarker(K, T, r.price);
  } catch (e) {
    $('#priceErr').textContent = e.message;
    $('#priceVal').textContent = '-';
  }
}

// ---- wiring ----------------------------------------------------------------
function setMode(mode) {
  MODE = mode;
  document.querySelectorAll('#modeSeg button').forEach((b) =>
    b.classList.toggle('active', b.dataset.mode === mode));
  renderView();
}

function init() {
  if (!window.nn) { console.error('init: window.nn bridge missing'); return; }
  surf = new Surface3D($('#viewport'), $('#hoverTip'));

  document.querySelectorAll('#modeSeg button').forEach((b) =>
    b.addEventListener('click', () => setMode(b.dataset.mode)));
  $('#vModel').addEventListener('change', renderView);
  $('#cmpA').addEventListener('change', renderView);
  $('#cmpB').addEventListener('change', renderView);
  $('#attnRefresh').addEventListener('click', renderView);
  $('#priceBtn').addEventListener('click', priceOption);
  $('#wireToggle').addEventListener('change', (e) => surf.setOption('wireframe', e.target.checked));
  $('#heightSlider').addEventListener('input', (e) => surf.setOption('heightScale', +e.target.value));
  $('#resetView').addEventListener('click', () => surf.resetView());

  (async () => {
    try {
      await Promise.all([loadSession(), loadModels()]);
      try {
        const ss = await api('sessions');
        const el = $('#multiCount');
        if (el) el.textContent = `(${ss.length} available)`;
      } catch { /* non-fatal */ }
      await renderView();
      await priceOption();
      console.log('init: 3D dashboard ready');
    } catch (e) { console.error('init failed:', e.message); }
  })();
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
else init();
