// Standalone smoke test for the Node compute core (no Electron).
// Run: node test/smoke.js
/* eslint-disable no-console */

const { handle } = require('../src/handlers');
const worker = require('../src/worker');

function assert(cond, msg) {
  if (!cond) throw new Error('ASSERT FAILED: ' + msg);
}

async function poll(jobId, label) {
  for (let i = 0; i < 2000; i++) {
    const { body } = await handle('train-status', { job_id: jobId });
    if (body.status === 'completed') {
      console.log(`  ${label}: completed, final loss=${body.history.at(-1).loss.toFixed(5)}`);
      return body;
    }
    if (body.status === 'failed') throw new Error(`${label} training failed: ${body.error}`);
    await new Promise((r) => setTimeout(r, 25));
  }
  throw new Error(`${label} timed out`);
}

(async () => {
  console.log('== latest-session ==');
  let r = await handle('latest-session', {});
  assert(r.status === 200, 'session status');
  console.log(`  symbol=${r.body.meta.symbol} strikes=${r.body.strikes.length} expiries=${r.body.expiries.length} frames=${r.body.nframes} spotAvg=${r.body.spot_avg.toFixed(2)}`);

  console.log('== models ==');
  r = await handle('models', {});
  console.log('  ' + r.body.map((m) => `${m.name}(${m.trained ? 'T' : '-'})`).join(' '));

  console.log('== surface bs / binomial ==');
  for (const m of ['bs', 'binomial']) {
    r = await handle('surface', { model: m });
    assert(r.status === 200, `${m} surface status`);
    const flat = r.body.prices.flat();
    console.log(`  ${m}: ${r.body.prices.length}x${r.body.prices[0].length}, range ${Math.min(...flat).toFixed(2)}..${Math.max(...flat).toFixed(2)}, S=${r.body.S.toFixed(2)}`);
  }

  console.log('== infer bs ==');
  r = await handle('infer', { S: 530, K: 525, T: 0.0822, model_type: 'bs' });
  console.log(`  price=${r.body.price.toFixed(4)} delta=${r.body.delta.toFixed(4)} gamma=${r.body.gamma.toFixed(5)} vega=${r.body.vega.toFixed(4)} theta=${r.body.theta.toFixed(5)}`);
  assert(r.body.price > 0, 'bs price positive');

  console.log('== compare bs vs binomial ==');
  r = await handle('compare', { models: 'bs,binomial' });
  assert(r.body.differences['bs-binomial'], 'diff present');
  const dflat = r.body.differences['bs-binomial'].flat();
  console.log(`  max|diff|=${Math.max(...dflat.map(Math.abs)).toFixed(4)}`);

  console.log('== train mlp (8 epochs) ==');
  r = await handle('train', { model_type: 'mlp', epochs: 8 });
  await poll(r.body.job_id, 'mlp');
  r = await handle('surface', { model: 'mlp' });
  assert(r.status === 200, 'mlp surface after train');
  console.log(`  mlp surface ok, [0][0]=${r.body.prices[0][0].toFixed(4)}`);
  r = await handle('infer', { S: 530, K: 525, T: 0.0822, model_type: 'mlp' });
  console.log(`  mlp infer price=${r.body.price.toFixed(4)} delta=${r.body.delta.toFixed(4)}`);

  console.log('== train transformer (5 epochs) ==');
  r = await handle('train', { model_type: 'transformer', epochs: 5 });
  await poll(r.body.job_id, 'transformer');
  r = await handle('surface', { model: 'transformer' });
  const tflat = r.body.prices.flat();
  console.log(`  transformer surface range ${Math.min(...tflat).toFixed(3)}..${Math.max(...tflat).toFixed(3)}`);
  r = await handle('attention', { S: 530, K: 525, T: 0.0822 });
  console.log(`  attention map ${r.body.attention.length}x${r.body.attention[0].length}`);

  console.log('== train lstm (5 epochs) ==');
  r = await handle('train', { model_type: 'lstm', epochs: 5 });
  await poll(r.body.job_id, 'lstm');

  console.log('== compare mlp vs bs (trained) ==');
  r = await handle('compare', { models: 'mlp,bs' });
  console.log('  keys: ' + r.body.models.join(','));

  console.log('== reload from disk (save/load roundtrip) ==');
  await worker.loadSavedModels();
  r = await handle('models', {});
  console.log('  ' + r.body.map((m) => `${m.name}(${m.trained ? 'T' : '-'})`).join(' '));

  console.log('\nALL SMOKE TESTS PASSED');
})().catch((e) => {
  console.error('\nSMOKE TEST FAILED:', e);
  process.exit(1);
});
