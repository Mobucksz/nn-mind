// Persist tfjs LayersModels with the pure-JS @tensorflow/tfjs build.
// The 'file://' IOHandler only ships with tfjs-node, so we serialize the
// model artifacts (topology + weight specs + weights) ourselves into a single
// JSON file alongside any extra metadata (scalers, training flags, grid).

const fs = require('fs');
const tf = require('@tensorflow/tfjs');

async function saveModel(model, filePath, extra = {}) {
  let artifacts = null;
  await model.save(
    tf.io.withSaveHandler(async (a) => {
      artifacts = a;
      return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
    })
  );
  const weightBuf = Buffer.from(artifacts.weightData);
  const payload = {
    modelTopology: artifacts.modelTopology,
    weightSpecs: artifacts.weightSpecs,
    weightDataB64: weightBuf.toString('base64'),
    extra,
  };
  fs.writeFileSync(filePath, JSON.stringify(payload));
}

async function loadModel(filePath) {
  const payload = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  const buf = Buffer.from(payload.weightDataB64, 'base64');
  // Slice to the exact backing range so we hand tfjs a clean ArrayBuffer.
  const weightData = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const model = await tf.loadLayersModel(
    tf.io.fromMemory({
      modelTopology: payload.modelTopology,
      weightSpecs: payload.weightSpecs,
      weightData,
    })
  );
  return { model, extra: payload.extra || {} };
}

module.exports = { saveModel, loadModel };
