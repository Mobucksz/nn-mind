// Model registry and factory for nn-mind. Port of models/__init__.py.

const { OptionPricer } = require('./base');
const { BlackScholesPricer, BinomialPricer } = require('./traditional');
const { MLPPricer } = require('./mlp');
const { LSTMPricer } = require('./lstm');
const { TransformerPricer } = require('./transformer');
const { EnsemblePricer } = require('./ensemble');

const REGISTRY = {
  bs: BlackScholesPricer,
  binomial: BinomialPricer,
  mlp: MLPPricer,
  lstm: LSTMPricer,
  transformer: TransformerPricer,
  ensemble: EnsemblePricer,
};

// Factory: instantiate a pricer by model type string.
function createPricer(modelType) {
  const Cls = REGISTRY[modelType];
  if (!Cls) {
    throw new Error(
      `Unknown model type '${modelType}'. Available: ${Object.keys(REGISTRY).join(', ')}`
    );
  }
  return new Cls();
}

module.exports = {
  OptionPricer,
  BlackScholesPricer,
  BinomialPricer,
  MLPPricer,
  LSTMPricer,
  TransformerPricer,
  EnsemblePricer,
  REGISTRY,
  createPricer,
};
