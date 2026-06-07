# NN-Mind

Neural network options pricer & testing framework.

The idea: instead of using closed-form models (BS, Heston, etc.) to price options, train neural networks on market data and let them learn the surface. Compare performance, find where NNs capture things traditional models miss, and build an options-language model that understands the Greeks, the skew, and the smile from data rather than equations.

## Structure

- src/ - Core training, pricing, and evaluation code
- data/ - Market data loaders and feature engineering
- models/ - Neural network architectures (MLP, LSTM, Transformer for options)
- notebooks/ - Exploration and visualization
- tests/ - Unit tests and backtesting against traditional models

## Status

Created 2026-06-07. Bare repo - nothing built yet.
