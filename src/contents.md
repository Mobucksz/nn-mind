# NN-Mind - Project Map

## src/

- train.py - Training loop for option pricing NNs
- pricer.py - Inference API: price an option from features
- data.py - IBKR/YFinance data loaders, feature engineering
- compare.py - Compare NN-priced vs BS/Heston/SABR surface
- mind.py - The language model layer: explain what the NN sees

## models/

- mlp.py - Simple feedforward for IV surface regression
- lstm.py - Sequence model for vol term structure
- transformer.py - Attention-based surface encoder
- ensemble.py - Stack multiple NNs for uncertainty

## data/

- loader.py - Fetch options chains (IBKR during market, yfinance fallback)
- features.py - Build feature vectors from chain snapshots
- augment.py - Synthetic data augmentation
- surface.py - Convert chain snapshots to training grids

## notebooks/

- 01-data-exploration.ipynb - Visualize the option surface
- 02-mlp-baseline.ipynb - Train a simple MLP, compare to BS
- 03-attention-surface.ipynb - Transformer on the full surface
- 04-nn-vs-models.ipynb - Systematic comparison across regimes
