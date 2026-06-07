# Data Directory

This is where all saved data lives — next to the app, easy to find.

## Structure

```
data/
├── models/          Trained model checkpoints (.pt files)
│   ├── mlp.latest.pt       Latest trained MLP
│   ├── mlp.v1.pt           Previous version (auto-archived)
│   ├── mlp.v2.pt           Older version
│   ├── lstm.latest.pt
│   ├── transformer.latest.pt
│   └── ensemble.latest.pt
├── sessions/        IBKR recorded session files (optional local copies)
├── logs/            Training run logs
└── README.md        This file
```

## Storage estimates

Each trained model checkpoint is roughly:
- MLP: ~50 KB
- LSTM: ~200 KB
- Transformer: ~500 KB
- Ensemble: ~1 MB

Training logs are negligible. Session data from one trading day of SPY options (131 snapshots, 21 strikes, 7 expiries) is about 1.5 MB.

So the entire data directory after months of use would be under 100 MB.

## Checkpoint versioning

Every time you train a model, the previous checkpoint gets archived:
- `mlp.latest.pt` -> `mlp.v1.pt` -> `mlp.v2.pt` -> etc.
- The app always loads the `.latest.pt` version on startup.
- Old versions are kept so you can roll back if needed.
