# Data Pipeline

Connects to IBKR paper account via TWS/Gateway (same as OpVis) to record
option chain snapshots for training neural network pricers.

## How it works

1. **IBC** auto-logs into TWS Paper at 8:20 CT each trading day
2. `record_session.py` connects on port 7497 and records snapshots through market close
3. Files land in `recordings/` with full IV, greeks, OI, and volume data
4. `record_daily.ps1` is called by Windows Task Scheduler (Mon-Fri 8:20 CT)

## Files

- `record_session.py` — Copies the same IBKR recorder from OpVis, tuned for ML training
- `record_daily.ps1` — PowerShell launcher that starts TWS paper if needed, then records
- `loader.py` (TODO) — Load recorded sessions into numpy/pytorch tensors
- `features.py` (TODO) — Build feature vectors from chain snapshots

## IBKR Config

Uses the same IBC config as OpVis:
- Config: `C:\IBC\config.paper.ini`
- User: Moebuckszz
- Port: 7497
- TWS: C:\Jts

Recording data at `recordings/` dir for training, not just replay.
