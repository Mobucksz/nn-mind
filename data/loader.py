"""
Load IBKR-recorded option chain sessions for NN training.

Reads from the same session files that OpVis records (shared data pipeline).
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path

# OpVis records sessions here
OPVIS_SESSIONS_DIR = Path(
    r"C:\Users\Bryce\3d-option-application\public\data\sessions"
)
OPVIS_MANIFEST = (
    Path(r"C:\Users\Bryce\3d-option-application\public\data\manifest.json")
)

# nn-mind keeps its own recordings here
LOCAL_RECORDINGS = Path(__file__).parent / "recordings"


def list_sessions():
    """List all available recorded sessions (from OpVis + local)."""
    sessions = []

    # From OpVis
    if OPVIS_MANIFEST.exists():
        man = json.loads(OPVIS_MANIFEST.read_text(encoding='utf-8-sig'))
        for s in man.get("sessions", []):
            fp = OPVIS_SESSIONS_DIR / s["file"]
            if "sessions/" in s.get("file", ""):
                fp = OPVIS_SESSIONS_DIR / s["file"].replace("sessions/", "")
            elif (OPVIS_SESSIONS_DIR.parent / s['file']).exists():
                fp = OPVIS_SESSIONS_DIR.parent / s['file']
            if fp.exists():
                sessions.append({
                    "date": s["date"],
                    "symbol": s["symbol"],
                    "snapshots": s["snapshots"],
                    "path": fp,
                    "source": "opvis",
                })

    # From local recordings
    if LOCAL_RECORDINGS.exists():
        for fp in sorted(LOCAL_RECORDINGS.glob("session-*.json")):
            sessions.append({
                "date": fp.stem.replace("session-", ""),
                "symbol": "SPY",
                "snapshots": None,  # read on demand
                "path": fp,
                "source": "local",
            })

    return sorted(sessions, key=lambda s: s["date"], reverse=True)


def load_session(path):
    """Load a full recorded session into numpy arrays."""
    raw = json.loads(Path(path).read_text())

    meta = raw.get("meta", {})
    strikes = np.array(raw.get("strikes", []), dtype=np.float32)
    expiries = raw.get("expiries", [])
    snapshots = raw.get("snapshots", [])

    if not snapshots or len(strikes) == 0:
        return None

    NE = len(expiries)
    NS = len(strikes)
    NF = len(snapshots)

    # Build tensor: [frames, expiries, strikes, features]
    # features: iv, delta, gamma, vega, theta, oi, volume
    data = np.zeros((NF, NE, NS, 7), dtype=np.float32)
    spot = np.zeros(NF, dtype=np.float32)
    timestamps = []

    for fi, snap in enumerate(snapshots):
        timestamps.append(snap.get("t", ""))
        spot[fi] = snap.get("underlying", 0)

        iv = np.array(snap.get("iv", []), dtype=np.float32).reshape(NE, NS)
        delta = np.array(snap.get("callDelta", []), dtype=np.float32).reshape(NE, NS)
        gamma = np.array(snap.get("gamma", []), dtype=np.float32).reshape(NE, NS)
        vega = np.array(snap.get("vega", []), dtype=np.float32).reshape(NE, NS)
        theta = np.array(snap.get("callTheta", []), dtype=np.float32).reshape(NE, NS)

        call_oi = np.array(snap.get("callOI", []), dtype=np.float32).reshape(NE, NS)
        put_oi = np.array(snap.get("putOI", []), dtype=np.float32).reshape(NE, NS)
        oi = call_oi + put_oi

        call_vol = np.array(snap.get("callVol", []), dtype=np.float32).reshape(NE, NS)
        put_vol = np.array(snap.get("putVol", []), dtype=np.float32).reshape(NE, NS)
        volume = call_vol + put_vol

        data[fi, :, :, 0] = iv
        data[fi, :, :, 1] = delta
        data[fi, :, :, 2] = gamma
        data[fi, :, :, 3] = vega
        data[fi, :, :, 4] = theta
        data[fi, :, :, 5] = oi
        data[fi, :, :, 6] = volume

    return {
        "meta": meta,
        "strikes": strikes,
        "expiries": expiries,
        "spot": spot,
        "timestamps": timestamps,
        "data": data,  # [NF, NE, NS, 7]
        "nframes": NF,
        "nexp": NE,
        "nstrikes": NS,
    }


def latest_session():
    """Load the most recent recorded session."""
    sessions = list_sessions()
    if not sessions:
        return None
    return load_session(sessions[0]["path"])


if __name__ == "__main__":
    sessions = list_sessions()
    print(f"Found {len(sessions)} sessions:")
    for s in sessions[:5]:
        print(f"  {s['date']}  {s['symbol']}  ({s['source']})")

    latest = latest_session()
    if latest:
        print(f"\nLatest: {latest['meta'].get('sessionDate', '?')}")
        print(f"  Shape: {latest['data'].shape} (frames x expiries x strikes x features)")
        print(f"  Spot range: {latest['spot'].min():.2f} - {latest['spot'].max():.2f}")
        print(f"  IV range: {latest['data'][:,:,:,0].min():.4f} - {latest['data'][:,:,:,0].max():.4f}")
