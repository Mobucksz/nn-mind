"""nn-mind compute sidecar — speaks JSON over stdio, no HTTP server.

The Electron main process spawns this script and exchanges newline-delimited
JSON messages with it:

    request   {"id": 7, "channel": "infer", "payload": {...}}
    response  {"id": 7, "ok": true,  "result": {...}}
              {"id": 7, "ok": false, "error": "message"}
    event     {"event": "ibkr.snapshot", "data": {...}}   (unsolicited)

Channels mirror the old Flask routes (models/surface/infer/compare/train/
session) plus the new live IBKR feed (ibkr.*). All heavy lifting still lives
in models/, src/worker.py and data/loader.py — this file is just transport
and dispatch.
"""

import os
import sys
import json
import threading
import traceback

import numpy as np

# Project root on path so `models`, `src`, `data` import cleanly.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.worker import (
    start_training, get_job_status, list_jobs,
    get_model, list_available_models, compute_compare_surfaces,
)
from data.loader import latest_session, list_sessions, load_session
from models import create_pricer

# stdout is the wire — protect it with a lock and never print() to it elsewhere.
_out_lock = threading.Lock()


def _send(obj):
    line = json.dumps(obj)
    with _out_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def emit(event, data):
    """Push an unsolicited event to the Electron main process."""
    _send({"event": event, "data": data})


# ── live IBKR manager (lazy: only built when first used) ──────────
_ibkr = None


def _get_ibkr():
    global _ibkr
    if _ibkr is None:
        from backend.ibkr_live import IBKRManager
        _ibkr = IBKRManager(emit=emit)
    return _ibkr


# ── session helpers (shared by several channels) ─────────────────

def _expiry_list(sd):
    return [{
        "date": e["date"] if isinstance(e, dict) else "?",
        "days": e["days"] if isinstance(e, dict) else None,
        "T": e["T"] if isinstance(e, dict) else float(e),
    } for e in sd["expiries"]]


def _require_session():
    sd = latest_session()
    if sd is None:
        raise ValueError("No session data. Connect to IBKR or load a recording.")
    return sd


# ── channel handlers ─────────────────────────────────────────────

def h_models(_):
    trained = list_available_models()
    closed_form = [
        {"name": "bs", "type": "bs", "trained": True, "has_attention": False,
         "description": "Black-Scholes closed-form"},
        {"name": "binomial", "type": "binomial", "trained": True, "has_attention": False,
         "description": "Binomial tree (American)"},
    ]
    all_models = closed_form + [m for m in trained if m["name"] not in ("bs", "binomial")]
    nn_types = ["mlp", "lstm", "transformer", "ensemble"]
    existing = {m["name"] for m in all_models}
    descs = {
        "mlp": "Feedforward MLP (3 layers)",
        "lstm": "LSTM for vol term structure",
        "transformer": "Attention-based surface encoder",
        "ensemble": "Stacked ensemble with uncertainty",
    }
    for nt in nn_types:
        if nt not in existing:
            all_models.append({
                "name": nt, "type": nt, "trained": False,
                "has_attention": nt == "transformer", "description": descs[nt],
            })
    return all_models


def h_session_latest(_):
    sd = _require_session()
    return {
        "meta": sd["meta"],
        "strikes": sd["strikes"].tolist(),
        "expiries": _expiry_list(sd),
        "spot_min": float(sd["spot"].min()),
        "spot_max": float(sd["spot"].max()),
        "spot_avg": float(sd["spot"].mean()),
        "nframes": sd["nframes"],
        "shape": list(sd["data"].shape),
    }


def h_session_list(_):
    return [{
        "date": s["date"], "symbol": s["symbol"],
        "source": s["source"], "path": str(s["path"]),
    } for s in list_sessions()]


def h_surface(payload):
    model_type = payload.get("model", "bs")
    sd = _require_session()
    strikes = sd["strikes"].tolist()
    expiries = _expiry_list(sd)
    S0 = float(np.median(sd["spot"]))
    if model_type in ("bs", "binomial"):
        pricer = create_pricer(model_type)
    else:
        model = get_model(model_type)
        if model is None or not getattr(model, "is_trained", False):
            raise ValueError(f"Model '{model_type}' not trained")
        pricer = model
    prices = pricer.surface(sd["strikes"], sd["expiries"], S0)
    if isinstance(prices, np.ndarray):
        prices = prices.tolist()
    return {"model": model_type, "strikes": strikes, "expiries": expiries,
            "prices": prices, "S": S0}


def h_infer(payload):
    S = float(payload["S"]); K = float(payload["K"]); T = float(payload["T"])
    model_type = payload.get("model_type", "bs")
    r = float(payload.get("r", 0.045))
    if model_type in ("bs", "binomial"):
        pricer = create_pricer(model_type)
        price = pricer.price(S, K, T, r=r)
        greeks = pricer.greeks(S, K, T, r=r)
    else:
        model = get_model(model_type)
        if model is None or not getattr(model, "is_trained", False):
            raise ValueError(f"Model '{model_type}' not trained yet")
        price = model.price(S, K, T, r=r)
        greeks = model.greeks(S, K, T, r=r)
    return {"price": price, "model": model_type, **greeks}


def h_compare(payload):
    model_types = payload.get("models", ["bs", "binomial"])
    if isinstance(model_types, str):
        model_types = [m.strip() for m in model_types.split(",")]
    sd = _require_session()
    S0 = float(np.median(sd["spot"]))
    surfaces = compute_compare_surfaces(model_types, sd["strikes"], sd["expiries"], S0)
    for mt in model_types:
        if mt not in surfaces and mt in ("bs", "binomial"):
            surfaces[mt] = create_pricer(mt).surface(sd["strikes"], sd["expiries"], S0).tolist()
    differences = {}
    keys = list(surfaces.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = np.array(surfaces[keys[i]]); b = np.array(surfaces[keys[j]])
            differences[f"{keys[i]}-{keys[j]}"] = (a - b).tolist()
    return {"models": keys, "surfaces": surfaces, "differences": differences,
            "strikes": sd["strikes"].tolist(),
            "expiries": [e["date"] if isinstance(e, dict) else str(e) for e in sd["expiries"]],
            "S": S0}


def h_train_start(payload):
    model_type = payload.get("model_type", "mlp")
    if model_type in ("bs", "binomial"):
        raise ValueError(f"'{model_type}' is closed-form, no training needed")
    job_id = start_training(model_type,
                            epochs=int(payload.get("epochs", 50)),
                            lr=float(payload.get("lr", 0.001)))
    return {"job_id": job_id, "status": "queued"}


def h_train_status(payload):
    job = get_job_status(payload["job_id"])
    if job is None:
        raise ValueError("Job not found")
    return job


def h_train_jobs(_):
    return list_jobs()


# IBKR live feed
def h_ibkr_connect(payload):
    return _get_ibkr().connect(
        host=payload.get("host", "127.0.0.1"),
        port=int(payload.get("port", 7497)),
        client_id=int(payload.get("client_id", 17)),
    )


def h_ibkr_status(_):
    return _get_ibkr().status_dict()


def h_ibkr_disconnect(_):
    return _get_ibkr().disconnect()


def h_ibkr_stream_start(payload):
    return _get_ibkr().start_stream(payload or {})


def h_ibkr_stream_stop(_):
    return _get_ibkr().stop_stream()


HANDLERS = {
    "ping": lambda _: {"pong": True},
    "models.list": h_models,
    "session.latest": h_session_latest,
    "session.list": h_session_list,
    "surface": h_surface,
    "infer": h_infer,
    "compare": h_compare,
    "train.start": h_train_start,
    "train.status": h_train_status,
    "train.jobs": h_train_jobs,
    "ibkr.connect": h_ibkr_connect,
    "ibkr.status": h_ibkr_status,
    "ibkr.disconnect": h_ibkr_disconnect,
    "ibkr.stream.start": h_ibkr_stream_start,
    "ibkr.stream.stop": h_ibkr_stream_stop,
}


def _dispatch(msg):
    rid = msg.get("id")
    channel = msg.get("channel")
    payload = msg.get("payload") or {}
    handler = HANDLERS.get(channel)
    if handler is None:
        _send({"id": rid, "ok": False, "error": f"unknown channel '{channel}'"})
        return
    try:
        result = handler(payload)
        _send({"id": rid, "ok": True, "result": result})
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        _send({"id": rid, "ok": False, "error": str(e)})


def main():
    emit("sidecar.ready", {"pid": os.getpid()})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            _send({"id": None, "ok": False, "error": "invalid JSON"})
            continue
        # Each request handled on its own thread so a slow channel never blocks
        # the stdin reader (training already detaches, but infer/surface can be
        # chunky and IBKR connect blocks while the socket handshakes).
        threading.Thread(target=_dispatch, args=(msg,), daemon=True).start()


if __name__ == "__main__":
    main()
