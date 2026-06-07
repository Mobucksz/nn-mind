"""nn-mind Flask server.

Serves:
- Dashboard HTML (index.html)
- Training API (start, status, list models)
- Inference API (price + greeks)
- Surface API (3D matrix for visualization)
- Compare API (difference surfaces)
- Latest session info
"""

import os
import json
import sys
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.worker import (
    start_training, get_job_status, list_jobs,
    get_model, list_available_models,
    compute_surface, compute_compare_surfaces,
)
from data.loader import latest_session, list_sessions, load_session
from models import create_pricer, OptionPricer

app = Flask(__name__, static_folder=None)
CORS(app)

# Canvas directory for serving frontend
CANVAS_DIR = os.path.join(os.path.expanduser("~"), ".openclaw", "canvas", "nn-mind")
os.makedirs(CANVAS_DIR, exist_ok=True)

# Default session for reference (preload BS model)
_bs_pricer = create_pricer("bs")


# ── Frontend ──────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the dashboard HTML."""
    return send_from_directory(CANVAS_DIR, "index.html")


@app.route("/<path:path>")
def static_files(path):
    """Serve static files from canvas dir (JS libs, etc.)."""
    return send_from_directory(CANVAS_DIR, path)


# ── Training API ─────────────────────────────────────────

@app.route("/api/train", methods=["POST"])
def api_train():
    """Start a training job.

    Body: {"model_type": "mlp", "epochs": 100, "lr": 0.001}
    """
    data = request.get_json(force=True)
    model_type = data.get("model_type", "mlp")
    epochs = int(data.get("epochs", 50))
    lr = float(data.get("lr", 0.001))

    if model_type in ("bs", "binomial"):
        return jsonify({"error": f"'{model_type}' is a closed-form model, no training needed"}), 400

    job_id = start_training(model_type, epochs=epochs, lr=lr)
    return jsonify({"job_id": job_id, "status": "queued"}), 202


@app.route("/api/train/<job_id>/status", methods=["GET"])
def api_train_status(job_id):
    """Get training job status."""
    job = get_job_status(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/train/jobs", methods=["GET"])
def api_train_jobs():
    """List all training jobs."""
    return jsonify(list_jobs())


# ── Inference API ────────────────────────────────────────

@app.route("/api/infer", methods=["POST"])
def api_infer():
    """Price a single option with a trained model.

    Body: {"S": 530, "K": 520, "T": 0.25, "model_type": "mlp"}
    Returns: {"price": ..., "delta": ..., "gamma": ..., "vega": ..., "theta": ...}
    """
    data = request.get_json(force=True)
    S = float(data["S"])
    K = float(data["K"])
    T = float(data["T"])
    model_type = data.get("model_type", "mlp")
    r = float(data.get("r", 0.045))

    # Use BS as default
    if model_type in ("bs", "binomial"):
        pricer = create_pricer(model_type)
        price = pricer.price(S, K, T, r=r)
        greeks = pricer.greeks(S, K, T, r=r)
    else:
        model = get_model(model_type)
        if model is None or not getattr(model, "is_trained", False):
            return jsonify({"error": f"Model '{model_type}' not trained yet"}), 400
        price = model.price(S, K, T, r=r)
        greeks = model.greeks(S, K, T, r=r)

    return jsonify({
        "price": price,
        "model": model_type,
        **greeks,
    })


# ── Surface API ──────────────────────────────────────────

@app.route("/api/surface", methods=["GET"])
def api_surface():
    """Get full surface matrix for a model.

    Query: ?model=mlp
    Returns: {strikes: [...], expiries: [...], prices: [[...]]}
    """
    model_type = request.args.get("model", "mlp")

    # Get latest session for reference strikes/expiries
    sd = latest_session()
    if sd is None:
        return jsonify({"error": "No session data"}), 400

    strikes = sd["strikes"].tolist()
    expiries = [{"date": e["date"], "days": e["days"], "T": e["T"]} if isinstance(e, dict)
                else e for e in sd["expiries"]]
    S0 = float(np.median(sd["spot"]))

    if model_type in ("bs", "binomial"):
        pricer = create_pricer(model_type)
    else:
        model = get_model(model_type)
        if model is None or not getattr(model, "is_trained", False):
            return jsonify({"error": f"Model '{model_type}' not trained"}), 400
        pricer = model

    prices = pricer.surface(sd["strikes"], sd["expiries"], S0)
    if isinstance(prices, np.ndarray):
        prices = prices.tolist()

    return jsonify({
        "model": model_type,
        "strikes": strikes,
        "expiries": expiries,
        "prices": prices,
        "S": S0,
    })


# ── Compare API ──────────────────────────────────────────

@app.route("/api/compare", methods=["GET"])
def api_compare():
    """Compare surfaces from multiple models.

    Query: ?models=mlp,lstm,bs
    Returns: {surfaces: {model: [[...]]}, differences: {pair: diff}}
    """
    model_str = request.args.get("models", "mlp,bs")
    model_types = [m.strip() for m in model_str.split(",")]

    sd = latest_session()
    if sd is None:
        return jsonify({"error": "No session data"}), 400

    S0 = float(np.median(sd["spot"]))
    surfaces = compute_compare_surfaces(model_types, sd["strikes"], sd["expiries"], S0)

    # Add BS/binomial if missing
    for mt in model_types:
        if mt not in surfaces and mt in ("bs", "binomial"):
            pricer = create_pricer(mt)
            surf = pricer.surface(sd["strikes"], sd["expiries"], S0)
            surfaces[mt] = surf.tolist()

    # Compute pairwise differences
    differences = {}
    keys = list(surfaces.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = np.array(surfaces[keys[i]])
            b = np.array(surfaces[keys[j]])
            diff = (a - b).tolist()
            differences[f"{keys[i]}-{keys[j]}"] = diff

    return jsonify({
        "models": keys,
        "surfaces": surfaces,
        "differences": differences,
        "strikes": sd["strikes"].tolist(),
        "expiries": [e["date"] if isinstance(e, dict) else str(e) for e in sd["expiries"]],
        "S": S0,
    })


# ── Models API ────────────────────────────────────────────

@app.route("/api/models", methods=["GET"])
def api_models():
    """List all available models and their training status.

    Returns: [{name, type, trained, ...}]
    """
    trained = list_available_models()

    # Always include closed-form models
    closed_form = [
        {"name": "bs", "type": "bs", "trained": True, "has_attention": False, "description": "Black-Scholes closed-form"},
        {"name": "binomial", "type": "binomial", "trained": True, "has_attention": False, "description": "Binomial tree (American)"},
    ]

    all_models = closed_form + [
        m for m in trained if m["name"] not in ("bs", "binomial")
    ]

    # Add placeholders for untrained models
    nn_types = ["mlp", "lstm", "transformer", "ensemble"]
    existing_names = {m["name"] for m in all_models}
    for nt in nn_types:
        if nt not in existing_names:
            all_models.append({
                "name": nt,
                "type": nt,
                "trained": False,
                "has_attention": nt == "transformer",
                "description": {
                    "mlp": "Feedforward MLP (3 layers)",
                    "lstm": "LSTM for vol term structure",
                    "transformer": "Attention-based surface encoder",
                    "ensemble": "Stacked ensemble with uncertainty",
                }[nt],
            })

    return jsonify(all_models)


# ── Session Info API ─────────────────────────────────────

@app.route("/api/latest-session", methods=["GET"])
def api_latest_session():
    """Get info about the most recent IBKR recording."""
    sd = latest_session()
    if sd is None:
        return jsonify({"error": "No sessions available"}), 404

    return jsonify({
        "meta": sd["meta"],
        "strikes": sd["strikes"].tolist(),
        "expiries": [{
            "date": e["date"] if isinstance(e, dict) else "?",
            "days": e["days"] if isinstance(e, dict) else 0,
            "T": e["T"] if isinstance(e, dict) else float(e),
        } for e in sd["expiries"]],
        "spot_min": float(sd["spot"].min()),
        "spot_max": float(sd["spot"].max()),
        "spot_avg": float(sd["spot"].mean()),
        "nframes": sd["nframes"],
        "shape": list(sd["data"].shape),
    })


@app.route("/api/sessions", methods=["GET"])
def api_sessions():
    """List all available recorded sessions."""
    sessions = list_sessions()
    return jsonify([{
        "date": s["date"],
        "symbol": s["symbol"],
        "source": s["source"],
        "path": str(s["path"]),
    } for s in sessions])


# ── Attention API ─────────────────────────────────────────

@app.route("/api/attention", methods=["POST"])
def api_attention():
    """Get attention map from transformer model.

    Body: {"S": 530, "K": 520, "T": 0.25}
    """
    data = request.get_json(force=True)
    S = float(data["S"])
    K = float(data["K"])
    T = float(data["T"])

    model = get_model("transformer")
    if model is None or not getattr(model, "is_trained", False):
        return jsonify({"error": "Transformer not trained yet"}), 400

    attn = model.attention(S, K, T)
    if isinstance(attn, np.ndarray):
        attn = attn.tolist()

    return jsonify({"attention": attn})


# ── Main ──────────────────────────────────────────────────

if __name__ == "__main__":
    # Ensure canvas dir exists with index.html
    index_path = os.path.join(CANVAS_DIR, "index.html")
    if not os.path.exists(index_path):
        print(f"WARNING: Frontend not found at {index_path}")
        print("Create index.html in the canvas directory before opening the app.")

    port = int(os.environ.get("PORT", 7891))
    print(f"  nn-mind server starting on http://127.0.0.1:{port}")
    print(f"  Canvas dir: {CANVAS_DIR}")
    print(f"  Models: bs, binomial, mlp, lstm, transformer, ensemble")
    print(f"  Session data: {list_sessions.__doc__}")

    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
