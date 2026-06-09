"""Pattern & correlation analysis over recorded option surfaces.

These channels power the 3D pattern lab in the frontend:

- surface3d            -> grids for the rotatable 3D viewer:
                          market IV, a model's learned surface, or the
                          residual (model − market) where the NN sees
                          something the closed-form world doesn't.
- analysis.pca         -> eigenmodes of surface dynamics. Each recorded frame
                          is a point in (NE*NS)-dim IV space; SVD of the
                          demeaned frames gives the dominant deformation
                          patterns (level / skew-tilt / smile-flex) plus how
                          strongly each mode tracks the underlying spot.
- analysis.correlation -> correlation matrix across the 7 recorded features
                          (iv, delta, gamma, vega, theta, oi, volume) and the
                          spot-vs-feature correlations across frames (e.g. the
                          iv/spot leverage effect).
"""

import numpy as np

from data.loader import latest_session
from models import create_pricer

FEATURES = ["iv", "delta", "gamma", "vega", "theta", "oi", "volume"]

# Flat sigma the BS baseline assumes; the market-minus-this residual is
# exactly the smile/skew structure a constant-vol model cannot express.
BS_FLAT_SIGMA = 0.25


def _require_session():
    sd = latest_session()
    if sd is None:
        raise ValueError("No session data. Connect to IBKR or load a recording.")
    return sd


def _expiry_days(sd):
    return [e["days"] if isinstance(e, dict) else int(float(e) * 365)
            for e in sd["expiries"]]


def _market_iv(sd):
    """Session-mean IV surface [NE, NS]."""
    return sd["data"][:, :, :, 0].mean(axis=0)


def _model_surface(sd, model_type):
    from src.worker import get_model
    S0 = float(np.median(sd["spot"]))
    if model_type in ("bs", "binomial"):
        return create_pricer(model_type).surface(sd["strikes"], sd["expiries"], S0), "price"
    model = get_model(model_type)
    if model is None or not getattr(model, "is_trained", False):
        raise ValueError(f"Model '{model_type}' not trained yet — train it in the Models panel")
    # NN targets are IV-space (trainer uses IV as the regression target).
    return model.surface(sd["strikes"], sd["expiries"], S0), "iv"


def h_surface3d(payload):
    sd = _require_session()
    kind = payload.get("kind", "market")
    model_type = payload.get("model", "mlp")
    market = _market_iv(sd)

    if kind == "market":
        z, units = market, "iv"
        label = "Market IV surface (session mean)"
        diverge = False
    elif kind == "model":
        z, units = _model_surface(sd, model_type)
        label = (f"{model_type} learned surface" if units == "iv"
                 else f"{model_type} price surface")
        diverge = False
    elif kind == "residual":
        if model_type in ("bs", "binomial"):
            z, units = market - BS_FLAT_SIGMA, "iv"
            label = f"Market IV − flat σ={BS_FLAT_SIGMA}: the smile/skew {model_type} can't see"
        else:
            surf, units = _model_surface(sd, model_type)
            z = np.asarray(surf) - market
            label = f"{model_type} − market IV: where the NN diverges from the market"
        diverge = True
    else:
        raise ValueError(f"unknown kind '{kind}' (market | model | residual)")

    z = np.asarray(z, dtype=np.float64)
    return {
        "kind": kind, "model": model_type, "label": label, "units": units,
        "diverge": diverge,
        "strikes": sd["strikes"].tolist(),
        "expiry_days": _expiry_days(sd),
        "z": z.tolist(),
        "zmin": float(z.min()), "zmax": float(z.max()),
    }


def h_pca(payload):
    sd = _require_session()
    iv = sd["data"][:, :, :, 0]          # [NF, NE, NS]
    NF, NE, NS = iv.shape
    if NF < 3:
        raise ValueError(f"Need ≥3 frames for PCA, have {NF} — keep streaming")

    X = iv.reshape(NF, -1)
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2
    total = var.sum() or 1.0
    spot = sd["spot"]

    modes = []
    for k in range(min(3, len(S))):
        comp = Vt[k].reshape(NE, NS)
        loading = Xc @ Vt[k]             # per-frame strength of this mode
        if np.std(loading) > 0 and np.std(spot) > 0:
            spot_corr = float(np.corrcoef(loading, spot)[0, 1])
        else:
            spot_corr = 0.0
        modes.append({
            "z": comp.tolist(),
            "explained": float(var[k] / total),
            "spot_corr": spot_corr,
        })

    return {
        "modes": modes,
        "strikes": sd["strikes"].tolist(),
        "expiry_days": _expiry_days(sd),
        "n_frames": NF,
    }


def h_correlation(payload):
    sd = _require_session()
    d = sd["data"]                       # [NF, NE, NS, 7]
    flat = d.reshape(-1, len(FEATURES)).astype(np.float64)

    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std == 0] = 1.0                  # constant features -> zero correlation
    Xn = (flat - mean) / std
    matrix = np.nan_to_num(Xn.T @ Xn / len(Xn))
    np.fill_diagonal(matrix, 1.0)

    # How each feature's per-frame mean co-moves with spot across the session.
    frame_means = d.mean(axis=(1, 2))    # [NF, 7]
    spot = sd["spot"]
    spot_corr = {}
    for i, f in enumerate(FEATURES):
        s = frame_means[:, i]
        if np.std(s) > 0 and np.std(spot) > 0:
            spot_corr[f] = float(np.corrcoef(s, spot)[0, 1])
        else:
            spot_corr[f] = 0.0

    return {"features": FEATURES, "matrix": matrix.tolist(),
            "spot_corr": spot_corr, "n_samples": int(len(flat))}
