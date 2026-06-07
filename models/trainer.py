"""Training loop for neural network option pricers.

Loads recorded session data, computes features, trains models,
and stores training history for live status updates.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.mlp import MLPPricer
from models.lstm import LSTMPricer
from models.transformer import TransformerPricer


def prepare_features(session_data, mode="mlp"):
    """Convert raw session data to features and targets for a given model type.

    Args:
        session_data: dict from data.loader.load_session()
        mode: "mlp", "lstm", or "transformer"

    Returns:
        dict with X_train, y_train, X_val, y_val, scaler, target_scaler
    """
    data = session_data["data"]  # [NF, NE, NS, 7]
    strikes = session_data["strikes"]
    expiries = session_data["expiries"]
    spot = session_data["spot"]
    S0 = float(np.median(spot))

    NF, NE, NS = data.shape[:3]
    iv = data[:, :, :, 0]  # [NF, NE, NS]

    if mode == "mlp":
        # Build flat feature set: one sample per (frame, expiry, strike)
        rows = []
        targets = []
        for fi in range(NF):
            for ei in range(NE):
                T = expiries[ei]["T"] if isinstance(expiries[ei], dict) else expiries[ei]
                for si in range(NS):
                    K = strikes[si]
                    moneyness = K / S0
                    rows.append([S0, K, T, 0.045, 0.0, moneyness, T])
                    targets.append([
                        iv[fi, ei, si],         # target: price (using IV as proxy)
                        data[fi, ei, si, 1],     # delta
                        data[fi, ei, si, 2],     # gamma
                        data[fi, ei, si, 3],     # vega
                        data[fi, ei, si, 4],     # theta
                    ])
        X = np.array(rows, dtype=np.float32)
        y = np.array(targets, dtype=np.float32)

        # Normalize
        scaler = {"mean": X.mean(axis=0), "std": X.std(axis=0)}
        target_scaler = {"mean": y.mean(axis=0), "std": y.std(axis=0)}

        X_n = (X - scaler["mean"]) / (scaler["std"] + 1e-8)
        y_n = (y - target_scaler["mean"]) / (target_scaler["std"] + 1e-8)

        # Split: last frame as validation
        n_train = (NF - 1) * NE * NS
        return {
            "X_train": X_n[:n_train],
            "y_train": y_n[:n_train],
            "X_val": X_n[n_train:],
            "y_val": y_n[n_train:],
            "scaler": scaler,
            "target_scaler": target_scaler,
        }

    elif mode == "transformer":
        # Build sequences of (strike, expiry) pairs per frame
        rows = []
        targets = []
        for fi in range(NF):
            grid_feats = []
            grid_targets = []
            for ei in range(NE):
                T = expiries[ei]["T"] if isinstance(expiries[ei], dict) else expiries[ei]
                for si in range(NS):
                    K = strikes[si]
                    grid_feats.append([
                        K / S0,
                        T,
                        ei / max(NE - 1, 1),
                        si / max(NS - 1, 1),
                    ])
                    grid_targets.append(iv[fi, ei, si])
            rows.append(grid_feats)
            targets.append(grid_targets)

        X = np.array(rows, dtype=np.float32)  # [NF, NE*NS, 4]
        y = np.array(targets, dtype=np.float32)  # [NF, NE*NS]

        scaler = {"mean": X.mean(axis=(0, 1)), "std": X.std(axis=(0, 1))}
        target_scaler = {"mean": float(y.mean()), "std": float(y.std())}

        X_n = (X - scaler["mean"]) / (scaler["std"] + 1e-8)
        y_n = (y - target_scaler["mean"]) / (target_scaler["std"] + 1e-8)

        n_train = NF - 1
        return {
            "X_train": X_n[:n_train],
            "y_train": y_n[:n_train],
            "X_val": X_n[n_train:],
            "y_val": y_n[n_train:],
            "scaler": scaler,
            "target_scaler": target_scaler,
        }

    elif mode == "lstm":
        # Build IV term structure sequences per strike
        rows = []
        targets = []
        for fi in range(NF):
            for si in range(NS):
                seq = iv[fi, :, si]  # [NE]
                rows.append(seq.reshape(-1, 1))
                targets.append(seq)
        X = np.array(rows, dtype=np.float32)  # [NF*NS, NE, 1]
        y = np.array(targets, dtype=np.float32)  # [NF*NS, NE]

        scaler = {"mean": X.mean(), "std": X.std()}
        target_scaler = {"mean": float(y.mean()), "std": float(y.std())}

        X_n = (X - scaler["mean"]) / (scaler["std"] + 1e-8)
        y_n = (y - target_scaler["mean"]) / (target_scaler["std"] + 1e-8)

        n_train = (NF - 1) * NS
        return {
            "X_train": X_n[:n_train],
            "y_train": y_n[:n_train],
            "X_val": X_n[n_train:],
            "y_val": y_n[n_train:],
            "scaler": scaler,
            "target_scaler": target_scaler,
        }

    else:
        raise ValueError(f"Unknown mode '{mode}'")


def train_model(pricer, session_data, epochs=50, lr=0.001,
                status_callback=None):
    """Train an NN option pricer on recorded session data.

    Args:
        pricer: MLPPricer, LSTMPricer, or TransformerPricer instance
        session_data: dict from data.loader.load_session()
        epochs: number of training epochs
        lr: learning rate
        status_callback: optional fn(epoch, loss, val_loss, lr) for live updates

    Returns:
        training_history: list of dicts per epoch
    """
    mode_map = {
        "mlp": "mlp",
        "lstm": "lstm",
        "transformer": "transformer",
    }
    pricer_type = type(pricer).__name__.lower().replace("pricer", "")
    mode = mode_map.get(pricer_type, pricer_type)

    data = prepare_features(session_data, mode=mode)

    X_train = torch.from_numpy(data["X_train"]).to(pricer.device)
    y_train = torch.from_numpy(data["y_train"]).to(pricer.device)
    X_val = torch.from_numpy(data["X_val"]).to(pricer.device)
    y_val = torch.from_numpy(data["y_val"]).to(pricer.device)

    pricer.model.train()
    optimizer = optim.Adam(pricer.model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred_train = pricer.model(X_train)
        loss = criterion(pred_train, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        pricer.model.eval()
        with torch.no_grad():
            pred_val = pricer.model(X_val)
            val_loss = criterion(pred_val, y_val).item()
        pricer.model.train()

        current_lr = scheduler.get_last_lr()[0]
        history.append({
            "epoch": epoch + 1,
            "loss": float(loss.item()),
            "val_loss": float(val_loss),
            "lr": float(current_lr),
        })

        if status_callback:
            status_callback(epoch + 1, float(loss.item()), float(val_loss), float(current_lr))

    pricer.model.eval()
    pricer._trained = True
    pricer._train_history = history
    pricer.set_normalization(data["scaler"], data["target_scaler"])

    # Set grid on transformer
    if isinstance(pricer, TransformerPricer):
        pricer.set_grid(session_data["strikes"], session_data["expiries"])

    return history
