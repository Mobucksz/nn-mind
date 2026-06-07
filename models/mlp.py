"""Feedforward MLP for option pricing.

Input: [S, K, T, r, q, moneyness, time_to_expiry]
Output: [price, delta, gamma, vega, theta]
3 hidden layers (128, 64, 32) with ReLU + dropout.
"""

import numpy as np
import torch
import torch.nn as nn

from models.base import OptionPricer


class MLPNet(nn.Module):
    """PyTorch MLP for direct option price + greeks regression."""

    def __init__(self, input_dim: int = 7, hidden_dims=None, dropout: float = 0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 5))  # price, delta, gamma, vega, theta
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPPricer(OptionPricer):
    """NN option pricer using a feedforward MLP."""

    def __init__(self, name: str = "mlp", input_dim: int = 7, device: str = "cpu"):
        super().__init__(name)
        self.device = torch.device(device)
        self.model = MLPNet(input_dim=input_dim).to(self.device)
        self.model.eval()
        self._trained = False
        self._scaler = None  # stores feature stats for normalization
        self._target_scaler = None  # stores price stats
        self._train_history = []

    @property
    def is_trained(self) -> bool:
        return self._trained

    def _prepare_input(self, S, K, T, r=0.045, q=0):
        """Build feature vector: raw + derived features."""
        moneyness = K / S
        tte = T
        return np.array([S, K, T, r, q, moneyness, tte], dtype=np.float32)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self._scaler is None:
            return x
        return (x - self._scaler["mean"]) / (self._scaler["std"] + 1e-8)

    def _denormalize_price(self, p: np.ndarray) -> np.ndarray:
        if self._target_scaler is None:
            return p
        return p * self._target_scaler["std"] + self._target_scaler["mean"]

    def price(self, S, K, T, r=0.045, q=0):
        if not self._trained:
            return 0.0
        feat = self._prepare_input(S, K, T, r, q)
        feat_n = self._normalize(feat)
        with torch.no_grad():
            x = torch.from_numpy(feat_n).unsqueeze(0).to(self.device)
            out = self.model(x).cpu().numpy()[0]
        price_raw = out[0]
        price = self._denormalize_price(np.array([price_raw]))[0]
        return float(max(price, 0.0))

    def greeks(self, S, K, T, r=0.045, q=0):
        if not self._trained:
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        feat = self._prepare_input(S, K, T, r, q)
        feat_n = self._normalize(feat)
        with torch.no_grad():
            x = torch.from_numpy(feat_n).unsqueeze(0).to(self.device)
            out = self.model(x).cpu().numpy()[0]
        # Greeks are directly output by the model
        return {
            "delta": float(out[1]),
            "gamma": float(out[2]),
            "vega": float(out[3]),
            "theta": float(out[4]),
        }

    def surface(self, strikes, expiries, S, r=0.045, q=0):
        if not self._trained:
            return np.zeros((len(expiries), len(strikes)), dtype=np.float32)
        strikes = np.asarray(strikes, dtype=np.float32)
        expiries_T = np.array([e["T"] if isinstance(e, dict) else e for e in expiries], dtype=np.float32)
        NE, NS = len(expiries_T), len(strikes)
        result = np.zeros((NE, NS), dtype=np.float32)
        for i, T in enumerate(expiries_T):
            for j, K in enumerate(strikes):
                result[i, j] = self.price(S, K, T, r, q)
        return result

    def save(self, path):
        """Save model state dict + scaler info."""
        torch.save({
            "state_dict": self.model.state_dict(),
            "scaler": self._scaler,
            "target_scaler": self._target_scaler,
            "trained": self._trained,
            "history": self._train_history,
            "name": self.name,
        }, path)

    def load(self, path):
        """Load model from checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self._scaler = ckpt.get("scaler")
        self._target_scaler = ckpt.get("target_scaler")
        self._trained = ckpt.get("trained", False)
        self._train_history = ckpt.get("history", [])
        self.model.eval()

    def set_normalization(self, scaler, target_scaler):
        """Set normalization parameters computed from training data."""
        self._scaler = scaler
        self._target_scaler = target_scaler
