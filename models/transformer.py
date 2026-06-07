"""Attention-based Transformer for surface encoding.

Input: flattened (strike, expiry) pairs + features
Positional encoding + multi-head attention
Output: price per (strike, expiry)
4 heads, 2 layers, d_model=64.
"""

import math
import numpy as np
import torch
import torch.nn as nn

from models.base import OptionPricer


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for surface grid positions."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerNet(nn.Module):
    """Transformer encoder applied to (strike x expiry) grid patches."""

    def __init__(self, input_dim: int = 4, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout,
            activation="relu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.output_proj(x).squeeze(-1)  # [batch, seq_len]
        return x


class TransformerPricer(OptionPricer):
    """Transformer-based surface pricer with attention explainability."""

    def __init__(self, name: str = "transformer", device: str = "cpu"):
        super().__init__(name)
        self.device = torch.device(device)
        self.model = TransformerNet().to(self.device)
        self.model.eval()
        self._trained = False
        self._strikes = None
        self._expiries = None
        self._scaler = None
        self._target_scaler = None
        self._train_history = []

    @property
    def is_trained(self) -> bool:
        return self._trained

    def _build_grid_features(self, S, strikes, expiries, r=0.045, q=0):
        """Build feature vectors for each (strike_idx, expiry_idx) pair."""
        NE = len(expiries)
        NS = len(strikes)
        features = []
        for ei, exp in enumerate(expiries):
            T = exp["T"] if isinstance(exp, dict) else exp
            for si, K in enumerate(strikes):
                features.append([
                    K / S,  # moneyness
                    T,      # time to expiry
                    ei / max(NE - 1, 1),  # norm expiry index
                    si / max(NS - 1, 1),  # norm strike index
                ])
        return np.array(features, dtype=np.float32)

    def _unflatten(self, arr, NE, NS):
        """Unflatten [N] back to [NE, NS]."""
        return arr.reshape(NE, NS)

    def _normalize(self, x):
        if self._scaler is None:
            return x
        return (x - self._scaler["mean"]) / (self._scaler["std"] + 1e-8)

    def _denormalize(self, p):
        if self._target_scaler is None:
            return p
        return p * self._target_scaler["std"] + self._target_scaler["mean"]

    def price(self, S, K, T, r=0.045, q=0):
        if not self._trained:
            return 0.0
        # Approximate: use closest strike/expiry from stored grid
        if self._strikes is None or self._expiries is None:
            return 0.0
        strikes = np.asarray(self._strikes)
        expiries = self._expiries
        exp_T = np.array([e["T"] if isinstance(e, dict) else e for e in expiries])
        si = np.argmin(np.abs(strikes - K))
        ei = np.argmin(np.abs(exp_T - T))
        result = self.surface(strikes, expiries, S, r, q)
        return float(result[ei, si])

    def surface(self, strikes, expiries, S, r=0.045, q=0):
        if not self._trained:
            return np.zeros((len(expiries), len(strikes)), dtype=np.float32)
        NE, NS = len(expiries), len(strikes)
        features = self._build_grid_features(S, strikes, expiries, r, q)
        features_n = self._normalize(features)
        with torch.no_grad():
            x = torch.from_numpy(features_n).unsqueeze(0).to(self.device)  # [1, N, input_dim]
            out = self.model(x).cpu().numpy()[0]  # [N]
        prices = self._denormalize(out)
        prices = np.maximum(prices, 0)  # enforce non-negative
        return self._unflatten(prices, NE, NS)

    def attention(self, S, K, T):
        """Return attention heatmap from the transformer encoder.

        Returns:
            np.ndarray [NE, NS] of attention weights for the CLS-like token.
        """
        if not self._trained:
            return None
        # Hook into the last encoder layer to get attention weights
        # For simplicity, return uniform attention as a placeholder
        NE = len(self._expiries)
        NS = len(self._strikes)
        attn = np.ones((NE, NS), dtype=np.float32) / (NE * NS)
        return attn

    def save(self, path):
        torch.save({
            "state_dict": self.model.state_dict(),
            "scaler": self._scaler,
            "target_scaler": self._target_scaler,
            "trained": self._trained,
            "strikes": self._strikes,
            "expiries": self._expiries,
            "history": self._train_history,
            "name": self.name,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self._scaler = ckpt.get("scaler")
        self._target_scaler = ckpt.get("target_scaler")
        self._trained = ckpt.get("trained", False)
        self._strikes = ckpt.get("strikes")
        self._expiries = ckpt.get("expiries")
        self._train_history = ckpt.get("history", [])
        self.model.eval()

    def set_normalization(self, scaler, target_scaler):
        self._scaler = scaler
        self._target_scaler = target_scaler

    def set_grid(self, strikes, expiries):
        self._strikes = strikes
        self._expiries = expiries
