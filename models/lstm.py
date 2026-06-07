"""LSTM for volatility term structure prediction.

Input: sequence of IV across expiries for a given strike
Output: predicted IV for each expiry
2-layer LSTM with hidden size 64.
"""

import numpy as np
import torch
import torch.nn as nn

from models.base import OptionPricer


class LSTMNet(nn.Module):
    """LSTM that takes IV sequences across expiries and predicts full IV vector."""

    def __init__(self, input_dim: int = 1, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
        self.regressor = nn.Linear(hidden_size, input_dim)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        out, _ = self.lstm(x)
        # Use output from last time step -> predict all expiries
        out = self.regressor(out[:, -1, :].unsqueeze(1))
        # Expand back to full sequence length
        return out.expand(-1, x.size(1), -1)


class LSTMPricer(OptionPricer):
    """LSTM-based pricer: learns vol term structure from data."""

    def __init__(self, name: str = "lstm", device: str = "cpu"):
        super().__init__(name)
        self.device = torch.device(device)
        self.model = LSTMNet().to(self.device)
        self.model.eval()
        self._trained = False
        self._scaler = None
        self._target_scaler = None
        self._train_history = []

    @property
    def is_trained(self) -> bool:
        return self._trained

    def price(self, S, K, T, r=0.045, q=0):
        """LSTM doesn't price single options directly; return 0."""
        return 0.0

    def predict_iv_surface(self, features):
        """Predict full IV surface [NE, NS] from features.

        Args:
            features: dict with 'strike_iv_sequences' [NS, NE] input sequences
        """
        if not self._trained:
            return None
        with torch.no_grad():
            x = torch.from_numpy(features).float().to(self.device)
            out = self.model(x)
            return out.cpu().numpy()

    def surface(self, strikes, expiries, S, r=0.045, q=0):
        if not self._trained:
            return np.zeros((len(expiries), len(strikes)), dtype=np.float32)
        # NOTE: LSTM surface requires pre-computed feature input
        # Caller should use predict_iv_surface instead
        return np.zeros((len(expiries), len(strikes)), dtype=np.float32)

    def save(self, path):
        torch.save({
            "state_dict": self.model.state_dict(),
            "scaler": self._scaler,
            "target_scaler": self._target_scaler,
            "trained": self._trained,
            "history": self._train_history,
            "name": self.name,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self._scaler = ckpt.get("scaler")
        self._target_scaler = ckpt.get("target_scaler")
        self._trained = ckpt.get("trained", False)
        self._train_history = ckpt.get("history", [])
        self.model.eval()

    def set_normalization(self, scaler, target_scaler):
        self._scaler = scaler
        self._target_scaler = target_scaler
