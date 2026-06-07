"""Abstract base class for all option pricers."""

from typing import Optional
import numpy as np


class OptionPricer:
    """Base class for option pricing models in nn-mind.

    Subclasses can be traditional (Black-Scholes, Binomial),
    neural (MLP, LSTM, Transformer), or ensembles.
    """

    def __init__(self, name: str):
        self.name = name

    def price(self, S: float, K: float, T: float, r: float = 0.045, q: float = 0) -> float:
        """Single option price."""
        raise NotImplementedError

    def surface(self, strikes: np.ndarray, expiries: np.ndarray,
                S: float, r: float = 0.045, q: float = 0) -> np.ndarray:
        """Return [NE, NS] price matrix given arrays of strikes and expiry T-values."""
        raise NotImplementedError

    def greeks(self, S: float, K: float, T: float,
               r: float = 0.045, q: float = 0) -> dict:
        """Return dict: delta, gamma, vega, theta."""
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

    def save(self, path: str) -> None:
        """Save model weights to disk."""
        pass

    def load(self, path: str) -> None:
        """Load model weights from disk."""
        pass

    def attention(self, S: float, K: float, T: float) -> Optional[np.ndarray]:
        """Return attention map for explainability, if applicable."""
        return None
