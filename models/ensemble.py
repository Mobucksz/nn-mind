"""Ensemble: stack multiple models, return mean + uncertainty."""

import numpy as np

from models.base import OptionPricer


class EnsemblePricer(OptionPricer):
    """Ensemble pricer that combines multiple sub-pricers.

    Returns mean, std, and per-model outputs for uncertainty quantification.
    """

    def __init__(self, name: str = "ensemble", models=None):
        super().__init__(name)
        self.models = models or []

    def add_model(self, model: OptionPricer):
        self.models.append(model)

    @property
    def is_trained(self):
        return len(self.models) > 0

    def price(self, S, K, T, r=0.045, q=0):
        if not self.models:
            return 0.0
        prices = [m.price(S, K, T, r, q) for m in self.models if hasattr(m, 'is_trained') and m.is_trained]
        if not prices:
            return 0.0
        return float(np.mean(prices))

    def price_with_uncertainty(self, S, K, T, r=0.045, q=0):
        """Return (mean_price, std_price, per_model_prices)."""
        prices = []
        details = []
        for m in self.models:
            if hasattr(m, 'is_trained') and m.is_trained:
                p = m.price(S, K, T, r, q)
                prices.append(p)
                details.append({"model": m.name, "price": p})
        if not prices:
            return 0.0, 0.0, []
        return float(np.mean(prices)), float(np.std(prices)), details

    def surface(self, strikes, expiries, S, r=0.045, q=0):
        if not self.models:
            return np.zeros((len(expiries), len(strikes)), dtype=np.float32)
        surfaces = []
        for m in self.models:
            if hasattr(m, 'is_trained') and m.is_trained:
                surfaces.append(m.surface(strikes, expiries, S, r, q))
        if not surfaces:
            return np.zeros((len(expiries), len(strikes)), dtype=np.float32)
        return np.mean(surfaces, axis=0)

    def surface_with_uncertainty(self, strikes, expiries, S, r=0.045, q=0):
        """Return (mean_surface, std_surface)."""
        surfaces = []
        for m in self.models:
            if hasattr(m, 'is_trained') and m.is_trained:
                surfaces.append(m.surface(strikes, expiries, S, r, q))
        if not surfaces:
            NE, NS = len(expiries), len(strikes)
            return np.zeros((NE, NS)), np.zeros((NE, NS))
        return np.mean(surfaces, axis=0), np.std(surfaces, axis=0)

    def greeks(self, S, K, T, r=0.045, q=0):
        if not self.models:
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        greeks_list = [m.greeks(S, K, T, r, q) for m in self.models
                       if hasattr(m, 'is_trained') and m.is_trained]
        if not greeks_list:
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        return {k: float(np.mean([g[k] for g in greeks_list])) for k in greeks_list[0]}

    def save(self, path):
        import json
        meta = {
            "name": self.name,
            "model_names": [m.name for m in self.models],
        }
        with open(path, "w") as f:
            json.dump(meta, f)

    def load(self, path):
        pass  # Sub-models must be loaded individually
