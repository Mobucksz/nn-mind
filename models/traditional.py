"""Black-Scholes and Binomial option pricers (closed-form, no training)."""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from models.base import OptionPricer


class BlackScholesPricer(OptionPricer):
    """Black-Scholes closed-form pricer for European calls.

    Always available, no training required.
    """

    def __init__(self, name: str = "bs"):
        super().__init__(name)

    def _d1(self, S, K, T, r, q, sigma=0.25):
        return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    def _d2(self, d1, T, sigma=0.25):
        return d1 - sigma * np.sqrt(T)

    def price(self, S, K, T, r=0.045, q=0, sigma=0.25):
        """European call price."""
        if T <= 0:
            return max(float(S - K), 0.0)
        d1 = self._d1(S, K, T, r, q, sigma)
        d2 = self._d2(d1, T, sigma)
        return float(S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))

    def greeks(self, S, K, T, r=0.045, q=0, sigma=0.25):
        """Compute Greeks for a European call."""
        if T <= 0:
            return {"delta": float(S > K), "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        d1 = self._d1(S, K, T, r, q, sigma)
        d2 = self._d2(d1, T, sigma)
        delta = np.exp(-q * T) * norm.cdf(d1)
        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100  # per 1% vol move
        theta = (-(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)
                 + q * S * np.exp(-q * T) * norm.cdf(d1)) / 365
        return {"delta": float(delta), "gamma": float(gamma),
                "vega": float(vega), "theta": float(theta)}

    def surface(self, strikes, expiries, S, r=0.045, q=0, sigma=0.25):
        """Return [NE, NS] price matrix."""
        strikes = np.asarray(strikes, dtype=np.float32)
        expiries_T = np.array([e["T"] if isinstance(e, dict) else e for e in expiries], dtype=np.float32)
        NE, NS = len(expiries_T), len(strikes)
        result = np.zeros((NE, NS), dtype=np.float32)
        for i, T in enumerate(expiries_T):
            for j, K in enumerate(strikes):
                result[i, j] = self.price(S, K, T, r, q, sigma)
        return result

    def implied_vol(self, S, K, T, market_price, r=0.045, q=0):
        """Compute implied vol from a market price using Brent's method."""
        if T <= 0:
            return 0.0
        try:
            def f(sigma):
                return self.price(S, K, T, r, q, sigma) - market_price
            return float(brentq(f, 0.001, 2.0))
        except (ValueError, RuntimeError):
            return 0.25  # fallback to ATM guess

    def attention(self, S, K, T):
        """No attention for closed-form models."""
        return None


class BinomialPricer(OptionPricer):
    """Binomial tree pricer for American options (with early exercise)."""

    def __init__(self, name: str = "binomial", steps: int = 100):
        super().__init__(name)
        self.steps = steps

    def price(self, S, K, T, r=0.045, q=0, sigma=0.25, is_call=True):
        """American option price via binomial tree."""
        if T <= 0:
            return max(float(S - K) if is_call else float(K - S), 0.0)
        dt = T / self.steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)

        # Price at expiry (terminal nodes)
        ST = S * (u ** (np.arange(self.steps + 1))) * (d ** (self.steps - np.arange(self.steps + 1)))
        if is_call:
            V = np.maximum(ST - K, 0)
        else:
            V = np.maximum(K - ST, 0)

        # Backwards induction
        for i in range(self.steps - 1, -1, -1):
            V = np.exp(-r * dt) * (p * V[:i + 1] + (1 - p) * V[1:i + 2])
            # Early exercise check for American
            S_t = S * (u ** np.arange(i + 1)) * (d ** (i - np.arange(i + 1)))
            if is_call:
                V = np.maximum(V, S_t - K)
            else:
                V = np.maximum(V, K - S_t)
        return float(V[0])

    def surface(self, strikes, expiries, S, r=0.045, q=0, sigma=0.25):
        strikes = np.asarray(strikes, dtype=np.float32)
        expiries_T = np.array([e["T"] if isinstance(e, dict) else e for e in expiries], dtype=np.float32)
        NE, NS = len(expiries_T), len(strikes)
        result = np.zeros((NE, NS), dtype=np.float32)
        for i, T in enumerate(expiries_T):
            for j, K in enumerate(strikes):
                result[i, j] = self.price(S, K, T, r, q, sigma)
        return result
