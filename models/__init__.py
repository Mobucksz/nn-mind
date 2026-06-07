"""Model registry and factory for nn-mind."""

from models.base import OptionPricer
from models.traditional import BlackScholesPricer, BinomialPricer
from models.mlp import MLPPricer
from models.lstm import LSTMPricer
from models.transformer import TransformerPricer
from models.ensemble import EnsemblePricer

REGISTRY = {
    "bs": BlackScholesPricer,
    "binomial": BinomialPricer,
    "mlp": MLPPricer,
    "lstm": LSTMPricer,
    "transformer": TransformerPricer,
    "ensemble": EnsemblePricer,
}


def create_pricer(model_type: str, **kwargs) -> OptionPricer:
    """Factory: instantiate a pricer by model type string."""
    cls = REGISTRY.get(model_type)
    if cls is None:
        raise ValueError(f"Unknown model type '{model_type}'. Available: {list(REGISTRY.keys())}")
    return cls(**kwargs)
