from .predictive_cache import PredictiveExpertCache
from .hot_set import HotExpertSet

__all__ = ["PredictiveExpertCache", "HotExpertSet", "MultiTierCache", "TieredExpertWeights"]


def __getattr__(name):
    if name not in {"MultiTierCache", "TieredExpertWeights"}:
        raise AttributeError(name)
    from .multi_tier import MultiTierCache, TieredExpertWeights

    return {"MultiTierCache": MultiTierCache, "TieredExpertWeights": TieredExpertWeights}[name]
