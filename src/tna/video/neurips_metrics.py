"""Small, explicit distribution measures for the NeurIPS edition.

These functions operate only on already-computed weighting records. They do not
infer political meaning from the numbers; they make the concentration produced by
a chosen weighting perceptible and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Mapping


@dataclass(frozen=True)
class WeightDistributionStats:
    intent: str
    contributor_count: int
    normalised_entropy: float
    effective_count: float
    top_share: float


def _probabilities(weights: Mapping[str, float]) -> list[float]:
    values = [max(0.0, float(value)) for value in weights.values()]
    total = sum(values)
    if total <= 0.0:
        return [0.0 for _ in values]
    return [value / total for value in values]


def normalised_entropy(weights: Mapping[str, float]) -> float:
    """Shannon entropy normalised to [0, 1] for the positive contributors."""

    probabilities = [p for p in _probabilities(weights) if p > 0.0]
    if len(probabilities) <= 1:
        return 0.0
    entropy = -sum(p * log(p) for p in probabilities)
    return entropy / log(len(probabilities))


def effective_count(weights: Mapping[str, float]) -> float:
    """Entropy effective number of contributors, exp(H)."""

    probabilities = [p for p in _probabilities(weights) if p > 0.0]
    if not probabilities:
        return 0.0
    entropy = -sum(p * log(p) for p in probabilities)
    return exp(entropy)


def top_share(weights: Mapping[str, float]) -> float:
    probabilities = _probabilities(weights)
    return max(probabilities, default=0.0)


def count_below(weights: Mapping[str, float], threshold: float) -> int:
    """Count contributors whose normalised share is below ``threshold``."""

    return sum(p < threshold for p in _probabilities(weights))


def stats_for(intent: str, weights: Mapping[str, float]) -> WeightDistributionStats:
    probabilities = _probabilities(weights)
    return WeightDistributionStats(
        intent=intent,
        contributor_count=sum(p > 0.0 for p in probabilities),
        normalised_entropy=normalised_entropy(weights),
        effective_count=effective_count(weights),
        top_share=max(probabilities, default=0.0),
    )
