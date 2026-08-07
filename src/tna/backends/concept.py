"""Rung 6: optional concept-space averaging scaffold."""

from __future__ import annotations

from .base import UnavailableBackend


def build_concept_backend(arrays, seed: int = 0):
    # This placeholder is explicit because concept tokens would require
    # separate training assets and would otherwise be easy to overclaim.
    return UnavailableBackend(
        "concept",
        "learned per-nation concept token embedding",
        "Concept-space averaging requires textual-inversion assets that are not part of the base repository.",
        seed=seed,
    )
