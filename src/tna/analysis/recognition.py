"""Real CLIP recognition of an averaged embedding.

Averages the selected flags' CLIP embeddings under the given weights, retrieves
the nearest real flags from the corpus, and reports a real confidence (softmax
over the corpus cosine similarities at a stated temperature) plus the top1-top2
margin. It also reports where named probe nations fall in the ranking. This
replaces the renderer's animated ``confidence`` and hardcoded
``recognised: false``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..backends.clip_retrieval import CLIPBackend


@dataclass(frozen=True)
class Retrieved:
    code: str
    name: str
    similarity: float


@dataclass(frozen=True)
class RecognitionRecord:
    intent: str
    temperature: float
    ranking: list[Retrieved]
    confidence: float
    margin: float
    probe_ranks: dict[str, tuple[int, float]]  # code -> (1-based rank, similarity)
    corpus_similarity: dict[str, float] = field(default_factory=dict)  # code -> cosine to query
    provenance: str = (
        "tna.analysis.recognition.recognition_record via averaged CLIP embedding "
        "and corpus cosine retrieval"
    )

    @property
    def nearest(self) -> Retrieved:
        return self.ranking[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "temperature": self.temperature,
            "confidence": self.confidence,
            "margin": self.margin,
            "ranking": [vars(r) for r in self.ranking],
            "probe_ranks": {code: list(value) for code, value in self.probe_ranks.items()},
            "provenance": self.provenance,
        }


def recognition_record(
    clip: CLIPBackend,
    codes: list[str],
    weights: dict[str, float],
    names: dict[str, str],
    *,
    intent: str = "",
    probes: tuple[str, ...] = (),
    temperature: float = 0.01,
    topk: int = 8,
) -> RecognitionRecord:
    """Retrieve and score the nearest real nations to an averaged embedding."""

    clip._ensure_corpus()
    corpus_codes = clip._corpus_codes
    averaged: np.ndarray | None = None
    for code in codes:
        term = weights[code] * clip.encode(code)
        averaged = term if averaged is None else averaged + term
    if averaged is None:
        raise ValueError("cannot recognise an empty selection")
    query = averaged / (np.linalg.norm(averaged) or 1.0)
    sims = clip._corpus_matrix @ query
    order = np.argsort(-sims)
    ranking = [
        Retrieved(corpus_codes[i], names.get(corpus_codes[i], corpus_codes[i]), float(sims[i]))
        for i in order[:topk]
    ]
    # Real confidence: a softmax over the corpus similarities at a stated
    # temperature. Low temperature sharpens; the value is reported, not tuned.
    scaled = sims / temperature
    scaled = scaled - scaled.max()
    probabilities = np.exp(scaled)
    probabilities /= probabilities.sum()
    confidence = float(probabilities[order[0]])
    margin = float(sims[order[0]] - sims[order[1]]) if len(order) > 1 else 0.0
    rank_of = {corpus_codes[idx]: rank for rank, idx in enumerate(order)}
    index_of = {code: i for i, code in enumerate(corpus_codes)}
    probe_ranks: dict[str, tuple[int, float]] = {}
    for probe in probes:
        if probe in rank_of:
            probe_ranks[probe] = (rank_of[probe] + 1, float(sims[index_of[probe]]))
    corpus_similarity = {corpus_codes[i]: float(sims[i]) for i in range(len(corpus_codes))}
    return RecognitionRecord(
        intent=intent,
        temperature=temperature,
        ranking=ranking,
        confidence=confidence,
        margin=margin,
        probe_ranks=probe_ranks,
        corpus_similarity=corpus_similarity,
    )
