"""Real weight-distribution accounting for the averaging weightings.

Given a weighting intent (``population``, ``gdp``, ``cumulative_co2`` ...), this
records which polities dominate the average and which are driven below a stated
weight threshold — the erasure that a "neutral" average performs. Every value is
derived from :func:`tna.weights.weights_from_intent`; nothing is invented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..data import Polity
from ..weights import weights_from_intent


@dataclass(frozen=True)
class Contributor:
    code: str
    name: str
    weight: float


@dataclass(frozen=True)
class ErasureRecord:
    intent: str
    source: str
    threshold: float
    total: int
    contributors: list[Contributor]  # all selected, sorted descending by weight
    erased: list[Contributor]  # weight strictly below threshold
    top5_share: float
    provenance: str = (
        "tna.analysis.erasure.erasure_record via weights.weights_from_intent"
    )

    @property
    def erased_count(self) -> int:
        return len(self.erased)

    def top(self, n: int) -> list[Contributor]:
        return self.contributors[:n]

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "source": self.source,
            "threshold": self.threshold,
            "total": self.total,
            "erased_count": self.erased_count,
            "top5_share": self.top5_share,
            "contributors": [vars(c) for c in self.contributors],
            "provenance": self.provenance,
        }


def erasure_record(
    intent: str, selected: list[Polity], *, threshold: float = 0.001
) -> ErasureRecord:
    """Compute the real contributor/erased split for one weighting intent."""

    run = weights_from_intent(intent, selected)
    names = {polity.code: polity.name for polity in selected}
    ordered = sorted(run.weights.items(), key=lambda item: -item[1])
    contributors = [
        Contributor(code, names.get(code, code), float(weight))
        for code, weight in ordered
    ]
    erased = [contributor for contributor in contributors if contributor.weight < threshold]
    top5_share = float(sum(weight for _, weight in ordered[:5]))
    return ErasureRecord(
        intent=run.name,
        source=run.source,
        threshold=threshold,
        total=len(contributors),
        contributors=contributors,
        erased=erased,
        top5_share=top5_share,
    )
