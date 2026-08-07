"""Backend contract for representation-space averaging.

The central experiment in this repository is deliberately small:

1. Convert each selected flag into a vector-like representation.
2. Compute the same weighted average in that representation.
3. Convert the averaged representation back into an artefact if possible.

Backends differ in *what their vectors mean*, not in the averaging operation.
That makes visual differences between outputs attributable to representation
space, rather than to a hidden backend-specific blending rule.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image


def barycentric_average(zs: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Compute ``z_avg = sum_i w_i z_i`` for every backend.

    "Barycentric" means a weighted point inside the coordinate system defined
    by the source points. The function is intentionally boring: if an output
    is surprising, the explanation should be the representation, not a special
    averaging trick.
    """
    out = None
    for code, z in zs.items():
        term = weights[code] * np.asarray(z, dtype=np.float64)
        out = term if out is None else out + term
    if out is None:
        raise ValueError("cannot average an empty representation set")
    return out


@dataclass
class BackendResult:
    """Standard result object returned by all visual backends.

    Some backends decode to images, some may return retrieval results, and
    optional heavy backends may only return an unavailable trace. Keeping those
    cases in one shape lets the CLI write comparable provenance records.
    """

    image: Image.Image | None
    representation: Any
    trace: dict[str, Any] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    retrieval: dict[str, Any] | None = None
    status: str = "ok"
    reason: str | None = None


class BackendUnavailable(RuntimeError):
    pass


class Backend(ABC):
    """Interface for one rung of the representation ladder.

    Subclasses implement only ``encode`` and ``decode``. The inherited
    ``average`` method performs the shared barycentric average and appends
    common trace fields such as latent dimension, checkpoint, and seed.
    """

    name: str = "backend"
    space: str = "unspecified"
    learned: bool = False
    decodable: bool = True
    model_checkpoint: str | None = None
    latent_dim: int | None = None

    def __init__(self, arrays: dict[str, np.ndarray], seed: int = 0):
        self.arrays = arrays
        self.seed = seed

    @abstractmethod
    def encode(self, code: str) -> np.ndarray:
        ...

    @abstractmethod
    def decode(self, z: np.ndarray) -> BackendResult:
        ...

    def average(self, codes: list[str], weights: dict[str, float]) -> BackendResult:
        # The CLI passes already-normalised weights; traces keep both raw and
        # normalised values so runs remain auditable after the fact.
        zs = {code: self.encode(code) for code in codes}
        z_avg = barycentric_average(zs, weights)
        result = self.decode(z_avg)
        result.trace.update(
            {
                "backend": self.name,
                "space": self.space,
                "learned": self.learned,
                "decodable": self.decodable,
                "model_checkpoint": self.model_checkpoint,
                "latent_dim": self.latent_dim,
                "seed": self.seed,
                "encoded_shape": list(np.asarray(z_avg).shape),
            }
        )
        return result


class UnavailableBackend(Backend):
    """Trace-only backend used when optional dependencies/assets are absent."""

    decodable = False

    def __init__(self, name: str, space: str, reason: str, seed: int = 0):
        super().__init__({}, seed=seed)
        self.name = name
        self.space = space
        self.reason = reason

    def encode(self, code: str) -> np.ndarray:
        raise BackendUnavailable(self.reason)

    def decode(self, z: np.ndarray) -> BackendResult:
        raise BackendUnavailable(self.reason)

    def average(self, codes: list[str], weights: dict[str, float]) -> BackendResult:
        return BackendResult(
            image=None,
            representation=None,
            status="unavailable",
            reason=self.reason,
            trace={
                "backend": self.name,
                "space": self.space,
                "learned": True,
                "decodable": False,
                "model_checkpoint": None,
                "latent_dim": None,
                "seed": self.seed,
            },
        )
