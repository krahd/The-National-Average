"""What each representation discards.

The residual is the honest counterpart of a reconstruction: original minus
decode. For PCA it exposes the structure the linear basis cannot hold; the
palette residual exposes what dominant-colour banding throws away. Surfacing the
residual makes the lossiness of "averaging" visible instead of hiding it behind a
clean output. Reuses the existing backends' encode/decode round-trip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from ..backends.palette import PaletteBackend
from ..backends.pca import PCABackend


@dataclass(frozen=True)
class ResidualRecord:
    code: str
    space: str
    residual: np.ndarray  # HxW absolute error magnitude in [0, 255]
    energy: float  # mean absolute error (0..255): a real scalar measurement
    provenance: str = (
        "tna.analysis.residual via backend encode/decode round-trip (|original - decode|)"
    )

    def heatmap(self) -> Image.Image:
        values = self.residual
        norm = values / (values.max() or 1.0)
        return Image.fromarray(np.clip(norm * 255, 0, 255).astype(np.uint8))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "space": self.space,
            "energy": self.energy,
            "provenance": self.provenance,
        }


def _abs_error(original: np.ndarray, reconstruction: np.ndarray) -> np.ndarray:
    diff = np.abs(original.astype(np.float64) - reconstruction.astype(np.float64))
    return diff.mean(axis=2) if diff.ndim == 3 else diff


def pca_residual(pca: PCABackend, code: str) -> ResidualRecord:
    original = pca.arrays[code]
    reconstruction = np.asarray(pca.decode(pca.encode(code)).image, dtype=np.float64)
    error = _abs_error(original, reconstruction)
    return ResidualRecord(
        code=code,
        space="linear eigenflag subspace",
        residual=error,
        energy=float(error.mean()),
    )


def palette_residual(palette: PaletteBackend, code: str) -> ResidualRecord:
    original = palette.arrays[code]
    reconstruction = np.asarray(palette.decode(palette.encode(code)).image, dtype=np.float64)
    error = _abs_error(original, reconstruction)
    return ResidualRecord(
        code=code,
        space="dominant-colour bands",
        residual=error,
        energy=float(error.mean()),
    )
