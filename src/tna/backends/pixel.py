"""Rung 0: transparent per-pixel RGB averaging."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .base import Backend, BackendResult


class PixelBackend(Backend):
    name = "pixel"
    space = "per-pixel RGB"
    learned = False
    decodable = True

    def encode(self, code: str) -> np.ndarray:
        # Pixel space is the control condition: the representation is just the
        # raster image tensor, so decoding is clipping the averaged RGB values.
        return self.arrays[code]

    def decode(self, z: np.ndarray) -> BackendResult:
        image = Image.fromarray(np.clip(z, 0, 255).astype(np.uint8))
        return BackendResult(image=image, representation=z, trace={"synthesis": "weighted RGB average"})
