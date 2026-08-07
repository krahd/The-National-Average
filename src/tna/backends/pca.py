"""Rung 2: deterministic PCA / eigenflag averaging with NumPy SVD."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .base import Backend, BackendResult


class PCABackend(Backend):
    name = "pca"
    space = "linear eigenflag subspace"
    learned = True
    decodable = True

    def __init__(self, arrays: dict[str, np.ndarray], seed: int = 0, components: int = 32):
        super().__init__(arrays, seed=seed)
        self.codes = list(arrays)
        # PCA learns a linear coordinate system from the full raster corpus.
        # The rows are flags, the columns are flattened RGB pixel channels.
        matrix = np.stack([arrays[code].reshape(-1) / 255.0 for code in self.codes])
        self.mean = matrix.mean(axis=0)
        centered = matrix - self.mean
        # NumPy SVD is deterministic for a fixed matrix and avoids adding a
        # larger machine-learning dependency to the base prototype.
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        self.components = vt[: min(components, vt.shape[0])]
        self.latent_dim = int(self.components.shape[0])
        self.model_checkpoint = "fit-at-runtime:numpy-svd"
        self.shape = next(iter(arrays.values())).shape

    def encode(self, code: str) -> np.ndarray:
        # Projection answers: where does this flag sit in the eigenflag basis?
        vector = self.arrays[code].reshape(-1) / 255.0
        return (vector - self.mean) @ self.components.T

    def decode(self, z: np.ndarray) -> BackendResult:
        # The inverse projection returns to pixel space. Clipping is necessary
        # because a linear average can leave the displayable 0..1 range.
        vector = self.mean + z @ self.components
        array = np.clip(vector.reshape(self.shape) * 255.0, 0, 255).astype(np.uint8)
        return BackendResult(
            image=Image.fromarray(array),
            representation=z,
            trace={"synthesis": "inverse PCA projection from averaged coordinates"},
        )

    def eigenflag_images(self, count: int = 8) -> list[Image.Image]:
        # Components have positive and negative values; normalising each one
        # independently makes its spatial structure inspectable as an image.
        images = []
        for component in self.components[:count]:
            values = component.reshape(self.shape)
            values = (values - values.min()) / ((values.max() - values.min()) or 1.0)
            images.append(Image.fromarray(np.clip(values * 255, 0, 255).astype(np.uint8)))
        return images

    def embedding_points(self, codes: list[str]) -> dict[str, tuple[float, float]]:
        points = {}
        for code in codes:
            z = self.encode(code)
            x = float(z[0]) if len(z) else 0.0
            y = float(z[1]) if len(z) > 1 else 0.0
            points[code] = (x, y)
        return points
