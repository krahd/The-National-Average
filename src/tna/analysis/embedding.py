"""Real CLIP embedding geometry for the corpus.

Projects the corpus CLIP embeddings to 2-D with a deterministic NumPy PCA and
builds a cosine k-nearest-neighbour graph. The renderer uses this to lay flags
out at their actual learned semantic positions and to draw real similarity
edges, replacing the former random scatter and decorative graph lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..backends.clip_retrieval import CLIPBackend


@dataclass(frozen=True)
class EmbeddingGeometry:
    codes: list[str]
    coords2d: dict[str, tuple[float, float]]
    knn: dict[str, list[tuple[str, float]]]
    provenance: str = (
        "tna.analysis.embedding.embedding_geometry via CLIP image embeddings "
        "(PCA-2 of L2-normalised vectors; cosine kNN)"
    )

    def normalised_coords(self) -> dict[str, tuple[float, float]]:
        """Coordinates rescaled into the unit square for layout convenience."""

        if not self.coords2d:
            return {}
        xs = [point[0] for point in self.coords2d.values()]
        ys = [point[1] for point in self.coords2d.values()]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)

        def norm(value: float, low: float, high: float) -> float:
            return (value - low) / ((high - low) or 1.0)

        return {
            code: (norm(x, x0, x1), norm(y, y0, y1))
            for code, (x, y) in self.coords2d.items()
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "codes": self.codes,
            "coords2d": {code: list(point) for code, point in self.coords2d.items()},
            "knn": {code: [list(n) for n in nbrs] for code, nbrs in self.knn.items()},
            "provenance": self.provenance,
        }


def embedding_geometry(
    clip: CLIPBackend, codes: list[str], *, k: int = 3
) -> EmbeddingGeometry:
    """Compute a real 2-D layout and cosine kNN graph over the CLIP embeddings."""

    matrix = np.stack([clip.encode(code) for code in codes])  # (N, 512) unit vectors
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    # Deterministic PCA via SVD: the first two right-singular vectors span the
    # plane of greatest variance in CLIP space. No scipy dependency required.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    coords2d = {
        code: (float(coords[i, 0]), float(coords[i, 1]))
        for i, code in enumerate(codes)
    }
    # Cosine equals the dot product because the embeddings are L2-normalised.
    cosine = matrix @ matrix.T
    knn: dict[str, list[tuple[str, float]]] = {}
    for i, code in enumerate(codes):
        order = np.argsort(-cosine[i])
        neighbours = [(codes[j], float(cosine[i, j])) for j in order if j != i][:k]
        knn[code] = neighbours
    return EmbeddingGeometry(codes=list(codes), coords2d=coords2d, knn=knn)
