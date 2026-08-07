"""Real flag-component detection — "the machine parsing the flag into parts".

Classical computer vision over a single flag raster: dominant colour regions with
bounding boxes and shares (octree quantisation), stripe/band detection (row/column
colour-change runs), edge contours (Sobel), and bilateral symmetry scores (flag vs
its mirror). Every value is measured from the pixels; nothing is invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Region:
    rgb: tuple[int, int, int]
    hex: str
    share: float
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) normalised 0..1


@dataclass(frozen=True)
class ComponentRecord:
    code: str
    regions: list[Region]
    vertical: bool  # dominant stripe orientation
    boundaries: list[float]  # detected band boundaries (normalised along the axis)
    symmetry_h: float  # 0..1 mirror symmetry (left-right)
    symmetry_v: float  # 0..1 mirror symmetry (top-bottom)
    edge_density: float
    edges: np.ndarray = field(default=None, repr=False)  # HxW edge magnitude 0..1

    def edge_image(self) -> Image.Image:
        e = self.edges if self.edges is not None else np.zeros((2, 2))
        return Image.fromarray(np.clip(e * 255, 0, 255).astype(np.uint8))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "vertical": self.vertical,
            "symmetry_h": self.symmetry_h,
            "symmetry_v": self.symmetry_v,
            "edge_density": self.edge_density,
            "regions": [
                {"hex": r.hex, "share": r.share, "bbox": list(r.bbox)} for r in self.regions
            ],
            "provenance": "tna.analysis.components.component_record (octree regions, band runs, Sobel, mirror symmetry)",
        }


def component_record(code: str, array: np.ndarray, *, k: int = 6) -> ComponentRecord:
    """Detect colour regions, stripes, edges, and symmetry for one flag raster."""

    image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).convert("RGB")
    arr = np.asarray(image, dtype=np.float32)
    height, width, _ = arr.shape

    # Dominant colour regions with bounding boxes (octree quantisation).
    quant = image.quantize(colors=k, method=Image.FASTOCTREE)
    palette = quant.getpalette() or []
    labels = np.asarray(quant)
    counts = sorted(quant.getcolors() or [], reverse=True)
    total = sum(c for c, _ in counts) or 1
    regions: list[Region] = []
    for count, index in counts[:k]:
        r, g, b = palette[index * 3: index * 3 + 3]
        ys, xs = np.where(labels == index)
        if len(xs) == 0:
            continue
        bbox = (float(xs.min()) / width, float(ys.min()) / height,
                float(xs.max()) / width, float(ys.max()) / height)
        regions.append(Region((int(r), int(g), int(b)), f"#{r:02x}{g:02x}{b:02x}", count / total, bbox))

    # Stripe orientation + band boundaries from colour-change runs.
    rows = arr.mean(axis=1)  # (H, 3) mean colour per row
    cols = arr.mean(axis=0)  # (W, 3) mean colour per column
    rdiff = np.abs(np.diff(rows, axis=0)).sum(axis=1)
    cdiff = np.abs(np.diff(cols, axis=0)).sum(axis=1)
    vertical = float(cdiff.sum()) > float(rdiff.sum())
    series = cdiff if vertical else rdiff
    threshold = series.mean() + series.std()
    boundaries = [float(i + 1) / len(series) for i, v in enumerate(series) if v > threshold]

    # Bilateral symmetry (1 = perfect mirror).
    symmetry_h = 1.0 - float(np.abs(arr - arr[:, ::-1]).mean()) / 255.0
    symmetry_v = 1.0 - float(np.abs(arr - arr[::-1, :]).mean()) / 255.0

    # Edge magnitude (Sobel-ish finite difference).
    gray = arr.mean(axis=2)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    mag = np.sqrt(gx * gx + gy * gy)
    mag = mag / (mag.max() or 1.0)
    edge_density = float((mag > 0.16).mean())

    return ComponentRecord(
        code=code,
        regions=regions,
        vertical=vertical,
        boundaries=boundaries,
        symmetry_h=symmetry_h,
        symmetry_v=symmetry_v,
        edge_density=edge_density,
        edges=mag.astype(np.float32),
    )
