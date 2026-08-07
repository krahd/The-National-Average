"""Rung 5: deterministic structured SVG-spec averaging."""

from __future__ import annotations

import io
from pathlib import Path

import cairosvg
import numpy as np
from PIL import Image

from .base import Backend, BackendResult
from .palette import dominant_palette_from_array


class SVGBackend(Backend):
    name = "svg"
    space = "structured SVG program spec"
    learned = False
    decodable = True

    def __init__(self, arrays: dict[str, np.ndarray], seed: int = 0, palette_size: int = 4, renderer: str = "deterministic"):
        super().__init__(arrays, seed=seed)
        self.palette_size = palette_size
        self.renderer = renderer
        self.latent_dim = palette_size * 4 + 1

    def encode(self, code: str) -> np.ndarray:
        # This is a structured procedural representation, not the original SVG
        # DOM. It keeps colour shares plus one layout hint that can be averaged.
        palette = dominant_palette_from_array(self.arrays[code], self.palette_size)
        vector: list[float] = []
        for color in palette:
            red, green, blue = color["rgb"]
            vector.extend([red / 255.0, green / 255.0, blue / 255.0, float(color["share"])])
        while len(vector) < self.palette_size * 4:
            vector.extend([1.0, 1.0, 1.0, 0.0])
        array = self.arrays[code]
        vertical_energy = float(np.abs(np.diff(array.mean(axis=0), axis=0)).mean())
        horizontal_energy = float(np.abs(np.diff(array.mean(axis=1), axis=0)).mean())
        # The orientation scalar estimates whether strong colour changes run
        # left-right or top-bottom. Averaging it lets the decoder choose bands.
        orientation = vertical_energy / ((vertical_energy + horizontal_energy) or 1.0)
        vector.append(orientation)
        return np.asarray(vector[: self.latent_dim], dtype=np.float64)

    def decode(self, z: np.ndarray) -> BackendResult:
        # The deterministic renderer is preferred for auditability. The CLI
        # accepts "llm" as a named mode, but this backend still falls back to
        # deterministic SVG generation rather than depending on an API call.
        height, width, _ = next(iter(self.arrays.values())).shape
        svg = self.to_svg(z, width, height)
        png = cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=width, output_height=height)
        image = Image.open(io.BytesIO(png)).convert("RGB")
        return BackendResult(
            image=image,
            representation=z,
            trace={
                "synthesis": "deterministic SVG from averaged colour/layout spec",
                "svg_renderer": self.renderer,
            },
            files={"svg_text": svg},
        )

    def to_svg(self, z: np.ndarray, width: int, height: int) -> str:
        # SVG output is kept tiny and inspectable: a reader can open the trace
        # and see every rectangle produced by the averaged representation.
        orientation = "vertical" if float(z[-1]) >= 0.5 else "horizontal"
        colors = []
        shares = []
        for index in range(0, self.palette_size * 4, 4):
            red, green, blue, share = z[index : index + 4]
            colors.append("#%02x%02x%02x" % tuple(int(max(0, min(255, round(c * 255)))) for c in (red, green, blue)))
            shares.append(max(0.0, float(share)))
        total = sum(shares) or 1.0
        cursor = 0.0
        rects = []
        for color, share in zip(colors, shares):
            span = share / total
            if orientation == "vertical":
                rects.append(f'<rect x="{cursor * width:.3f}" y="0" width="{span * width:.3f}" height="{height}" fill="{color}"/>')
            else:
                rects.append(f'<rect x="0" y="{cursor * height:.3f}" width="{width}" height="{span * height:.3f}" fill="{color}"/>')
            cursor += span
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}">'
            + "".join(rects)
            + "</svg>"
        )


def write_svg(path: Path, svg_text: str) -> None:
    path.write_text(svg_text + "\n", encoding="utf-8")
