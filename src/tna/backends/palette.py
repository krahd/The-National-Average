"""Rung 1: dominant-colour / coarse-spec averaging."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .base import Backend, BackendResult


def dominant_palette_from_array(array: np.ndarray, k: int = 5) -> list[dict[str, object]]:
    """Extract a small colour summary from a raster flag.

    The output is not a semantic model of the flag. It is a compact procedural
    representation: dominant RGB values plus their image shares.
    """

    image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    quantized = image.quantize(colors=k, method=Image.FASTOCTREE)
    palette = quantized.getpalette()
    counts = sorted(quantized.getcolors(), key=lambda item: -item[0])
    total = sum(count for count, _ in counts) or 1
    output = []
    for count, index in counts:
        red, green, blue = palette[index * 3 : index * 3 + 3]
        output.append(
            {
                "rgb": [red, green, blue],
                "hex": f"#{red:02x}{green:02x}{blue:02x}",
                "share": count / total,
            }
        )
    return output


class PaletteBackend(Backend):
    name = "palette"
    space = "dominant-colour vector and stripe shares"
    learned = False
    decodable = True

    def __init__(self, arrays: dict[str, np.ndarray], seed: int = 0, palette_size: int = 5):
        super().__init__(arrays, seed=seed)
        self.palette_size = palette_size
        self.latent_dim = palette_size * 4

    def encode(self, code: str) -> np.ndarray:
        # Layout information is deliberately coarse. Each colour contributes
        # [red, green, blue, share], and missing slots are padded as white with
        # zero share so every flag has the same vector length.
        palette = dominant_palette_from_array(self.arrays[code], self.palette_size)
        vector: list[float] = []
        for color in palette:
            red, green, blue = color["rgb"]
            vector.extend([red / 255.0, green / 255.0, blue / 255.0, float(color["share"])])
        while len(vector) < self.latent_dim:
            vector.extend([1.0, 1.0, 1.0, 0.0])
        return np.asarray(vector[: self.latent_dim], dtype=np.float64)

    def decode(self, z: np.ndarray) -> BackendResult:
        # Decoding makes the lossy representation visible: it renders only the
        # averaged palette shares as vertical bands, not a reconstructed flag.
        height, width, _ = next(iter(self.arrays.values())).shape
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        colors = []
        shares = []
        for index in range(0, len(z), 4):
            red, green, blue, share = z[index : index + 4]
            colors.append(tuple(int(max(0, min(255, round(channel * 255)))) for channel in (red, green, blue)))
            shares.append(max(0.0, float(share)))
        total = sum(shares) or 1.0
        x = 0
        for color, share in zip(colors, shares):
            next_x = x + int(round(width * (share / total)))
            draw.rectangle([x, 0, min(width, next_x), height], fill=color)
            x = next_x
        if x < width:
            draw.rectangle([x, 0, width, height], fill=colors[-1])
        return BackendResult(
            image=image,
            representation=z,
            trace={"synthesis": "dominant colours rendered as proportional vertical bands"},
        )
