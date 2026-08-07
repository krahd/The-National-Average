"""Hypermodern image treatments: grade, grain, glitch, aberration, bloom.

These operate on *imagery only* — they are aesthetic treatments of already-real
frames and never produce a displayed number. Randomness is permitted here (the
no-fabrication invariant guards ``scenes.py``, not this module), but every effect
is driven by a seeded ``numpy`` generator so renders stay deterministic.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def to_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")


def vignette_mask(size: tuple[int, int], strength: float = 0.45) -> np.ndarray:
    """Precomputed radial darkening mask in [0,1], shape (h, w, 1)."""

    width, height = size
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx, cy = width / 2.0, height / 2.0
    dist = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    mask = 1.0 - strength * np.clip(dist - 0.35, 0.0, 1.2)
    return np.clip(mask, 0.0, 1.0)[:, :, None]


def grade(array: np.ndarray) -> np.ndarray:
    """Cold institutional grade: slight teal lift and added contrast."""

    out = array.copy()
    out[:, :, 0] *= 0.97
    out[:, :, 1] *= 1.01
    out[:, :, 2] *= 1.05
    # Offset lifted (122 -> 130) so text/midtones read brighter; the lower
    # contrast keeps blacks near-black after clipping.
    return (out - 128.0) * 1.06 + 130.0


def bloom(array: np.ndarray, threshold: float = 205.0, radius: float = 6.0, strength: float = 0.45) -> np.ndarray:
    bright = np.clip(array - threshold, 0.0, None)
    blurred = np.asarray(
        to_image(bright).filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32
    )
    return array + blurred * strength


def chromatic_aberration(array: np.ndarray, shift: int = 2) -> np.ndarray:
    out = array.copy()
    out[:, :, 0] = np.roll(array[:, :, 0], shift, axis=1)
    out[:, :, 2] = np.roll(array[:, :, 2], -shift, axis=1)
    return out


def scanlines(array: np.ndarray, strength: float = 0.10) -> np.ndarray:
    modulation = np.ones(array.shape[0], dtype=np.float32)
    modulation[::2] = 1.0 - strength
    return array * modulation[:, None, None]


def grain(array: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, amount, array.shape[:2]).astype(np.float32)
    return array + noise[:, :, None]


def block_glitch(array: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """Datamosh-style horizontal block displacement with channel splits."""

    if amount <= 0.0:
        return array
    height, width, _ = array.shape
    out = array.copy()
    # Many thin bands with large but brief jumps — snappier and faster-reading
    # than a few thick sustained shifts.
    bands = int(3 + amount * 24)
    for _ in range(bands):
        y = int(rng.integers(0, height))
        band_h = int(rng.integers(1, max(2, int(height * 0.045))))
        y2 = min(height, y + band_h)
        out[y:y2] = np.roll(out[y:y2], int(rng.normal(0, 10 + 70 * amount)), axis=1)
        if rng.random() < 0.75:
            channel = int(rng.integers(0, 3))
            out[y:y2, :, channel] = np.roll(
                out[y:y2, :, channel], int(rng.normal(0, 12 + 40 * amount)), axis=1
            )
    return out


def channel_split(array: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """Hard RGB separation — the image coming apart into its channels."""

    out = array.copy()
    dx = int(4 + 26 * amount)
    out[:, :, 0] = np.roll(array[:, :, 0], dx, axis=1)
    out[:, :, 2] = np.roll(array[:, :, 2], -dx, axis=1)
    out[:, :, 1] = np.roll(out[:, :, 1], int(rng.normal(0, 3 + 10 * amount)), axis=0)
    return out


def pixel_sort(array: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """Sort whole rows by luminance — streaked, processed-looking corruption."""

    out = array.copy()
    height = array.shape[0]
    luma = array.mean(axis=2)
    rows = np.unique(rng.integers(0, height, size=max(1, int(height * min(0.6, amount)))))
    for y in rows:
        out[y] = out[y][np.argsort(luma[y])]
    return out


def patch_shuffle(array: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """Swap 32px patches — echoes the ViT tokenisation the film visualises."""

    out = array.copy()
    height, width, _ = array.shape
    ps = 32
    rows, cols = height // ps, width // ps
    if rows < 2 or cols < 2:
        return out
    for _ in range(max(1, int(rows * cols * min(0.5, amount * 0.6)))):
        r1, c1 = int(rng.integers(0, rows)), int(rng.integers(0, cols))
        r2, c2 = int(rng.integers(0, rows)), int(rng.integers(0, cols))
        a = out[r1 * ps:(r1 + 1) * ps, c1 * ps:(c1 + 1) * ps].copy()
        out[r1 * ps:(r1 + 1) * ps, c1 * ps:(c1 + 1) * ps] = out[r2 * ps:(r2 + 1) * ps, c2 * ps:(c2 + 1) * ps]
        out[r2 * ps:(r2 + 1) * ps, c2 * ps:(c2 + 1) * ps] = a
    return out


def latent_noise(array: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """Posterise + low-res structured noise — a 'latent corruption' look."""

    levels = max(2, int(8 - amount * 6))
    out = np.round(array / 255.0 * levels) / levels * 255.0
    height, width, _ = array.shape
    small = rng.normal(0, 40 * amount, (max(1, height // 12), max(1, width // 12), 3)).astype(np.float32) + 128.0
    big = np.asarray(
        Image.fromarray(np.clip(small, 0, 255).astype(np.uint8)).resize((width, height), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) - 128.0
    return out + big


GLITCH_MODES = ("block", "channel", "pixel_sort", "patch_shuffle", "latent")


def apply_glitch(array: np.ndarray, amount: float, mode: str, rng: np.random.Generator) -> np.ndarray:
    if amount <= 0.01:
        return array
    if mode == "channel":
        return channel_split(array, amount, rng)
    if mode == "pixel_sort":
        return pixel_sort(array, amount, rng)
    if mode == "patch_shuffle":
        return patch_shuffle(array, amount, rng)
    if mode == "latent":
        return latent_noise(array, amount, rng)
    return block_glitch(array, amount, rng)


def treat(
    image: Image.Image,
    *,
    vmask: np.ndarray,
    glitch: float = 0.0,
    glitch_mode: str = "block",
    seed: int = 0,
    frame: int = 0,
    grain_amount: float = 6.0,
    aberration: int = 2,
) -> Image.Image:
    """Compose the hypermodern treatment over one real frame.

    The ``glitch`` amount and ``glitch_mode`` are supplied by the renderer from
    real per-phase signals, so corruption is sparse, varied, and meaningful
    rather than a constant per-frame sprinkle. A seeded generator is built here
    (not in the scenes) so the renderer stays free of randomness while treatments
    remain deterministic.
    """

    rng = np.random.default_rng(seed + frame)
    array = to_array(image)
    array = grade(array)
    array = bloom(array)
    array = apply_glitch(array, glitch, glitch_mode, rng)
    array = chromatic_aberration(array, aberration + int(glitch * 2))
    array = scanlines(array, 0.07)
    array = grain(array, grain_amount + glitch * 6.0, rng)
    array = array * vmask
    return to_image(array)
