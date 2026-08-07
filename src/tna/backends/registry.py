"""Backend construction."""

from __future__ import annotations

from .clip_retrieval import build_clip_backend
from .concept import build_concept_backend
from .palette import PaletteBackend
from .pca import PCABackend
from .pixel import PixelBackend
from .sdvae import build_sdvae_backend
from .svg import SVGBackend
from .vae import build_vae_backend


AVAILABLE_BACKENDS = ["pixel", "palette", "pca", "vae", "sdvae", "svg", "concept", "clip", "music_vae"]


def build_backend(
    name: str,
    arrays,
    *,
    seed: int,
    palette_size: int,
    pca_components: int,
    svg_renderer: str,
    train_vae: bool,
    vae_epochs: int,
):
    """Factory used by the CLI to construct backends from command-line names."""

    if name == "pixel":
        return PixelBackend(arrays, seed=seed)
    if name == "palette":
        return PaletteBackend(arrays, seed=seed, palette_size=palette_size)
    if name == "pca":
        return PCABackend(arrays, seed=seed, components=pca_components)
    if name == "svg":
        return SVGBackend(arrays, seed=seed, palette_size=palette_size, renderer=svg_renderer)
    if name == "vae":
        return build_vae_backend(arrays, seed=seed, train=train_vae, epochs=vae_epochs)
    if name == "sdvae":
        return build_sdvae_backend(arrays, seed=seed)
    if name == "concept":
        return build_concept_backend(arrays, seed=seed)
    if name == "clip":
        return build_clip_backend(arrays, seed=seed)
    if name == "music_vae":
        from ..music.music_vae import unavailable_trace
        from .base import UnavailableBackend

        trace = unavailable_trace()
        return UnavailableBackend("music_vae", trace["space"], trace["reason"], seed=seed)
    raise ValueError(f"unknown backend: {name}")
