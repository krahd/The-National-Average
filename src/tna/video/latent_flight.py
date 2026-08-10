"""A separate 3-D latent-space film for *The National Average*.

This renderer deliberately shares no scene language with the canonical or
NeurIPS cuts.  It treats computed representations as a navigable world:

* the complete corpus occupies repeated weather systems derived from its first
  three real PCA/eigenflag coordinates;
* Palestine, Israel, the United States, and Germany become diffuse colour-density
  volumes assembled from source flags and truncated reconstructions;
* eigenflags and Stable Diffusion VAE tensors become moving particulate fields;
* a continuously reweighted PCA barycentre becomes a volumetric calm front that
  recedes as the camera searches for it;
* trace-backed annotations turn the flight into a legible computational route
  through selection, representation, commensuration, reconstruction, weighting,
  erasure, retrieval, synthesis, and non-settlement.

The camera, cuts, deformation, colour, and glitches are artistic operations.
Coordinates, reconstructions, eigenflags, and latent samples come from the real
backends and are written to a provenance manifest.
"""

from __future__ import annotations

import gc
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..analysis import erasure_record
from ..backends.pca import PCABackend
from ..backends.sdvae import SDVAEBackend
from ..data import DATA_DIR, corpus_arrays, load_corpus, rasterize_flag
from ..trace import write_json
from ..weights import weights_from_intent
from . import effects
from .compositor import load_font, render_video_stream
from .foundation import prepare_foundation_assets
from .neurips_metrics import stats_for
from .pipeline import INTENT_LABELS, INTENTS, VideoRenderConfig, build_video_backend, production_codes


TITLE = "The National Average — Latent Flight"
AUTHOR = "Tomas Laurenzo"
WEBSITE = "laurenzo.net"
FOCUS_CODES = ("ps", "il", "us", "de")
FOCUS_COLOURS = {
    "ps": (206, 34, 51),
    "il": (28, 78, 177),
    "us": (178, 34, 52),
    "de": (255, 206, 0),
}
LATENT_CANVAS = (96, 72)
PCA_COMPONENTS = 32
RECONSTRUCTION_COMPONENTS = (1, 2, 4, 8, 16, 32)
WORLD_START = -42.0
WORLD_END = 252.0
WORLD_FAR = 372.0

TITLE_END_FRACTION = 0.07
STATEMENT_END_FRACTION = 0.15
SEARCH_END_FRACTION = 0.84

WORLD_CHAPTERS = (
    ("boundary", 1.0),
    ("encode", 1.2),
    ("contested", 1.5),
    ("reconstruct", 1.3),
    ("weight", 1.6),
    ("erase", 1.2),
    ("retrieve", 1.1),
    ("synthesize", 1.0),
    ("unresolved", 0.8),
)

CHAPTER_LABELS = {
    "boundary": "SELECTION",
    "encode": "REPRESENTATION",
    "contested": "COMMENSURATION",
    "reconstruct": "RECONSTRUCTION",
    "weight": "WEIGHT / POWER",
    "erase": "THRESHOLD / ERASURE",
    "retrieve": "FOUNDATION-MODEL VERDICT",
    "synthesize": "MOVING AVERAGE",
    "unresolved": "NO SETTLEMENT",
}

ROUTE_LABELS = {
    "boundary": "SELECT",
    "encode": "ENCODE",
    "contested": "PAIR",
    "reconstruct": "RECON",
    "weight": "WEIGHT",
    "erase": "ERASE",
    "retrieve": "RETRIEVE",
    "synthesize": "SYNTH",
    "unresolved": "UNRESOLVED",
}

# Weighted rather than timed in seconds, so preview and production preserve the
# same cut structure. Cuts are intentionally hard; this is not a slide sequence.
FLIGHT_SHOTS = (
    ("breach", 0.72),
    ("corpus_dive", 0.80),
    ("palestine_field", 0.74),
    ("collision", 0.38),
    ("palestine_field", 0.48),
    ("collision", 0.31),
    ("israel_field", 0.65),
    ("axis_snap", 0.30),
    ("corpus_dive", 0.46),
    ("american_scale", 0.72),
    ("axis_snap", 0.26),
    ("german_basis", 0.70),
    ("sdvae_interior", 0.62),
    ("collision", 0.29),
    ("american_scale", 0.46),
    ("german_basis", 0.42),
    ("sdvae_interior", 0.72),
    ("false_calm", 0.66),
    ("parameter_break", 0.28),
    ("corpus_dive", 0.40),
    ("pursuit", 0.56),
    ("collision", 0.24),
    ("pursuit", 0.48),
    ("sdvae_interior", 0.44),
    ("parameter_break", 0.24),
    ("pursuit", 0.52),
    ("false_calm", 0.38),
    ("parameter_break", 0.23),
    ("pursuit", 0.48),
    ("almost_rest", 0.64),
    ("escape", 0.44),
)


def flight_segments(duration: float) -> list[tuple[str, float, float]]:
    """Return contiguous hard-cut segments spanning ``duration``."""

    total = sum(weight for _, weight in FLIGHT_SHOTS)
    out: list[tuple[str, float, float]] = []
    acc = 0.0
    for key, weight in FLIGHT_SHOTS:
        start = acc / total * duration
        acc += weight
        out.append((key, start, acc / total * duration))
    return out


def chapter_segments(duration: float) -> list[tuple[str, float, float]]:
    """Resolve the annotated computational route across the film duration."""

    total = sum(weight for _, weight in WORLD_CHAPTERS)
    out: list[tuple[str, float, float]] = []
    acc = 0.0
    for key, weight in WORLD_CHAPTERS:
        start = acc / total * duration
        acc += weight
        out.append((key, start, acc / total * duration))
    return out


def film_phase_segments(duration: float) -> list[tuple[str, float, float]]:
    """Return the opening, search, and conclusion structure for the film."""

    return [
        ("title", 0.0, duration * TITLE_END_FRACTION),
        ("statement", duration * TITLE_END_FRACTION, duration * STATEMENT_END_FRACTION),
        ("search", duration * STATEMENT_END_FRACTION, duration * SEARCH_END_FRACTION),
        ("conclusion", duration * SEARCH_END_FRACTION, duration),
    ]


def accelerated_search_progress(progress: float) -> float:
    """Map elapsed search time to a strictly accelerating forward distance."""

    return float(np.clip(progress, 0.0, 1.0)) ** 1.85


def parameter_weights(progress: float) -> np.ndarray:
    """Four changing artistic parameters, normalised as barycentric weights.

    The formula is deterministic and exists to make the pursued average move;
    it is never presented as a measured political quantity.
    """

    u = float(progress) % 1.0
    phases = np.array((0.0, 1.7, 3.6, 5.1), dtype=np.float64)
    speeds = np.array((1.0, 1.37, 0.73, 1.91), dtype=np.float64)
    logits = (
        1.15 * np.sin((2.0 * np.pi * u) * speeds + phases)
        + 0.72 * np.sin((2.0 * np.pi * u) * (speeds * 3.1) - phases * 0.6)
        + 0.30 * np.cos((2.0 * np.pi * u) * 11.0 + phases)
    )
    logits -= logits.max()
    values = np.exp(logits)
    return values / values.sum()


def _normalise_world(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centre = np.median(scores, axis=0)
    spread = np.quantile(np.abs(scores - centre), 0.90, axis=0)
    spread = np.where(spread < 1e-8, 1.0, spread)
    return (scores - centre) / spread * 6.2, centre, spread


def _nearest_edges(points: np.ndarray, neighbours: int = 2) -> list[tuple[int, int]]:
    delta = points[:, None, :] - points[None, :, :]
    distances = np.sum(delta * delta, axis=2)
    np.fill_diagonal(distances, np.inf)
    edges: set[tuple[int, int]] = set()
    for i in range(len(points)):
        for j in np.argsort(distances[i])[:neighbours]:
            edges.add(tuple(sorted((i, int(j)))))
    return sorted(edges)


def _sdvae_cloud(latent: np.ndarray) -> np.ndarray:
    """Sample a real 4xHxW SD-VAE tensor into local xyz/value/channel rows."""

    z = np.asarray(latent, dtype=np.float64)
    if z.ndim != 3:
        return np.empty((0, 5), dtype=np.float64)
    channels, height, width = z.shape
    scale = float(np.std(z)) or 1.0
    rows = []
    step_y = max(1, height // 9)
    step_x = max(1, width // 12)
    for channel in range(channels):
        for y in range(0, height, step_y):
            for x in range(0, width, step_x):
                value = float(z[channel, y, x] / scale)
                rows.append(
                    (
                        (x / max(1, width - 1) - 0.5) * 4.4,
                        (0.5 - y / max(1, height - 1)) * 3.3,
                        (channel - (channels - 1) / 2.0) * 0.62 + np.clip(value, -3, 3) * 0.18,
                        value,
                        float(channel),
                    )
                )
    return np.asarray(rows, dtype=np.float64)


@dataclass
class LatentFlightAssets:
    codes: list[str]
    names: dict[str, str]
    corpus_points: np.ndarray
    corpus_edges: list[tuple[int, int]]
    focus_points: dict[str, np.ndarray]
    focus_latents: dict[str, np.ndarray]
    flags: dict[str, Image.Image]
    reconstructions: dict[str, list[tuple[int, Image.Image]]]
    eigenflags: list[Image.Image]
    average_frames: list[Image.Image]
    weighted_averages: dict[str, Image.Image]
    sdvae_latents: dict[str, np.ndarray]
    sdvae_reconstructions: dict[str, Image.Image]
    sdvae_clouds: dict[str, np.ndarray]
    findings: dict[str, Any]
    weighting_vectors: dict[str, dict[str, float]]
    asset_dir: Path
    manifest: dict[str, Any]


def build_latent_flight_assets(
    config: VideoRenderConfig,
    model_manifest: dict[str, Any] | None,
) -> LatentFlightAssets:
    """Build only the representations used by the latent-flight film."""

    asset_dir = config.out_dir / "assets-latent-flight"
    flag_dir = asset_dir / "flags"
    recon_dir = asset_dir / "pca-reconstructions"
    eigen_dir = asset_dir / "eigenflags"
    average_dir = asset_dir / "weighted-averages"
    sdvae_dir = asset_dir / "sdvae-reconstructions"
    for path in (flag_dir, recon_dir, eigen_dir, average_dir, sdvae_dir):
        path.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(data_dir=DATA_DIR)
    missing = [code for code in FOCUS_CODES if code not in corpus]
    if missing:
        raise RuntimeError(f"latent-flight focus flags missing from corpus: {', '.join(missing)}")

    codes = sorted(corpus)
    arrays = corpus_arrays(corpus, LATENT_CANVAS)
    pca = PCABackend(arrays, seed=config.seed, components=PCA_COMPONENTS)
    latents = np.stack([pca.encode(code) for code in codes])
    world, pca_centre, pca_spread = _normalise_world(latents[:, :3])
    focus_latents = {code: pca.encode(code) for code in FOCUS_CODES}
    focus_points = {code: world[codes.index(code)].copy() for code in FOCUS_CODES}

    flags: dict[str, Image.Image] = {}
    reconstructions: dict[str, list[tuple[int, Image.Image]]] = {}
    reconstruction_mae: dict[str, dict[str, float]] = {}
    for code in FOCUS_CODES:
        source = rasterize_flag(corpus[code], (384, 288)).convert("RGB")
        source.save(flag_dir / f"{code}.png")
        flags[code] = source
        z = focus_latents[code]
        series: list[tuple[int, Image.Image]] = []
        errors: dict[str, float] = {}
        for count in RECONSTRUCTION_COMPONENTS:
            k = min(count, len(z))
            truncated = z.copy()
            truncated[k:] = 0.0
            reconstructed = pca.decode(truncated).image.convert("RGB")
            reconstructed.save(recon_dir / f"{code}_k{k:02d}.png")
            series.append((k, reconstructed))
            errors[str(k)] = float(
                np.abs(
                    np.asarray(arrays[code], dtype=np.float64)
                    - np.asarray(reconstructed, dtype=np.float64)
                ).mean()
            )
        reconstructions[code] = series
        reconstruction_mae[code] = errors

    eigenflags = pca.eigenflag_images(12)
    for index, image in enumerate(eigenflags, start=1):
        image.save(eigen_dir / f"eigenflag_{index:02d}.png")

    # Predecode a cycle of real barycentric PCA averages. Per-frame display uses
    # the nearest state; geometry follows the continuously evaluated weights.
    average_frames: list[Image.Image] = []
    for sample in np.linspace(0.0, 1.0, 48, endpoint=False):
        weights = parameter_weights(float(sample))
        latent = sum(weights[i] * focus_latents[code] for i, code in enumerate(FOCUS_CODES))
        average_frames.append(pca.decode(latent).image.convert("RGB"))

    sdvae_latents: dict[str, np.ndarray] = {}
    sdvae_reconstructions: dict[str, Image.Image] = {}
    sdvae_clouds: dict[str, np.ndarray] = {}
    sdvae_status = "off"
    sdvae_reason: str | None = None
    if config.foundation != "off":
        backend = build_video_backend("sdvae", arrays, config, model_manifest)
        if isinstance(backend, SDVAEBackend):
            sdvae_status = "ok"
            for code in FOCUS_CODES:
                latent = np.asarray(backend.encode(code), dtype=np.float64)
                reconstruction = backend.decode(latent).image.convert("RGB")
                reconstruction.save(sdvae_dir / f"{code}.png")
                sdvae_latents[code] = latent
                sdvae_reconstructions[code] = reconstruction
                sdvae_clouds[code] = _sdvae_cloud(latent)
        else:
            sdvae_status = "unavailable"
            sdvae_reason = getattr(backend, "reason", "Stable Diffusion VAE unavailable")
            if config.foundation == "required":
                raise RuntimeError(sdvae_reason)
        # The Diffusers model contains reference cycles. Releasing them here
        # prevents Python's cyclic collector from pausing a later video frame.
        del backend
        gc.collect()

    selected_codes = production_codes(corpus)
    selected = [corpus[code] for code in selected_codes]
    weighting_vectors: dict[str, dict[str, float]] = {}
    weighting_summaries: dict[str, dict[str, Any]] = {}
    for intent in INTENTS:
        run = weights_from_intent(intent, selected)
        weights = {code: float(value) for code, value in run.weights.items()}
        weighting_vectors[intent] = weights
        distribution = stats_for(intent, weights)
        erased = erasure_record(intent, selected)
        top = sorted(weights.items(), key=lambda item: item[1], reverse=True)[:5]
        weighting_summaries[intent] = {
            "label": INTENT_LABELS[intent],
            "normalised_entropy": float(distribution.normalised_entropy),
            "effective_contributor_count": float(distribution.effective_count),
            "maximum_share": float(distribution.top_share),
            "below_0_1_percent": int(erased.erased_count),
            "top_five": [
                {
                    "code": code,
                    "name": corpus[code].name,
                    "share": float(share),
                }
                for code, share in top
            ],
        }

    selected_stack = np.stack(
        [np.asarray(arrays[code], dtype=np.float64) for code in selected_codes],
        axis=0,
    )
    weighted_averages: dict[str, Image.Image] = {}
    for intent in INTENTS:
        vector = weighting_vectors[intent]
        ordered = np.asarray([vector[code] for code in selected_codes], dtype=np.float64)
        pixels = np.tensordot(ordered, selected_stack, axes=(0, 0))
        image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), "RGB")
        weighted_averages[intent] = image
        image.resize((384, 288), Image.Resampling.LANCZOS).save(average_dir / f"{intent}.png")

    findings_path = DATA_DIR.parent / "presets" / "latent-world-findings.json"
    archived_findings = json.loads(findings_path.read_text(encoding="utf-8"))
    findings = {
        "archive_entity_count": len(codes),
        "comparable_entity_count": len(selected_codes),
        "representations": [
            "pixel RGB",
            "dominant-colour palette",
            "structured SVG",
            "PCA-32 eigenflag coordinates",
            "CLIP-512 image embedding",
            "SD-VAE 4x24x32 posterior mean",
        ],
        "weightings": weighting_summaries,
        "pca_reconstruction_mae_0_255": reconstruction_mae,
        "focus_pca3": {
            code: [float(value) for value in focus_points[code]] for code in FOCUS_CODES
        },
        "focus_pca32_distance": {
            "ps_il": float(np.linalg.norm(focus_latents["ps"] - focus_latents["il"])),
            "us_de": float(np.linalg.norm(focus_latents["us"] - focus_latents["de"])),
        },
        "archived_clip": archived_findings,
    }

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "film": "latent-world-2026-v4-accelerating-verdict",
        "focus_codes": list(FOCUS_CODES),
        "focus_names": {code: corpus[code].name for code in FOCUS_CODES},
        "corpus_entity_count": len(codes),
        "pca": {
            "implementation": "tna.backends.pca.PCABackend / numpy.linalg.svd",
            "canvas": f"{LATENT_CANVAS[0]}x{LATENT_CANVAS[1]}",
            "components": int(pca.latent_dim),
            "world_axes": "first three full-corpus PCA coordinates, robustly scaled",
            "normalisation_centre": pca_centre.tolist(),
            "normalisation_spread": pca_spread.tolist(),
        },
        "reconstruction_components": list(RECONSTRUCTION_COMPONENTS),
        "sdvae": {
            "status": sdvae_status,
            "reason": sdvae_reason,
            "model": "stabilityai/sd-vae-ft-mse",
            "encoded_shapes": {
                code: list(latent.shape) for code, latent in sdvae_latents.items()
            },
        },
        "moving_average": {
            "codes": list(FOCUS_CODES),
            "operation": "barycentric average of real PCA coordinates",
            "weights": "deterministic artistic oscillators; parameter_weights(progress)",
        },
        "weighted_average_flags": {
            "intents": list(INTENTS),
            "operation": "normalised weighted mean of the 199 comparable 96x72 RGB rasters",
            "paths": {intent: f"weighted-averages/{intent}.png" for intent in INTENTS},
        },
        "findings": findings,
    }
    write_json(asset_dir / "asset_manifest.json", manifest)
    write_json(
        config.out_dir / "provenance" / "latent_flight_provenance.json",
        {
            "generated_at": manifest["generated_at"],
            "analytical_geometry": {
                "corpus_weather": "repeated, rotated fields of the first three PCABackend.encode coordinates",
                "focus_density_colour": "pixels sampled from source flags and PCABackend.decode reconstructions",
                "focus_density_geometry": "PCA coefficients deform non-planar particulate fields and filaments",
                "eigenflag_weather": "PCABackend.eigenflag_images sampled into cylindrical spectral mist",
                "sdvae_weather": "posterior-mean tensor values deform diffuse local particle geometry",
                "calm_front": "changing barycentric PCA mean modulates a mixed colour volume ahead of the camera",
                "verdict_flags": "five pixel-space weighted averages sampled into thick particulate apparitions",
            },
            "artistic_treatment": (
                "accelerating forward camera path, spatial repetition, turbulent advection, "
                "density, fog, route geometry, opening typography, transient process apertures, "
                "trace-bound tactical annotation, verdict staging, grain, glitch, bloom, and sonification"
            ),
        },
    )

    return LatentFlightAssets(
        codes=codes,
        names={code: corpus[code].name for code in codes},
        corpus_points=world,
        corpus_edges=_nearest_edges(world),
        focus_points=focus_points,
        focus_latents=focus_latents,
        flags=flags,
        reconstructions=reconstructions,
        eigenflags=eigenflags,
        average_frames=average_frames,
        weighted_averages=weighted_averages,
        sdvae_latents=sdvae_latents,
        sdvae_reconstructions=sdvae_reconstructions,
        sdvae_clouds=sdvae_clouds,
        findings=findings,
        weighting_vectors=weighting_vectors,
        asset_dir=asset_dir,
        manifest=manifest,
    )


@dataclass(frozen=True)
class Camera:
    position: np.ndarray
    target: np.ndarray
    roll: float
    focal: float

    @cached_property
    def _cached_basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        forward = self.target - self.position
        forward /= np.linalg.norm(forward) or 1.0
        reference_up = np.array((0.0, 1.0, 0.0), dtype=np.float64)
        right = np.cross(forward, reference_up)
        if np.linalg.norm(right) < 1e-6:
            reference_up = np.array((0.0, 0.0, 1.0), dtype=np.float64)
            right = np.cross(forward, reference_up)
        right /= np.linalg.norm(right) or 1.0
        up = np.cross(right, forward)
        cosine, sine = math.cos(self.roll), math.sin(self.roll)
        rolled_right = right * cosine + up * sine
        rolled_up = -right * sine + up * cosine
        return rolled_right, rolled_up, forward

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._cached_basis


def project_point(
    point: np.ndarray,
    camera: Camera,
    size: tuple[int, int],
    *,
    near: float = 0.16,
) -> tuple[float, float, float] | None:
    right, up, forward = camera.basis()
    relative = np.asarray(point, dtype=np.float64) - camera.position
    depth = float(relative @ forward)
    if depth <= near:
        return None
    x = float(relative @ right)
    y = float(relative @ up)
    width, height = size
    scale = camera.focal / depth
    return width * 0.5 + x * scale, height * 0.5 - y * scale, depth


def _ease(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def _lerp(a: np.ndarray, b: np.ndarray, value: float) -> np.ndarray:
    return a + (b - a) * value


def forward_depth(progress: float) -> float:
    """Monotonic depth of the searching camera through the latent world."""

    value = min(1.0, max(0.0, float(progress)))
    return WORLD_START + (WORLD_END - WORLD_START) * value


class PlanarLatentFlightRenderer:
    """Preserved renderer for the first latent-flight study."""

    def __init__(self, config: VideoRenderConfig, assets: LatentFlightAssets):
        self.config = config
        self.assets = assets
        self.width = config.width
        self.height = config.height
        self.segments = flight_segments(config.duration)
        self.vmask = effects.vignette_mask((self.width, self.height), strength=0.62)
        self.focus = np.stack([assets.focus_points[code] for code in FOCUS_CODES])
        self.centre = self.focus.mean(axis=0)
        rng = np.random.default_rng(config.seed)
        self.point_phases = rng.uniform(0, math.tau, len(assets.corpus_points))
        self.starfield = rng.uniform((-28, -18, -28), (28, 18, 28), size=(420, 3))
        self.shards = []
        for index in range(110):
            code = FOCUS_CODES[index % len(FOCUS_CODES)]
            anchor = assets.focus_points[code]
            centre = anchor + rng.normal(0.0, (4.8, 3.6, 4.8), 3)
            scale = rng.uniform(0.25, 1.65)
            vertices = centre + rng.normal(0.0, scale, (3, 3))
            self.shards.append((vertices, FOCUS_COLOURS[code], rng.uniform(0, math.tau)))
        self._build_shots()

    def _build_shots(self) -> None:
        ps, il, us, de = (self.assets.focus_points[code] for code in FOCUS_CODES)
        mid_pair = (ps + il) * 0.5
        c = self.centre
        self.shot_poses = {
            "breach": (c + (-2, 10, -24), c + (4, 2, -8), c, ps, -0.10, 0.14),
            "corpus_dive": (c + (13, 5, -17), ps + (-5, 2, -8), ps, ps, 0.06, -0.16),
            "palestine_field": (ps + (-7, 3, -8), ps + (4, -2, 5), ps, mid_pair, -0.14, 0.18),
            "collision": (mid_pair + (1, 1, -5), mid_pair + (-2, 0, 4), ps, il, 0.30, -0.28),
            "israel_field": (il + (7, -2, -8), il + (-5, 4, 6), il, il, -0.18, 0.12),
            "axis_snap": (c + (-2, 15, 1), c + (1, -10, 0), c, us, 0.48, -0.42),
            "american_scale": (us + (-10, 1, -11), us + (7, 5, 7), us, de, -0.08, 0.25),
            "german_basis": (de + (8, 6, -8), de + (-8, -3, 7), de, c, 0.16, -0.22),
            "sdvae_interior": (c + (-6, 1, -7), mid_pair + (5, 0, 5), mid_pair, c, -0.30, 0.36),
            "false_calm": (c + (8, 5, -13), c + (-5, 1, 7), c, c, 0.05, -0.05),
            "parameter_break": (c + (0, 2, -4), c + (0, -1, 4), c, ps, 0.42, -0.46),
            "pursuit": (c + (-7, 2, -10), c + (8, -4, 5), c, c, -0.16, 0.27),
            "almost_rest": (c + (0.4, 0.2, -3.8), c + (-0.5, 0.1, -3.2), c, c, 0.02, -0.02),
            "escape": (c + (1, 0, -3.4), c + (17, 9, -18), c, c + (8, 2, 4), -0.22, 0.55),
        }

    def _phase(self, time: float) -> tuple[int, str, float]:
        for index, (key, start, end) in enumerate(self.segments):
            if time < end or index == len(self.segments) - 1:
                return index, key, (time - start) / max(1e-9, end - start)
        return len(self.segments) - 1, self.segments[-1][0], 1.0

    def _chapter(self, time: float) -> tuple[int, str, float]:
        for index, (key, start, end) in enumerate(self.chapters):
            if time < end or index == len(self.chapters) - 1:
                return index, key, (time - start) / max(1e-9, end - start)
        return len(self.chapters) - 1, self.chapters[-1][0], 1.0

    def calm_point(self, progress: float) -> np.ndarray:
        weights = parameter_weights(progress)
        return sum(weights[i] * self.focus[i] for i in range(len(FOCUS_CODES)))

    def _camera(self, time: float) -> tuple[Camera, str, float]:
        _, key, local = self._phase(time)
        a, b, ta, tb, roll_a, roll_b = self.shot_poses[key]
        q = _ease(local)
        position = _lerp(np.asarray(a, dtype=float), np.asarray(b, dtype=float), q)
        target = _lerp(np.asarray(ta, dtype=float), np.asarray(tb, dtype=float), q)
        global_progress = time / max(self.config.duration, 1e-9)
        calm = self.calm_point(global_progress)
        if key in {"false_calm", "parameter_break", "pursuit", "almost_rest", "escape"}:
            target = _lerp(target, calm, 0.72 if key != "escape" else 0.38)
        frantic = 0.10 if key == "almost_rest" else 0.34
        position = position + np.array(
            (
                math.sin(time * 6.7) * frantic,
                math.sin(time * 9.1 + 1.2) * frantic * 0.55,
                math.sin(time * 4.3 + 2.1) * frantic * 0.38,
            )
        )
        # A second edit rhythm punches the camera to nearby positions every few
        # seconds. The false calm is the only passage spared this discontinuity.
        if key not in {"false_calm", "almost_rest"}:
            jump_index = int(time / 2.35)
            jump_rng = np.random.default_rng(self.config.seed + 7001 + jump_index)
            position += jump_rng.uniform((-1.4, -0.75, -1.4), (1.4, 0.75, 1.4))
            roll_a += float(jump_rng.uniform(-0.10, 0.10))
            roll_b += float(jump_rng.uniform(-0.10, 0.10))
        focal = self.width * (0.88 + 0.22 * math.sin(time * 0.31 + local * math.pi))
        return Camera(position, target, _lerp(np.array(roll_a), np.array(roll_b), q).item(), focal), key, local

    def _line3d(self, draw: ImageDraw.ImageDraw, a, b, camera, colour, width=1) -> None:
        pa = project_point(np.asarray(a), camera, (self.width, self.height))
        pb = project_point(np.asarray(b), camera, (self.width, self.height))
        if pa is None or pb is None:
            return
        if max(pa[0], pb[0]) < -100 or min(pa[0], pb[0]) > self.width + 100:
            return
        if max(pa[1], pb[1]) < -100 or min(pa[1], pb[1]) > self.height + 100:
            return
        draw.line((pa[0], pa[1], pb[0], pb[1]), fill=colour, width=width)

    def _background(self, camera: Camera, time: float) -> Image.Image:
        yy = np.linspace(0, 1, self.height, dtype=np.float32)[:, None]
        top = np.array((1, 1, 10), dtype=np.float32)
        bottom = np.array((6, 0, 18), dtype=np.float32)
        gradient = top[None, None, :] * (1 - yy[:, :, None]) + bottom[None, None, :] * yy[:, :, None]
        gradient = np.repeat(gradient, self.width, axis=1)
        image = Image.fromarray(np.clip(gradient, 0, 255).astype(np.uint8), "RGB")
        draw = ImageDraw.Draw(image)

        # Sparse stars are ordinary 3-D points and reinforce camera translation.
        for point in self.starfield:
            projected = project_point(point, camera, (self.width, self.height))
            if projected is None:
                continue
            x, y, depth = projected
            if 0 <= x < self.width and 0 <= y < self.height:
                level = max(18, min(145, int(190 - depth * 4)))
                draw.point((x, y), fill=(level // 3, level, level))

        # A volumetric-looking PCA lattice: floor, ceiling, and vertical slices.
        grid_colour = (19, 80, 104)
        for value in np.arange(-12, 12.1, 2.0):
            self._line3d(draw, (-12, -7, value), (12, -7, value), camera, grid_colour)
            self._line3d(draw, (value, -7, -12), (value, -7, 12), camera, grid_colour)
            self._line3d(draw, (-12, 7, value), (12, 7, value), camera, (13, 43, 71))
        for value in np.arange(-8, 8.1, 2.0):
            self._line3d(draw, (-12, value, 10), (12, value, 10), camera, (23, 44, 72))
            self._line3d(draw, (-12, value, -10), (12, value, -10), camera, (36, 24, 70))
        return image

    def _deformed_points(self, time: float) -> np.ndarray:
        points = self.assets.corpus_points.copy()
        amplitude = 0.18 + 0.34 * (0.5 + 0.5 * math.sin(time * 0.79))
        points[:, 1] += np.sin(self.point_phases + time * 1.7) * amplitude
        points[:, 0] += np.sin(self.point_phases * 1.7 - time * 0.9) * amplitude * 0.42
        return points

    def _draw_corpus(self, image: Image.Image, camera: Camera, time: float) -> None:
        points = self._deformed_points(time)
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        projected = [project_point(point, camera, image.size) for point in points]
        for a, b in self.assets.corpus_edges:
            pa, pb = projected[a], projected[b]
            if pa is None or pb is None:
                continue
            alpha = max(8, min(62, int(90 - (pa[2] + pb[2]) * 2.2)))
            draw.line((pa[0], pa[1], pb[0], pb[1]), fill=(38, 129, 174, alpha), width=1)
        focus_set = set(FOCUS_CODES)
        visible = sorted(
            ((i, p) for i, p in enumerate(projected) if p is not None),
            key=lambda item: item[1][2],
            reverse=True,
        )
        for index, (x, y, depth) in visible:
            if not (-12 <= x <= self.width + 12 and -12 <= y <= self.height + 12):
                continue
            code = self.assets.codes[index]
            colour = FOCUS_COLOURS.get(code, (64, 212, 238))
            radius = 5 if code in focus_set else max(1, min(3, int(28 / depth)))
            alpha = 245 if code in focus_set else max(30, min(150, int(210 - depth * 4)))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*colour, alpha))
            if code in focus_set:
                draw.ellipse((x - 13, y - 13, x + 13, y + 13), outline=(*colour, 150), width=1)
        # Short temporal vectors turn the corpus into streaks during flight.
        previous = self._deformed_points(time - 0.16)
        for index in range(0, len(points), 3):
            now = projected[index]
            before = project_point(previous[index], camera, image.size)
            if now is None or before is None:
                continue
            if 0 <= now[0] < self.width and 0 <= now[1] < self.height:
                draw.line((before[0], before[1], now[0], now[1]), fill=(80, 210, 250, 64), width=1)
        image.paste(overlay, (0, 0), overlay)

    def _draw_shards(self, image: Image.Image, camera: Camera, time: float) -> None:
        projected_shards = []
        for vertices, colour, phase in self.shards:
            offset = np.array(
                (
                    math.sin(time * 0.31 + phase) * 0.42,
                    math.sin(time * 0.53 + phase) * 0.28,
                    math.cos(time * 0.27 + phase) * 0.42,
                )
            )
            projected = [project_point(vertex + offset, camera, image.size) for vertex in vertices]
            if any(point is None for point in projected):
                continue
            depth = sum(point[2] for point in projected if point is not None) / 3
            projected_shards.append((depth, projected, colour))
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for depth, projected, colour in sorted(projected_shards, reverse=True):
            polygon = [(point[0], point[1]) for point in projected if point is not None]
            if not any(-80 < x < self.width + 80 and -80 < y < self.height + 80 for x, y in polygon):
                continue
            alpha = max(8, min(72, int(105 - depth * 3)))
            draw.polygon(polygon, fill=(*colour, alpha // 3), outline=(*colour, alpha))
        image.paste(overlay, (0, 0), overlay)

    def _active_focus_codes(self, key: str, time: float) -> tuple[str, ...]:
        if key == "palestine_field":
            return ("ps",)
        if key == "israel_field":
            return ("il",)
        if key == "american_scale":
            return ("us",)
        if key == "german_basis":
            return ("de",)
        if key == "collision":
            return ("ps", "il")
        if key == "sdvae_interior":
            return (FOCUS_CODES[int(time / 2.7) % len(FOCUS_CODES)],)
        return ()

    def _draw_focus_meshes(self, image: Image.Image, camera: Camera, time: float, key: str) -> None:
        """Deform source flags into real-coordinate-driven 3-D surfaces."""

        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        columns, rows = 14, 10
        for code in self._active_focus_codes(key, time):
            anchor = self.assets.focus_points[code]
            texture = self.assets.flags[code]
            latent = self.assets.focus_latents[code]
            latent_scale = float(np.std(latent)) or 1.0
            code_index = FOCUS_CODES.index(code)
            yaw = time * (0.16 + code_index * 0.025) + code_index * 1.3
            horizontal = np.array((math.cos(yaw), 0.0, math.sin(yaw)))
            vertical = np.array((0.0, 1.0, 0.0))
            normal = np.cross(horizontal, vertical)
            vertices: list[list[tuple[float, float, float] | None]] = []
            for row in range(rows + 1):
                vertex_row = []
                for column in range(columns + 1):
                    ux = column / columns
                    uy = row / rows
                    coefficient = float(latent[(row * (columns + 1) + column) % len(latent)] / latent_scale)
                    displacement = (
                        np.clip(coefficient, -2.5, 2.5) * 0.23
                        + math.sin(ux * 9.0 + uy * 4.0 + time * 2.4 + code_index) * 0.24
                    )
                    world = (
                        anchor
                        + horizontal * ((ux - 0.5) * 5.4)
                        + vertical * ((0.5 - uy) * 3.85)
                        + normal * displacement
                    )
                    vertex_row.append(project_point(world, camera, image.size))
                vertices.append(vertex_row)

            pixels = np.asarray(texture.convert("RGB"))
            for row in range(rows):
                for column in range(columns):
                    points = (
                        vertices[row][column],
                        vertices[row][column + 1],
                        vertices[row + 1][column + 1],
                        vertices[row + 1][column],
                    )
                    if any(point is None for point in points):
                        continue
                    # An animated dropout pattern makes the surface structurally
                    # incomplete rather than merely applying a post-process glitch.
                    dropout = math.sin(column * 5.3 + row * 8.1 + time * 4.7 + code_index)
                    if dropout > 0.72:
                        continue
                    polygon = [(point[0], point[1]) for point in points if point is not None]
                    sample_x = min(texture.width - 1, int((column + 0.5) / columns * texture.width))
                    sample_y = min(texture.height - 1, int((row + 0.5) / rows * texture.height))
                    colour = tuple(int(value) for value in pixels[sample_y, sample_x])
                    depth = sum(point[2] for point in points if point is not None) / 4
                    alpha = max(42, min(178, int(205 - depth * 4)))
                    edge = FOCUS_COLOURS[code]
                    draw.polygon(polygon, fill=(*colour, alpha), outline=(*edge, min(210, alpha + 24)))
        image.paste(overlay, (0, 0), overlay)

    def _plane_quad(self, position, width, height, yaw, pitch, camera):
        horizontal = np.array((math.cos(yaw), 0.0, math.sin(yaw)))
        vertical = np.array(
            (-math.sin(yaw) * math.sin(pitch), math.cos(pitch), math.cos(yaw) * math.sin(pitch))
        )
        corners = (
            position - horizontal * width / 2 + vertical * height / 2,
            position + horizontal * width / 2 + vertical * height / 2,
            position + horizontal * width / 2 - vertical * height / 2,
            position - horizontal * width / 2 - vertical * height / 2,
        )
        projected = [project_point(point, camera, (self.width, self.height)) for point in corners]
        if any(point is None for point in projected):
            return None
        return [(point[0], point[1]) for point in projected if point is not None]

    @staticmethod
    def _perspective_coefficients(destination, source) -> tuple[float, ...]:
        matrix = []
        vector = []
        for (x, y), (u, v) in zip(destination, source):
            matrix.append((x, y, 1, 0, 0, 0, -u * x, -u * y))
            vector.append(u)
            matrix.append((0, 0, 0, x, y, 1, -v * x, -v * y))
            vector.append(v)
        return tuple(np.linalg.lstsq(np.asarray(matrix), np.asarray(vector), rcond=None)[0])

    def _paste_plane(
        self,
        image,
        texture,
        quad,
        alpha=220,
        outline=(67, 237, 255, 180),
        fragmentation: float = 0.0,
        phase: float = 0.0,
    ) -> None:
        xs, ys = [p[0] for p in quad], [p[1] for p in quad]
        x0, y0 = max(0, int(min(xs))), max(0, int(min(ys)))
        x1, y1 = min(self.width, int(max(xs)) + 1), min(self.height, int(max(ys)) + 1)
        if x1 - x0 < 3 or y1 - y0 < 3 or x1 - x0 > self.width * 1.8 or y1 - y0 > self.height * 1.8:
            return
        local = [(x - x0, y - y0) for x, y in quad]
        signed_area = sum(
            local[i][0] * local[(i + 1) % 4][1] - local[(i + 1) % 4][0] * local[i][1]
            for i in range(4)
        )
        area = abs(signed_area) * 0.5
        box_area = (x1 - x0) * (y1 - y0)
        # Nearly edge-on planes produce an ill-conditioned homography and do not
        # contribute a readable surface. Keep their wire geometry elsewhere.
        if area < 9.0 or area / max(1.0, box_area) < 0.055:
            return
        tw, th = texture.size
        source = ((0, 0), (tw - 1, 0), (tw - 1, th - 1), (0, th - 1))
        coefficients = self._perspective_coefficients(local, source)
        warped = texture.convert("RGB").transform(
            (x1 - x0, y1 - y0),
            Image.Transform.PERSPECTIVE,
            coefficients,
            resample=Image.Resampling.BILINEAR,
        ).convert("RGBA")
        mask = Image.new("L", warped.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.polygon(local, fill=alpha)
        if fragmentation > 0.0:
            # Cut geometric voids into the image plane. This preserves the real
            # source texture while refusing the clean, report-like rectangle.
            width, height = warped.size
            cuts = max(1, int(2 + fragmentation * 7))
            for index in range(cuts):
                q = (math.sin(phase * 13.1 + index * 7.7) + 1.0) * 0.5
                y = int(q * height)
                band = max(1, int(height * (0.012 + fragmentation * 0.035)))
                mask_draw.rectangle((0, y, width, min(height, y + band)), fill=0)
            bite = int(min(width, height) * fragmentation * 0.38)
            if bite > 1:
                mask_draw.polygon(((0, 0), (bite, 0), (0, bite * 2)), fill=0)
                mask_draw.polygon(((width, height), (width - bite * 2, height), (width, height - bite)), fill=0)
        warped.putalpha(mask)
        image.paste(warped, (x0, y0), warped)
        draw = ImageDraw.Draw(image, "RGBA")
        if fragmentation < 0.72:
            draw.line((*quad, quad[0]), fill=outline, width=max(1, self.width // 900))

    def _artifact_candidates(self, time: float):
        candidates = []
        for focus_index, code in enumerate(FOCUS_CODES):
            anchor = self.assets.focus_points[code]
            colour = FOCUS_COLOURS[code]
            series = self.assets.reconstructions[code]
            for index, (_, texture) in enumerate(series):
                angle = index / len(series) * math.tau + time * (0.12 + focus_index * 0.018)
                radius = 3.1 + index * 0.38
                position = anchor + np.array(
                    (math.cos(angle) * radius, (index - 2.5) * 0.62, math.sin(angle) * radius)
                )
                candidates.append((position, texture, 2.45, 1.84, angle + 0.9, math.sin(angle) * 0.16, colour, 0.68, index + focus_index * 9))
            if code in self.assets.sdvae_reconstructions:
                angle = -time * 0.22 + focus_index * math.pi / 2
                position = anchor + np.array((math.cos(angle) * 4.6, 2.7, math.sin(angle) * 4.6))
                candidates.append(
                    (position, self.assets.sdvae_reconstructions[code], 3.2, 2.4, angle + 1.2, -0.14, (255, 84, 214), 0.46, focus_index + 31)
                )

        calm = self.calm_point(time / max(self.config.duration, 1e-9))
        for index, texture in enumerate(self.assets.eigenflags):
            angle = index / len(self.assets.eigenflags) * math.tau - time * 0.09
            radius = 5.8 + (index % 3) * 1.4
            position = calm + np.array((math.cos(angle) * radius, ((index % 4) - 1.5) * 1.7, math.sin(angle) * radius))
            candidates.append((position, texture, 2.1, 2.8, angle + 0.65, 0.24, (118, 88, 255), 0.84, index + 53))
        return candidates

    def _draw_artifacts(self, image: Image.Image, camera: Camera, time: float) -> None:
        candidates = []
        for values in self._artifact_candidates(time):
            centre = project_point(values[0], camera, (self.width, self.height))
            if centre is None:
                continue
            x, y, depth = centre
            if -self.width * 0.4 < x < self.width * 1.4 and -self.height * 0.4 < y < self.height * 1.4:
                candidates.append((depth, values))
        # Near objects matter most; limiting planes keeps 1080p production viable.
        chosen = sorted(candidates, key=lambda item: item[0])[:12]
        for depth, (position, texture, width, height, yaw, pitch, colour, fragmentation, phase) in sorted(chosen, reverse=True):
            quad = self._plane_quad(position, width, height, yaw, pitch, camera)
            if quad is None:
                continue
            alpha = max(70, min(235, int(270 - depth * 5)))
            self._paste_plane(
                image,
                texture,
                quad,
                alpha=alpha,
                outline=(*colour, 180),
                fragmentation=fragmentation,
                phase=time * 0.2 + phase,
            )

    def _draw_sdvae_clouds(self, image: Image.Image, camera: Camera, time: float) -> None:
        if not self.assets.sdvae_clouds:
            return
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        channel_colours = ((255, 45, 171), (44, 225, 255), (133, 88, 255), (255, 199, 49))
        for code, cloud in self.assets.sdvae_clouds.items():
            anchor = self.assets.focus_points[code]
            spin = time * (0.13 + FOCUS_CODES.index(code) * 0.017)
            cosine, sine = math.cos(spin), math.sin(spin)
            for row in cloud:
                local = row[:3]
                rotated = np.array((local[0] * cosine - local[2] * sine, local[1], local[0] * sine + local[2] * cosine))
                projected = project_point(anchor + rotated, camera, image.size)
                if projected is None:
                    continue
                x, y, depth = projected
                if not (0 <= x < self.width and 0 <= y < self.height):
                    continue
                strength = min(1.0, abs(float(row[3])) / 2.5)
                colour = channel_colours[int(row[4]) % len(channel_colours)]
                radius = 1 if depth > 10 else 2
                alpha = int(32 + strength * 150)
                draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=(*colour, alpha))
        image.paste(overlay, (0, 0), overlay)

    def _draw_calm_attractor(self, image: Image.Image, camera: Camera, time: float) -> None:
        progress = time / max(self.config.duration, 1e-9)
        calm = self.calm_point(progress)
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        radius = 1.25 + 0.18 * math.sin(time * 2.1)
        for axis in range(3):
            ring = []
            for step in range(49):
                angle = step / 48 * math.tau + time * (0.34 + axis * 0.09)
                point = np.zeros(3)
                a, b = ((0, 1), (1, 2), (2, 0))[axis]
                point[a] = math.cos(angle) * radius
                point[b] = math.sin(angle) * radius
                projected = project_point(calm + point, camera, image.size)
                if projected is not None:
                    ring.append((projected[0], projected[1]))
            if len(ring) > 1:
                draw.line(ring, fill=(210, 250, 255, 180 - axis * 30), width=max(1, self.width // 800))
        projected = project_point(calm, camera, image.size)
        if projected is not None:
            x, y, depth = projected
            r = max(2, min(22, int(camera.focal * 0.08 / depth)))
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(245, 255, 255, 220))
        image.paste(overlay, (0, 0), overlay)

        _, key, _ = self._phase(time)
        intermittently_visible = math.sin(time * 0.83) > 0.18
        if key in {"false_calm", "parameter_break", "pursuit", "almost_rest", "escape"} or intermittently_visible:
            frame_index = int(progress * len(self.assets.average_frames)) % len(self.assets.average_frames)
            texture = self.assets.average_frames[frame_index]
            yaw = time * 0.41
            quad = self._plane_quad(calm, 2.30, 1.72, yaw, math.sin(time * 0.23) * 0.12, camera)
            if quad is not None:
                self._paste_plane(
                    image,
                    texture,
                    quad,
                    alpha=185,
                    outline=(235, 255, 255, 225),
                    fragmentation=0.48,
                    phase=time,
                )

    def _cut_glitch(self, time: float, key: str, local: float) -> tuple[float, str]:
        edge = min(local, 1.0 - local)
        cut = max(0.0, 1.0 - edge / 0.035)
        micro_phase = (time % 2.35) / 2.35
        micro_cut = max(0.0, 1.0 - min(micro_phase, 1.0 - micro_phase) / 0.045)
        restless = 0.08 + 0.10 * (0.5 + 0.5 * math.sin(time * 3.7))
        if key in {"collision", "axis_snap", "parameter_break", "escape"}:
            restless += 0.15
        amount = min(0.96, restless + cut * 0.72 + micro_cut * 0.60)
        modes = ("block", "channel", "latent", "pixel_sort", "patch_shuffle")
        mode = modes[(int(time * 1.7) + list(dict(FLIGHT_SHOTS)).index(key)) % len(modes)]
        return amount, mode

    def render(self, frame_index: int) -> Image.Image:
        time = frame_index / self.config.fps
        camera, key, local = self._camera(time)
        image = self._background(camera, time)
        self._draw_corpus(image, camera, time)
        self._draw_shards(image, camera, time)
        self._draw_focus_meshes(image, camera, time, key)
        self._draw_sdvae_clouds(image, camera, time)
        self._draw_artifacts(image, camera, time)
        self._draw_calm_attractor(image, camera, time)

        glitch, mode = self._cut_glitch(time, key, local)
        image = effects.treat(
            image,
            vmask=self.vmask,
            glitch=glitch,
            glitch_mode=mode,
            seed=self.config.seed + 991,
            frame=frame_index,
            grain_amount=5.5,
            aberration=2,
        )
        # A very brief initial breach and terminal cut; no explanatory titles.
        fade_in = min(1.0, time / 0.45)
        fade_out = min(1.0, max(0.0, (self.config.duration - time) / 0.18))
        fade = min(fade_in, fade_out)
        if fade < 1.0:
            image = Image.blend(Image.new("RGB", image.size, (0, 0, 0)), image, fade)
        return image


class LatentFlightRenderer:
    """A forward-moving renderer in which representations become weather.

    Nothing in this scene is pasted as a flag or reconstruction plane. Source
    pixels supply colour, PCA coefficients supply deformation, eigenflags supply
    spectral mist, and SD-VAE values supply turbulent local geometry. The visual
    result is deliberately metaphorical: a large information climate rather than
    a set of objects being presented to a viewer.
    """

    _FIELD_SPECS = (
        ("ps", -8.0, -4.0, 1.0, 1.10),
        ("ps", 18.0, 3.0, -2.0, 0.92),
        ("il", 36.0, 4.0, 1.0, 1.04),
        ("ps", 57.0, -3.5, 1.0, 1.22),
        ("il", 63.0, 3.2, -1.5, 1.22),
        ("us", 88.0, -4.0, 2.0, 1.10),
        ("de", 112.0, 3.0, -1.0, 1.02),
        ("us", 132.0, 4.0, -2.0, 0.94),
        ("de", 150.0, -3.0, 2.0, 1.16),
        ("ps", 168.0, -4.5, 0.0, 1.34),
        ("il", 174.0, 4.0, 1.0, 1.34),
        ("us", 193.0, -3.5, -2.0, 1.34),
        ("de", 199.0, 3.4, 2.0, 1.34),
        ("ps", 222.0, -4.0, 1.2, 1.58),
        ("il", 224.0, 3.8, -1.2, 1.58),
        ("us", 227.0, -1.8, -2.6, 1.58),
        ("de", 230.0, 1.8, 2.6, 1.58),
        ("ps", 270.0, -6.0, 3.0, 1.72),
        ("il", 273.0, 5.0, -2.0, 1.72),
        ("us", 278.0, -4.0, -3.0, 1.72),
        ("de", 282.0, 4.0, 3.0, 1.72),
    )
    _NAV_WAYPOINTS = (
        (WORLD_START, 0.0, 0.0),
        (-8.0, -4.0, 1.0),
        (18.0, 3.0, -2.0),
        (36.0, 4.0, 1.0),
        (60.0, 0.0, 0.0),
        (88.0, -4.0, 2.0),
        (112.0, 3.0, -1.0),
        (132.0, 4.0, -2.0),
        (150.0, -3.0, 2.0),
        (171.0, 0.0, 0.0),
        (196.0, 0.0, 0.0),
        (226.0, 0.0, 0.0),
        (WORLD_END, 0.0, 0.0),
        (WORLD_FAR, 0.0, 0.0),
    )

    def __init__(self, config: VideoRenderConfig, assets: LatentFlightAssets):
        self.config = config
        self.assets = assets
        self.width = config.width
        self.height = config.height
        self.film_phases = film_phase_segments(config.duration)
        self.search_start = self.film_phases[2][1]
        self.search_end = self.film_phases[2][2]
        self.search_duration = self.search_end - self.search_start
        self.segments = [
            (key, start + self.search_start, end + self.search_start)
            for key, start, end in flight_segments(self.search_duration)
        ]
        self.chapters = [
            (key, start + self.search_start, end + self.search_start)
            for key, start, end in chapter_segments(self.search_duration)
        ]
        self.chapter_depths = {
            key: forward_depth(
                accelerated_search_progress(
                    (start - self.search_start) / max(self.search_duration, 1e-9)
                )
            )
            for key, start, _ in self.chapters
        }
        self.vmask = effects.vignette_mask((self.width, self.height), strength=0.54)
        self.focus = np.stack([assets.focus_points[code] for code in FOCUS_CODES])
        self.centre = self.focus.mean(axis=0)
        self.rng = np.random.default_rng(config.seed + 4401)
        self._palette_cache: dict[str, np.ndarray] = {}
        self.hud_title = load_font(max(24, self.width // 29), mono=True)
        self.hud_heading = load_font(max(16, self.width // 54), mono=True)
        self.hud_label = load_font(max(11, self.width // 96), mono=True)
        self.hud_tiny = load_font(max(9, self.width // 142), mono=True)
        self.open_title = load_font(max(54, self.width // 10))
        self.open_kicker = load_font(max(18, self.width // 42), mono=True)
        self.open_credit = load_font(max(20, self.width // 54), mono=True)
        self.statement_title = load_font(max(28, self.width // 28))
        self.statement_body = load_font(max(16, self.width // 54), mono=True)
        self.verdict_title = load_font(max(34, self.width // 24))
        self._font_cache: dict[tuple[int, bool], Any] = {}
        self.field_anchors: dict[str, list[np.ndarray]] = {code: [] for code in FOCUS_CODES}

        self.starfield = np.column_stack(
            (
                self.rng.uniform(-34.0, 34.0, 1900),
                self.rng.uniform(-20.0, 20.0, 1900),
                self.rng.uniform(WORLD_START - 8.0, WORLD_FAR + 35.0, 1900),
            )
        )
        self.star_strength = self.rng.uniform(0.25, 1.0, len(self.starfield))
        self._volume_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self.filaments: list[tuple[np.ndarray, tuple[int, int, int], float]] = []
        self._build_density_world()
        self.volume_base = np.concatenate([part[0] for part in self._volume_parts], axis=0)
        self.volume_colours = np.concatenate([part[1] for part in self._volume_parts], axis=0)
        self.volume_phases = np.concatenate([part[2] for part in self._volume_parts], axis=0)
        self.volume_sizes = np.concatenate([part[3] for part in self._volume_parts], axis=0)
        del self._volume_parts
        self._build_corpus_weather()
        self._build_calm_front()
        self._build_average_verdicts()

    @staticmethod
    def _lift_colours(colours: np.ndarray) -> np.ndarray:
        values = np.asarray(colours, dtype=np.float64)
        grey = values.mean(axis=1, keepdims=True)
        values = grey + (values - grey) * 1.10
        dark = grey[:, 0] < 18.0
        values[dark] = values[dark] * 0.45 + np.array((24.0, 27.0, 33.0))
        bright = grey[:, 0] > 242.0
        values[bright] = values[bright] * 0.82 + np.array((22.0, 24.0, 28.0))
        return np.clip(values, 0, 255).astype(np.uint8)

    def _sample_image(self, image: Image.Image, count: int) -> np.ndarray:
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).reshape(-1, 3)
        return pixels[self.rng.integers(0, len(pixels), count)]

    def _flag_palette(self, code: str) -> np.ndarray:
        cached = self._palette_cache.get(code)
        if cached is not None:
            return cached
        reduced = self.assets.flags[code].quantize(colors=6, method=Image.Quantize.MEDIANCUT)
        entries = sorted(reduced.getcolors() or (), reverse=True)
        raw_palette = reduced.getpalette() or []
        colours = []
        for _, palette_index in entries:
            offset = int(palette_index) * 3
            colour = np.asarray(raw_palette[offset : offset + 3], dtype=np.uint8)
            if len(colour) != 3:
                continue
            if not any(np.linalg.norm(colour.astype(float) - old.astype(float)) < 22.0 for old in colours):
                colours.append(colour)
        if not colours:
            colours = [np.asarray(FOCUS_COLOURS[code], dtype=np.uint8)]
        self._palette_cache[code] = self._lift_colours(np.stack(colours))
        return self._palette_cache[code]

    def _palette(self, code: str, count: int) -> np.ndarray:
        sources = [self.assets.flags[code]] + [image for _, image in self.assets.reconstructions[code]]
        choices = self.rng.choice(len(sources), count, p=(0.66, 0.08, 0.07, 0.06, 0.05, 0.04, 0.04))
        colours = np.empty((count, 3), dtype=np.uint8)
        flag_mask = choices == 0
        if np.any(flag_mask):
            palette = self._flag_palette(code)
            colours[flag_mask] = palette[self.rng.integers(0, len(palette), int(flag_mask.sum()))]
        for source_index, source in enumerate(sources[1:], start=1):
            mask = choices == source_index
            if np.any(mask):
                colours[mask] = self._sample_image(source, int(mask.sum()))
        return self._lift_colours(colours)

    def _append_particles(
        self,
        points: np.ndarray,
        colours: np.ndarray,
        phases: np.ndarray,
        sizes: np.ndarray,
    ) -> None:
        self._volume_parts.append(
            (
                np.asarray(points, dtype=np.float64),
                np.asarray(colours, dtype=np.uint8),
                np.asarray(phases, dtype=np.float64),
                np.asarray(sizes, dtype=np.float64),
            )
        )

    def _append_focus_field(
        self,
        code: str,
        anchor: np.ndarray,
        spread: float,
        count: int,
    ) -> None:
        latent = np.asarray(self.assets.focus_latents[code], dtype=np.float64)
        latent_scale = float(np.std(latent)) or 1.0
        coefficients = latent[self.rng.integers(0, len(latent), count)] / latent_scale
        theta = self.rng.uniform(0.0, math.tau, count) + coefficients * 0.24
        turbulence = np.clip(coefficients, -3.0, 3.0)
        body = self.rng.normal(0.0, 1.0, (count, 3))
        points = np.column_stack(
            (
                anchor[0] + body[:, 0] * 6.8 * spread + np.cos(theta) * (1.2 + np.abs(turbulence) * 0.48),
                anchor[1] + body[:, 1] * 4.7 * spread
                + np.sin(theta * 1.17 + turbulence * 0.16) * 1.4,
                anchor[2] + body[:, 2] * 9.8 * spread
                + np.sin(theta * 2.1) * (1.8 + np.abs(turbulence) * 0.52),
            )
        )
        phases = self.rng.uniform(0.0, math.tau, count)
        sizes = self.rng.lognormal(mean=-0.12, sigma=0.58, size=count) * spread
        self._append_particles(points, self._palette(code, count), phases, sizes)

        # Coefficient-driven streamlines give the cloud interior continuity while
        # remaining porous and non-rectangular.
        for strand in range(2):
            steps = 15
            u = np.linspace(-1.0, 1.0, steps)
            coefficient = float(latent[(strand * 5 + int(anchor[2])) % len(latent)] / latent_scale)
            angle = self.rng.uniform(0.0, math.tau)
            radial = self.rng.uniform(2.0, 7.2) * spread
            points_line = np.column_stack(
                (
                    anchor[0] + np.cos(angle + u * (2.2 + coefficient * 0.12)) * radial * (0.8 + 0.2 * np.cos(u * math.pi)),
                    anchor[1] + np.sin(angle * 1.3 + u * 2.7) * radial * 0.62,
                    anchor[2] + u * 8.2 * spread + np.sin(u * 5.0 + angle) * 1.4,
                )
            )
            colour = tuple(int(value) for value in self._palette(code, 1)[0])
            self.filaments.append((points_line, colour, self.rng.uniform(0.0, math.tau)))

    def _append_eigen_weather(self) -> None:
        for index, image in enumerate(self.assets.eigenflags):
            count = 125
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
            height, width = pixels.shape[:2]
            flat_indices = self.rng.integers(0, height * width, count)
            px = flat_indices % width
            py = flat_indices // width
            u = px / max(1, width - 1)
            v = py / max(1, height - 1)
            angle = u * math.tau * 2.0 + index * 0.63
            radial = 2.0 + v * 5.5 + self.rng.normal(0.0, 0.65, count)
            anchor_z = 20.0 + index * 21.0
            points = np.column_stack(
                (
                    np.cos(angle) * radial + math.sin(index * 1.7) * 2.8,
                    np.sin(angle * 0.71) * radial * 0.76 + math.cos(index) * 1.5,
                    anchor_z + self.rng.normal(0.0, 7.8, count) + (u - v) * 3.0,
                )
            )
            colours = self._lift_colours(pixels.reshape(-1, 3)[flat_indices])
            phases = self.rng.uniform(0.0, math.tau, count)
            sizes = self.rng.uniform(0.7, 2.4, count)
            self._append_particles(points, colours, phases, sizes)

    def _append_sdvae_weather(self) -> None:
        for code_index, code in enumerate(FOCUS_CODES):
            cloud = self.assets.sdvae_clouds.get(code)
            if cloud is None or not len(cloud):
                continue
            count = len(cloud)
            anchor = np.array(
                ((code_index - 1.5) * 3.6, (-1.0) ** code_index * 2.1, 138.0 + code_index * 21.0)
            )
            local = np.asarray(cloud[:, :3], dtype=np.float64)
            values = np.asarray(cloud[:, 3], dtype=np.float64)
            points = np.column_stack(
                (
                    anchor[0] + local[:, 0] * 1.65 + self.rng.normal(0.0, 0.85, count),
                    anchor[1] + local[:, 1] * 1.55 + self.rng.normal(0.0, 0.85, count),
                    anchor[2] + local[:, 2] * 4.0 + values * 1.2 + self.rng.normal(0.0, 1.25, count),
                )
            )
            phases = self.rng.uniform(0.0, math.tau, count) + values
            sizes = 0.65 + np.clip(np.abs(values), 0.0, 3.0) * 0.56
            self._append_particles(points, self._palette(code, count), phases, sizes)

    def _append_transitional_weather(self) -> None:
        """Join successive fields so the world behaves as continuous climate."""

        specs = list(self._FIELD_SPECS)
        bridge_count = 165 if self.width >= 1280 else 125
        for left, right in zip(specs, specs[1:]):
            left_code, left_z, left_x, left_y, left_spread = left
            right_code, right_z, right_x, right_y, right_spread = right
            t = self.rng.uniform(0.0, 1.0, bridge_count)
            points = np.column_stack(
                (
                    left_x + (right_x - left_x) * t + self.rng.normal(0.0, 6.2, bridge_count),
                    left_y + (right_y - left_y) * t + self.rng.normal(0.0, 4.4, bridge_count),
                    left_z + (right_z - left_z) * t + self.rng.normal(0.0, 4.8, bridge_count),
                )
            )
            choose_right = self.rng.random(bridge_count) < t
            colours = np.empty((bridge_count, 3), dtype=np.uint8)
            for code, mask in ((left_code, ~choose_right), (right_code, choose_right)):
                if np.any(mask):
                    colours[mask] = self._palette(code, int(mask.sum()))
            phases = self.rng.uniform(0.0, math.tau, bridge_count)
            spread_mix = left_spread + (right_spread - left_spread) * t
            sizes = self.rng.lognormal(0.08, 0.62, bridge_count) * spread_mix
            self._append_particles(points, colours, phases, sizes)

    def _build_density_world(self) -> None:
        count = 410 if self.width >= 1280 else 340
        for code, depth, x, y, spread in self._FIELD_SPECS:
            pca_offset = self.assets.focus_points[code] * np.array((0.30, 0.24, 0.0))
            anchor = np.array((x, y, depth), dtype=np.float64) + pca_offset
            self.field_anchors[code].append(anchor)
            self._append_focus_field(code, anchor, spread, count)
        self._append_transitional_weather()
        self._append_eigen_weather()
        self._append_sdvae_weather()

    def _build_corpus_weather(self) -> None:
        stations = (-2.0, 56.0, 116.0, 178.0, 238.0, 286.0)
        base_parts = []
        colour_parts = []
        phase_parts = []
        weight_parts: dict[str, list[np.ndarray]] = {intent: [] for intent in INTENTS}
        code_lookup = {code: index for index, code in enumerate(self.assets.codes)}
        for station_index, station in enumerate(stations):
            angle = station_index * 0.71
            cosine, sine = math.cos(angle), math.sin(angle)
            source = self.assets.corpus_points
            x = (source[:, 0] * cosine - source[:, 1] * sine) * 1.44
            y = (source[:, 0] * sine + source[:, 1] * cosine) * 0.94
            z = station + source[:, 2] * 1.72
            base_parts.append(np.column_stack((x, y, z)))
            colours = np.tile(np.array((72, 132, 151), dtype=np.uint8), (len(source), 1))
            for code in FOCUS_CODES:
                colours[code_lookup[code]] = np.asarray(FOCUS_COLOURS[code], dtype=np.uint8)
            colour_parts.append(colours)
            phase_parts.append(self.rng.uniform(0.0, math.tau, len(source)))
            for intent in INTENTS:
                vector = self.assets.weighting_vectors[intent]
                weight_parts[intent].append(
                    np.asarray([vector.get(code, 0.0) for code in self.assets.codes], dtype=np.float64)
                )
        self.corpus_weather = np.concatenate(base_parts, axis=0)
        self.corpus_colours = np.concatenate(colour_parts, axis=0)
        self.corpus_phases = np.concatenate(phase_parts, axis=0)
        self.corpus_weights = {
            intent: np.concatenate(parts, axis=0) for intent, parts in weight_parts.items()
        }

    def _build_calm_front(self) -> None:
        count_per_code = 190
        self.calm_base = self.rng.normal(0.0, 1.0, (count_per_code * len(FOCUS_CODES), 3))
        self.calm_base *= np.array((4.8, 3.2, 9.5), dtype=np.float64)
        self.calm_base += self.rng.normal(0.0, (0.9, 0.7, 1.4), self.calm_base.shape)
        self.calm_codes = np.repeat(np.arange(len(FOCUS_CODES)), count_per_code)
        self.calm_colours = np.concatenate(
            [self._palette(code, count_per_code) for code in FOCUS_CODES], axis=0
        )
        self.calm_phases = self.rng.uniform(0.0, math.tau, len(self.calm_base))
        self.calm_sizes = self.rng.uniform(0.65, 1.7, len(self.calm_base))

    def _build_average_verdicts(self) -> None:
        """Turn five real weighted-average rasters into thick particle apparitions."""

        self.verdict_clouds: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        grid_width, grid_height, layers = 24, 17, 2
        grid_x, grid_y = np.meshgrid(np.arange(grid_width), np.arange(grid_height))
        for intent in INTENTS:
            image = self.assets.weighted_averages[intent].resize(
                (grid_width, grid_height), Image.Resampling.BILINEAR
            )
            pixels = np.asarray(image, dtype=np.uint8).reshape(-1, 3)
            luminance = pixels.mean(axis=1) / 255.0
            parts = []
            colours = []
            for layer in range(layers):
                x = (grid_x.reshape(-1) / (grid_width - 1) - 0.5) * 10.8
                y = (0.5 - grid_y.reshape(-1) / (grid_height - 1)) * 7.2
                z = (
                    (layer - (layers - 1) / 2.0) * 1.05
                    + (luminance - 0.5) * 1.25
                    + self.rng.normal(0.0, 0.34, len(pixels))
                )
                parts.append(
                    np.column_stack(
                        (
                            x + self.rng.normal(0.0, 0.13, len(x)),
                            y + self.rng.normal(0.0, 0.13, len(y)),
                            z,
                        )
                    )
                )
                colours.append(pixels)
            points = np.concatenate(parts, axis=0)
            cloud_colours = self._lift_colours(np.concatenate(colours, axis=0))
            phases = self.rng.uniform(0.0, math.tau, len(points))
            sizes = self.rng.uniform(0.72, 1.52, len(points))
            self.verdict_clouds[intent] = (points, cloud_colours, phases, sizes)

    def _film_phase(self, time: float) -> tuple[str, float]:
        for index, (key, start, end) in enumerate(self.film_phases):
            if time < end or index == len(self.film_phases) - 1:
                local = (time - start) / max(1e-9, end - start)
                return key, float(np.clip(local, 0.0, 1.0))
        return "conclusion", 1.0

    def _search_progress(self, time: float) -> float:
        return float(
            np.clip(
                (time - self.search_start) / max(self.search_duration, 1e-9),
                0.0,
                1.0,
            )
        )

    def _phase(self, time: float) -> tuple[int, str, float]:
        for index, (key, start, end) in enumerate(self.segments):
            if time < end or index == len(self.segments) - 1:
                local = (time - start) / max(1e-9, end - start)
                return index, key, float(np.clip(local, 0.0, 1.0))
        return len(self.segments) - 1, self.segments[-1][0], 1.0

    def _chapter(self, time: float) -> tuple[int, str, float]:
        for index, (key, start, end) in enumerate(self.chapters):
            if time < end or index == len(self.chapters) - 1:
                local = (time - start) / max(1e-9, end - start)
                return index, key, float(np.clip(local, 0.0, 1.0))
        return len(self.chapters) - 1, self.chapters[-1][0], 1.0

    def calm_point(self, progress: float) -> np.ndarray:
        weights = parameter_weights(progress)
        return sum(weights[i] * self.focus[i] for i in range(len(FOCUS_CODES)))

    @classmethod
    def _route(cls, progress: float) -> np.ndarray:
        p = float(progress)
        depth = forward_depth(p)
        return cls._route_at_depth(depth)

    @classmethod
    def _route_at_depth(cls, depth: float) -> np.ndarray:
        p = (float(depth) - WORLD_START) / (WORLD_END - WORLD_START)
        depths = np.asarray([item[0] for item in cls._NAV_WAYPOINTS], dtype=np.float64)
        xs = np.asarray([item[1] for item in cls._NAV_WAYPOINTS], dtype=np.float64)
        ys = np.asarray([item[2] for item in cls._NAV_WAYPOINTS], dtype=np.float64)
        x = float(np.interp(depth, depths, xs)) + math.sin(p * math.tau * 4.7) * 0.75
        y = float(np.interp(depth, depths, ys)) + math.sin(p * math.tau * 5.9 + 0.8) * 0.48
        return np.array((x, y, depth), dtype=np.float64)

    def _camera(self, time: float) -> tuple[Camera, str, float]:
        _, key, local = self._phase(time)
        film_phase, film_local = self._film_phase(time)
        progress = self._search_progress(time)
        if film_phase in {"title", "statement"}:
            pre = time / max(self.search_start, 1e-9)
            depth = WORLD_START - 18.0 + 18.0 * _ease(pre)
            energy = 0.12 + pre * 0.18
        elif film_phase == "search":
            depth = forward_depth(accelerated_search_progress(progress))
            energy = 0.28 + 1.38 * progress**1.72
        else:
            travel = 0.12 * film_local + 0.88 * film_local**1.62
            depth = WORLD_END + (WORLD_FAR - WORLD_END - 9.0) * travel
            energy = 1.70 + film_local * 0.85

        position = self._route_at_depth(depth)
        lookahead = 14.0 + energy * 10.5 + 5.0 * (0.5 + 0.5 * math.sin(time * 0.31))
        target = self._route_at_depth(depth + lookahead)
        target[2] = depth + lookahead

        calm = self.calm_point(progress)
        search = (1.7 + 4.9 * progress**1.55) * (0.42 if key == "almost_rest" else 1.0)
        if film_phase == "conclusion":
            search = 5.4 + film_local * 4.2
        target[:2] += np.array(
            (
                math.sin(time * (0.38 + energy * 0.10) + calm[0] * 0.18) * search,
                math.sin(time * (0.49 + energy * 0.11) + calm[1] * 0.22) * search * 0.60,
            )
        )
        jitter = (0.07 + 0.42 * progress**1.65) * (0.38 if key == "almost_rest" else 1.0)
        if film_phase == "conclusion":
            jitter = 0.48 + film_local * 0.48
        position[:2] += np.array(
            (
                math.sin(time * (5.1 + energy * 1.8)) * jitter,
                math.sin(time * (7.0 + energy * 2.2) + 1.2) * jitter * 0.58,
            )
        )

        # Viewpoint edits shift laterally but never reset depth: every shot keeps
        # travelling further into the same world. Their interval contracts as
        # the search accelerates.
        interval = max(0.82, 2.65 - progress * 1.72)
        jump_index = int(max(0.0, time - self.search_start) / interval)
        jump_rng = np.random.default_rng(self.config.seed + 9901 + jump_index)
        if film_phase in {"search", "conclusion"} and key not in {"false_calm", "almost_rest"}:
            jump = 0.56 + progress * 1.16 + (film_local * 0.55 if film_phase == "conclusion" else 0.0)
            position[:2] += jump_rng.uniform((-jump, -jump * 0.62), (jump, jump * 0.62))
            target[:2] += jump_rng.uniform(
                (-jump * 1.34, -jump * 0.78),
                (jump * 1.34, jump * 0.78),
            )
        roll = 0.035 * math.sin(time * 0.37) + float(jump_rng.uniform(-0.05, 0.05)) * energy
        if key in {"collision", "axis_snap", "parameter_break", "escape"}:
            roll += math.sin(time * 3.3) * (0.10 + progress * 0.16)
        focal = self.width * (
            0.78
            + (0.065 + progress * 0.105)
            * math.sin(time * (0.16 + progress * 0.17) + local * math.pi)
        )
        return Camera(position, target, roll, focal), key, local

    def _project_many(
        self,
        points: np.ndarray,
        camera: Camera,
        *,
        far: float = 92.0,
        margin: float = 0.16,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        right, up, forward = camera.basis()
        relative = np.asarray(points, dtype=np.float64) - camera.position
        depth = relative @ forward
        scale = camera.focal / np.maximum(depth, 1e-6)
        x = self.width * 0.5 + (relative @ right) * scale
        y = self.height * 0.5 - (relative @ up) * scale
        mask = (
            (depth > 0.18)
            & (depth < far)
            & (x > -self.width * margin)
            & (x < self.width * (1.0 + margin))
            & (y > -self.height * margin)
            & (y < self.height * (1.0 + margin))
        )
        return x, y, depth, mask

    def _advect(self, points: np.ndarray, phases: np.ndarray, time: float, amount: float = 1.0) -> np.ndarray:
        moved = np.asarray(points, dtype=np.float64).copy()
        moved[:, 0] += (
            np.sin(phases + time * 0.61 + points[:, 2] * 0.071) * 1.10
            + np.sin(time * 0.23 - points[:, 2] * 0.037) * 0.72
        ) * amount
        moved[:, 1] += (
            np.cos(phases * 1.31 - time * 0.73 + points[:, 2] * 0.047) * 0.84
            + np.sin(time * 0.31 + points[:, 2] * 0.052) * 0.55
        ) * amount
        moved[:, 2] += (
            np.sin(phases * 0.73 + time * 0.47) * 1.35
            + np.cos(time * 0.19 + points[:, 0] * 0.12) * 0.58
        ) * amount
        return moved

    def _background(self, camera: Camera, time: float) -> Image.Image:
        yy = np.linspace(0.0, 1.0, self.height, dtype=np.float32)[:, None]
        top = np.array((0.0, 2.0, 7.0), dtype=np.float32)
        bottom = np.array((7.0, 0.0, 12.0), dtype=np.float32)
        gradient = top[None, None, :] * (1.0 - yy[:, :, None]) + bottom[None, None, :] * yy[:, :, None]
        image = Image.fromarray(np.repeat(gradient, self.width, axis=1).astype(np.uint8), "RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        x, y, depth, mask = self._project_many(self.starfield, camera, far=118.0, margin=0.02)
        indices = np.flatnonzero(mask)
        centre_x, centre_y = self.width * 0.5, self.height * 0.5
        for index in indices:
            strength = float(self.star_strength[index])
            alpha = int(28 + strength * 105 * max(0.15, 1.0 - depth[index] / 118.0))
            tail = 0.018 + 0.055 / max(1.0, depth[index])
            tx = x[index] + (centre_x - x[index]) * tail
            ty = y[index] + (centre_y - y[index]) * tail
            colour = (68, 128 + int(78 * strength), 188 + int(60 * strength), alpha)
            draw.line((tx, ty, x[index], y[index]), fill=colour, width=1)
        return image

    def _composite_particles(
        self,
        image: Image.Image,
        points: np.ndarray,
        colours: np.ndarray,
        sizes: np.ndarray,
        camera: Camera,
        *,
        alpha_weights: np.ndarray | None = None,
        far: float = 88.0,
        limit: int = 4000,
    ) -> None:
        x, y, depth, mask = self._project_many(points, camera, far=far)
        indices = np.flatnonzero(mask)
        if not len(indices):
            return
        indices = indices[np.argsort(depth[indices])[::-1]]
        if len(indices) > limit:
            indices = indices[-limit:]
        if alpha_weights is None:
            alpha_weights = np.ones(len(points), dtype=np.float64)

        haze_scale = 0.25
        haze_size = (max(1, self.width // 4), max(1, self.height // 4))
        haze = Image.new("RGBA", haze_size, (0, 0, 0, 0))
        haze_draw = ImageDraw.Draw(haze, "RGBA")
        for index in indices[::3]:
            fog = max(0.0, 1.0 - depth[index] / far)
            alpha = int((8.0 + 30.0 * fog) * alpha_weights[index])
            if alpha <= 0:
                continue
            radius = max(1.0, min(34.0, camera.focal * (0.24 + sizes[index] * 0.22) / depth[index]))
            hx, hy, hr = x[index] * haze_scale, y[index] * haze_scale, radius * haze_scale
            colour = tuple(int(value) for value in colours[index])
            haze_draw.ellipse((hx - hr, hy - hr, hx + hr, hy + hr), fill=(*colour, alpha))
        haze = haze.filter(ImageFilter.GaussianBlur(radius=max(1.0, self.width / 620.0)))
        haze = haze.resize(image.size, Image.Resampling.BILINEAR)
        image.paste(haze, (0, 0), haze)

        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer, "RGBA")
        centre_x, centre_y = self.width * 0.5, self.height * 0.5
        for order, index in enumerate(indices):
            fog = max(0.08, 1.0 - depth[index] / far)
            weight = float(alpha_weights[index])
            alpha = int(min(235.0, (28.0 + 178.0 * fog) * weight))
            if alpha <= 1:
                continue
            radius = max(0.7, min(10.0, camera.focal * (0.010 + sizes[index] * 0.012) / depth[index]))
            colour = tuple(int(value) for value in colours[index])
            if order % 6 == 0 and depth[index] < 34.0:
                tail = min(0.12, 0.016 + 0.40 / max(5.0, depth[index]))
                tx = x[index] + (centre_x - x[index]) * tail
                ty = y[index] + (centre_y - y[index]) * tail
                draw.line((tx, ty, x[index], y[index]), fill=(*colour, alpha // 2), width=max(1, int(radius)))
            draw.ellipse(
                (x[index] - radius, y[index] - radius, x[index] + radius, y[index] + radius),
                fill=(*colour, alpha),
            )
            if radius > 2.8:
                core = radius * 0.24
                draw.ellipse(
                    (x[index] - core, y[index] - core, x[index] + core, y[index] + core),
                    fill=(240, 246, 250, min(190, alpha)),
                )
        image.paste(layer, (0, 0), layer)

    @staticmethod
    def _erasure_threshold(local: float) -> float:
        return 0.00002 * (60.0 ** _ease(local))

    def _draw_corpus_weather(
        self,
        image: Image.Image,
        camera: Camera,
        time: float,
        chapter: str,
        chapter_local: float,
    ) -> None:
        points = self._advect(self.corpus_weather, self.corpus_phases, time, amount=0.72)
        sizes = np.full(len(points), 0.42, dtype=np.float64)
        weights = np.full(len(points), 0.30, dtype=np.float64)
        if chapter == "boundary":
            reveal = (self.corpus_phases % math.tau) / math.tau
            weights = np.where(reveal <= min(1.0, chapter_local * 1.35), 0.62, 0.035)
        elif chapter == "weight":
            intent_index = min(len(INTENTS) - 1, int(chapter_local * len(INTENTS)))
            values = self.corpus_weights[INTENTS[intent_index]]
            maximum = float(values.max()) or 1.0
            weights = 0.035 + np.sqrt(values / maximum) * 1.18
        elif chapter == "erase":
            values = self.corpus_weights["cumulative_co2"]
            threshold = self._erasure_threshold(chapter_local)
            weights = np.where(values >= threshold, 0.86, 0.018)
            sizes = np.where(values >= threshold, 0.58, 0.18)
        self._composite_particles(
            image,
            points,
            self.corpus_colours,
            sizes,
            camera,
            alpha_weights=weights,
            far=104.0,
            limit=1900,
        )

    def _draw_filaments(self, image: Image.Image, camera: Camera, time: float) -> None:
        glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow, "RGBA")
        core = Image.new("RGBA", image.size, (0, 0, 0, 0))
        core_draw = ImageDraw.Draw(core, "RGBA")
        for points, colour, phase in self.filaments:
            if points[-1, 2] < camera.position[2] - 4.0 or points[0, 2] > camera.position[2] + 92.0:
                continue
            moved = points.copy()
            u = np.linspace(0.0, 1.0, len(points))
            moved[:, 0] += np.sin(time * 0.48 + phase + u * 5.0) * 0.56
            moved[:, 1] += np.cos(time * 0.39 + phase * 1.3 + u * 4.0) * 0.42
            x, y, depth, mask = self._project_many(moved, camera, far=94.0)
            visible = np.flatnonzero(mask)
            if len(visible) < 3:
                continue
            path = [(float(x[index]), float(y[index])) for index in visible]
            alpha = int(max(9.0, min(48.0, 68.0 - float(np.mean(depth[visible])) * 0.58)))
            glow_draw.line(path, fill=(*colour, alpha // 3), width=max(2, self.width // 420))
            core_draw.line(path, fill=(*colour, alpha), width=max(1, self.width // 1100))
        glow = glow.filter(ImageFilter.GaussianBlur(radius=max(1.2, self.width / 720.0)))
        image.paste(glow, (0, 0), glow)
        image.paste(core, (0, 0), core)

    def _draw_density_world(
        self,
        image: Image.Image,
        camera: Camera,
        time: float,
        chapter: str,
        local: float,
    ) -> None:
        points = self._advect(self.volume_base, self.volume_phases, time, amount=1.55)
        alpha_weights = np.ones(len(points), dtype=np.float64)
        phase = self.volume_phases

        # The world is not a neutral container. Each computational choice makes
        # a visibly different space. These deformations are artistic mappings;
        # the analytical values remain confined to the trace-locked annotation.
        if chapter == "contested":
            split = np.where(np.sin(phase * 1.7) >= 0.0, 1.0, -1.0)
            pressure = math.sin(math.pi * _ease(local))
            points[:, 0] += split * (0.8 + 4.4 * pressure)
            points[:, 1] += split * np.sin(points[:, 2] * 0.13 + time) * 0.9
        elif chapter == "reconstruct":
            bands = 2.0 + 14.0 * _ease(local)
            points[:, 1] += np.sin(points[:, 2] * bands * 0.035 + phase) * 2.1
            alpha_weights *= 0.55 + 0.45 * (0.5 + 0.5 * np.cos(phase * bands))
        elif chapter == "weight":
            angle = 0.22 * math.sin(time * 0.9) + local * 0.72
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            x = points[:, 0].copy()
            y = points[:, 1].copy()
            leverage = 0.55 + 1.45 * (0.5 + 0.5 * np.sin(phase * 2.3))
            points[:, 0] = (x * cos_a - y * sin_a) * leverage
            points[:, 1] = x * sin_a + y * cos_a
            alpha_weights *= 0.28 + 0.92 * leverage / 2.0
        elif chapter == "erase":
            rank = (phase % math.tau) / math.tau
            survival = max(0.08, 1.0 - 0.90 * _ease(local))
            visible = rank <= survival
            alpha_weights = np.where(visible, 1.18, 0.018)
            points[~visible, :2] *= 0.18
        elif chapter == "retrieve":
            verdict = _ease(local)
            points[:, 0] *= 1.0 - 0.55 * verdict
            points[:, 1] *= 1.0 - 0.42 * verdict
            points[:, 0] += np.sin(points[:, 2] * 0.17 + time * 1.4) * verdict * 2.4
        elif chapter == "synthesize":
            instability = 0.72 + 0.62 * (0.5 + 0.5 * math.sin(time * 2.3))
            points[:, 0] *= instability
            points[:, 1] *= 1.58 - instability * 0.46
            points[:, 2] += np.sin(phase + time * 1.7) * (1.5 + 3.0 * local)
        elif chapter == "unresolved":
            rupture = _ease(local)
            points[:, :2] *= 1.0 + rupture * (0.6 + 0.7 * np.sin(phase))[:, None]
            alpha_weights *= 1.0 - rupture * 0.36
        self._composite_particles(
            image,
            points,
            self.volume_colours,
            self.volume_sizes,
            camera,
            alpha_weights=alpha_weights,
            far=90.0,
            limit=4300 if self.width >= 1280 else 3200,
        )

    def _draw_calm_front(self, image: Image.Image, camera: Camera, time: float, key: str) -> None:
        progress = self._search_progress(time)
        visibility = max(0.0, min(1.0, (progress - 0.42) / 0.24))
        visibility *= 0.42 + 0.58 * (0.5 + 0.5 * math.sin(time * 0.73 + 0.9))
        if visibility < 0.06:
            return
        weights = parameter_weights(progress)
        calm = self.calm_point(progress)
        ahead = 25.0 + 7.0 * (0.5 + 0.5 * math.sin(time * 0.21 + 0.8))
        centre = self._route_at_depth(camera.position[2] + ahead)
        centre[:2] += calm[:2] * 0.42

        contraction = 0.62 if key == "almost_rest" else 1.0
        if key in {"parameter_break", "escape"}:
            contraction = 1.8 + 0.55 * math.sin(time * 4.1) ** 2
        points = self.calm_base * contraction
        swirl = time * 0.31 + self.calm_phases
        points = points.copy()
        points[:, 0] += np.sin(swirl) * (0.35 + contraction * 0.22)
        points[:, 1] += np.cos(swirl * 1.19) * (0.28 + contraction * 0.18)
        points[:, 2] += np.sin(swirl * 0.77) * (0.42 + contraction * 0.28)
        points += centre
        alpha_weights = visibility * (0.08 + weights[self.calm_codes] * 1.48)
        self._composite_particles(
            image,
            points,
            self.calm_colours,
            self.calm_sizes * (1.12 if key == "almost_rest" else 0.92),
            camera,
            alpha_weights=alpha_weights,
            far=72.0,
            limit=len(points),
        )

    def _draw_route(self, image: Image.Image, camera: Camera, time: float) -> None:
        """Draw the navigable computational path as geometry inside the world."""

        depths = np.linspace(camera.position[2] + 1.1, camera.position[2] + 94.0, 92)
        centre = np.stack([self._route_at_depth(float(depth)) for depth in depths])
        # Two narrow rails keep the path unmistakable without returning to a grid.
        tangent = np.gradient(centre[:, :2], axis=0)
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
        rails = []
        for offset in (-0.34, 0.34):
            rail = centre.copy()
            rail[:, :2] += normal * offset
            rails.append(rail)

        glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow, "RGBA")
        core = Image.new("RGBA", image.size, (0, 0, 0, 0))
        core_draw = ImageDraw.Draw(core, "RGBA")
        for rail in rails:
            x, y, _, mask = self._project_many(rail, camera, far=96.0, margin=0.06)
            for index in range(len(rail) - 1):
                if not (mask[index] and mask[index + 1]) or index % 4 == 2:
                    continue
                segment = (x[index], y[index], x[index + 1], y[index + 1])
                glow_draw.line(segment, fill=(36, 224, 255, 74), width=max(3, self.width // 310))
                core_draw.line(segment, fill=(168, 248, 255, 210), width=max(1, self.width // 1050))

        for chapter_index, (key, _, _) in enumerate(self.chapters):
            depth_world = self.chapter_depths[key]
            relative = depth_world - camera.position[2]
            if not (2.0 < relative < 82.0):
                continue
            point = project_point(self._route_at_depth(depth_world), camera, image.size)
            if point is None:
                continue
            x, y, depth = point
            radius = max(5, min(self.width // 24, int(camera.focal * 0.21 / depth)))
            colour = (255, 67, 104, 210) if chapter_index == self._chapter(time)[0] + 1 else (84, 220, 244, 145)
            core_draw.rectangle((x - radius, y - radius, x + radius, y + radius), outline=colour, width=max(1, self.width // 1200))
            core_draw.line((x - radius * 1.35, y, x - radius * 0.45, y), fill=colour, width=1)
            core_draw.line((x + radius * 0.45, y, x + radius * 1.35, y), fill=colour, width=1)
            core_draw.text(
                (x + radius + 5, y - radius),
                f"{chapter_index + 1:02d} / {CHAPTER_LABELS[key]}",
                font=self.hud_tiny,
                fill=colour,
            )
        glow = glow.filter(ImageFilter.GaussianBlur(radius=max(1.3, self.width / 680.0)))
        image.paste(glow, (0, 0), glow)
        image.paste(core, (0, 0), core)

    def _chapter_content(
        self,
        chapter: str,
        local: float,
        time: float,
    ) -> tuple[str, str, list[str], tuple[int, int, int]]:
        findings = self.assets.findings
        if chapter == "boundary":
            return (
                "THE AVERAGE BEGINS WITH A BORDER",
                "SELECTION / WHO IS ALLOWED TO COUNT?",
                [
                    f"ARCHIVE SET      {findings['archive_entity_count']:03d} POLITICAL SYMBOLS",
                    f"COMPARABLE SET   {findings['comparable_entity_count']:03d} COMPLETE METADATA RECORDS",
                    "OPERATION         EXCLUSION PRECEDES AGGREGATION",
                ],
                (98, 228, 255),
            )
        if chapter == "encode":
            return (
                "ONE FLAG / INCOMPATIBLE COORDINATES",
                "REPRESENTATION / THE SPACE CHOOSES WHAT CAN SURVIVE",
                [
                    "RASTER            96 x 72 x RGB",
                    "LOCAL MODEL       PCA-32 / EIGENFLAG BASIS",
                    "FOUNDATION MODEL  SD-VAE / 4 x 24 x 32",
                    f"AVAILABLE SPACES  {len(findings['representations']):02d}",
                ],
                (80, 220, 255),
            )
        if chapter == "contested":
            pair = findings["archived_clip"]["clip_retrieval"]["israel_palestine_equal_pair"]
            return (
                "COMMENSURABLE IS NOT RESOLVED",
                "PALESTINE + ISRAEL / EQUAL INPUT TO A LEARNED SPACE",
                [
                    "INPUT WEIGHT       0.5000 / 0.5000",
                    f"CLIP COSINE       {pair['similarity_each']:.4f} / {pair['similarity_each']:.4f}",
                    f"RETRIEVAL MARGIN  {pair['margin']:.4f} / DEAD STATISTICAL TIE",
                    "POLITICAL OUTPUT   NONE",
                ],
                (255, 67, 104),
            )
        if chapter == "reconstruct":
            code_index = min(len(FOCUS_CODES) - 1, int(local * len(FOCUS_CODES)))
            code = FOCUS_CODES[code_index]
            sub = (local * len(FOCUS_CODES)) % 1.0
            k_index = min(len(RECONSTRUCTION_COMPONENTS) - 1, int(sub * len(RECONSTRUCTION_COMPONENTS)))
            k = RECONSTRUCTION_COMPONENTS[k_index]
            mae = findings["pca_reconstruction_mae_0_255"][code][str(k)]
            return (
                "RECONSTRUCTION IS A NEGOTIATED LOSS",
                f"{self.assets.names[code].upper()} / TRUNCATED EIGENFLAG DECODE",
                [
                    f"ACTIVE BASIS      K={k:02d} / 32 COMPONENTS",
                    f"PIXEL MAE         {mae:06.2f} / 255",
                    f"PCA-3 POSITION    {' '.join(f'{v:+05.2f}' for v in self.assets.focus_points[code])}",
                    "QUESTION           WHAT DID THE MODEL DISCARD?",
                ],
                FOCUS_COLOURS[code],
            )
        if chapter == "weight":
            intent_index = min(len(INTENTS) - 1, int(local * len(INTENTS)))
            intent = INTENTS[intent_index]
            stats = findings["weightings"][intent]
            top = stats["top_five"][0]
            return (
                "WEIGHT IS POLITICAL FORCE",
                INTENT_LABELS[intent].upper(),
                [
                    f"DOMINANT INPUT     {top['code'].upper()} / {top['name'].upper()} / {top['share'] * 100:05.2f}%",
                    f"EFFECTIVE N        {stats['effective_contributor_count']:06.2f} OF {findings['comparable_entity_count']}",
                    f"NORMALISED ENTROPY {stats['normalised_entropy']:.4f}",
                    f"BELOW 0.1%         {stats['below_0_1_percent']:03d} ENTITIES",
                ],
                (255, 198, 45),
            )
        if chapter == "erase":
            threshold = self._erasure_threshold(local)
            vector = self.assets.weighting_vectors["cumulative_co2"]
            count = sum(value < threshold for value in vector.values())
            return (
                "A THRESHOLD PERFORMS DISAPPEARANCE",
                "HISTORICAL CO2 / CONTRIBUTION FILTER",
                [
                    f"ACTIVE THRESHOLD   {threshold * 100:07.4f}%",
                    f"BELOW THRESHOLD    {count:03d} / {len(vector)}",
                    "ZEROED BY DISPLAY  YES",
                    "AUTHORITY           CHOSEN, NOT DISCOVERED",
                ],
                (255, 71, 76),
            )
        if chapter == "retrieve":
            clip_key = "equal_199" if local < 0.48 else "historical_co2_199"
            result = findings["archived_clip"]["clip_retrieval"][clip_key]
            intent = "ONE NATION / ONE VOTE" if clip_key == "equal_199" else "HISTORICAL CO2"
            return (
                "A FOUNDATION MODEL RETURNS A NAME",
                f"CLIP-512 / {intent} / RETRIEVAL HAS NO DECODER",
                [
                    f"NEAREST FLAG      {result['nearest_code'].upper()} / {result['nearest_name'].upper()}",
                    f"COSINE            {result['similarity']:.4f}",
                    "ANSWER TYPE        EXISTING CORPUS MEMBER",
                    "JUSTIFICATION      NOT PROVIDED BY THE MODEL",
                ],
                (189, 107, 255),
            )
        if chapter == "synthesize":
            weights = parameter_weights(time / max(self.config.duration, 1e-9))
            return (
                "THE TARGET MOVES WHEN PARAMETERS MOVE",
                "BARYCENTRIC PCA FRONT / ARTISTIC SEARCH SIGNAL",
                [
                    "PARAMETERS         " + "  ".join(
                        f"{code.upper()} {weights[index]:.3f}" for index, code in enumerate(FOCUS_CODES)
                    ),
                    "SUM                1.000",
                    "STATUS             VALID AVERAGE / UNSTABLE CONDITION",
                    "CAMERA             APPROACHING / NEVER ARRIVING",
                ],
                (229, 244, 255),
            )
        return (
            "NO UNIQUE AVERAGE",
            "THE COMPUTATION ENDS / THE POLITICS DO NOT",
            [
                "SYSTEM             CAN EXECUTE THE CHOICE",
                "SYSTEM             CANNOT JUSTIFY THE CHOICE",
                "RESULT             MULTIPLE / CONDITIONAL / UNRESOLVED",
            ],
            (255, 255, 255),
        )

    def _nearest_anchor_ahead(self, code: str, camera: Camera) -> np.ndarray | None:
        candidates = [anchor for anchor in self.field_anchors[code] if anchor[2] > camera.position[2] + 1.0]
        if candidates:
            return min(candidates, key=lambda anchor: anchor[2])
        return min(self.field_anchors[code], key=lambda anchor: abs(anchor[2] - camera.position[2]))

    def _draw_world_labels(
        self,
        draw: ImageDraw.ImageDraw,
        camera: Camera,
        chapter: str,
        local: float,
    ) -> None:
        codes: tuple[str, ...] = ()
        if chapter == "encode":
            codes = ("ps", "il", "us", "de")
        elif chapter == "contested":
            codes = ("ps", "il")
        elif chapter == "reconstruct":
            codes = (FOCUS_CODES[min(3, int(local * 4))],)
        elif chapter == "weight":
            intent_index = min(len(INTENTS) - 1, int(local * len(INTENTS)))
            top = self.assets.findings["weightings"][INTENTS[intent_index]]["top_five"][0]["code"]
            codes = (top,) if top in self.field_anchors else ()
        elif chapter == "retrieve" and local >= 0.48:
            codes = ("us",)
        for label_index, code in enumerate(codes):
            anchor = self._nearest_anchor_ahead(code, camera)
            if anchor is None:
                continue
            projected = project_point(anchor, camera, (self.width, self.height))
            if projected is None:
                continue
            x, y, depth = projected
            if not (18 < x < self.width - 18 and 18 < y < self.height - 18):
                continue
            radius = max(7, min(26, int(camera.focal * 0.13 / depth)))
            colour = FOCUS_COLOURS[code]
            draw.rectangle((x - radius, y - radius, x + radius, y + radius), outline=(*colour, 220), width=1)
            side = 1 if (label_index + int(x > self.width * 0.5)) % 2 == 0 else -1
            tx = x + side * (radius + self.width * 0.018)
            ty = y - radius - self.height * 0.018
            draw.line((x + side * radius, y, tx, ty + 5), fill=(*colour, 185), width=1)
            anchor_mode = "la" if side > 0 else "ra"
            draw.text(
                (tx, ty),
                f"{code.upper()} // {self.assets.names[code].upper()}",
                font=self.hud_tiny,
                fill=(*colour, 245),
                anchor=anchor_mode,
            )
            pca = self.assets.focus_points[code]
            draw.text(
                (tx, ty + max(11, self.height // 68)),
                "PCA3 " + " ".join(f"{value:+.2f}" for value in pca),
                font=self.hud_tiny,
                fill=(196, 222, 229, 205),
                anchor=anchor_mode,
            )

    @staticmethod
    def _wrap_lines(
        draw: ImageDraw.ImageDraw,
        text: str,
        font: Any,
        max_width: float,
    ) -> list[str]:
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if current and draw.textlength(candidate, font=font) > max_width:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        return lines or [""]

    def _fit_font(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        max_width: float,
        preferred: int,
        minimum: int,
        *,
        mono: bool = True,
    ) -> Any:
        for size in range(preferred, minimum - 1, -1):
            key = (size, mono)
            font = self._font_cache.get(key)
            if font is None:
                font = load_font(size, mono=mono)
                self._font_cache[key] = font
            if draw.textlength(text, font=font) <= max_width:
                return font
        return self._font_cache[(minimum, mono)]

    def _draw_opening_sequence(
        self,
        image: Image.Image,
        phase: str,
        local: float,
        time: float,
    ) -> None:
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer, "RGBA")
        margin = max(28, self.width // 12)
        draw.rectangle((0, 0, self.width, self.height), fill=(0, 2, 7, 184))
        scan_y = int((time * 71.0) % self.height)
        draw.line((0, scan_y, self.width, scan_y), fill=(76, 226, 255, 38), width=1)

        if phase == "title":
            alpha = int(255 * min(1.0, local / 0.18) * min(1.0, (1.0 - local) / 0.12))
            draw.text(
                (margin, self.height * 0.13),
                "ARTWORK / COMPUTATIONAL IMAGE / 2026",
                font=self.open_kicker,
                fill=(92, 225, 245, alpha),
            )
            title = "THE NATIONAL\nAVERAGE"
            title_y = self.height * 0.25
            displacement = max(2, self.width // 520)
            draw.multiline_text(
                (margin + displacement, title_y),
                title,
                font=self.open_title,
                fill=(255, 38, 91, alpha // 3),
                spacing=-max(4, self.height // 90),
            )
            draw.multiline_text(
                (margin - displacement, title_y),
                title,
                font=self.open_title,
                fill=(35, 226, 255, alpha // 3),
                spacing=-max(4, self.height // 90),
            )
            draw.multiline_text(
                (margin, title_y),
                title,
                font=self.open_title,
                fill=(244, 248, 248, alpha),
                spacing=-max(4, self.height // 90),
            )
            if local > 0.36:
                credit_alpha = int(255 * min(1.0, (local - 0.36) / 0.18))
                credit_y = self.height * 0.73
                draw.text(
                    (margin, credit_y),
                    AUTHOR.upper(),
                    font=self.open_credit,
                    fill=(229, 240, 243, credit_alpha),
                )
                draw.text(
                    (margin, credit_y + max(28, self.height // 24)),
                    WEBSITE.upper(),
                    font=self.open_kicker,
                    fill=(92, 225, 245, credit_alpha),
                )
        else:
            alpha = int(255 * min(1.0, local / 0.12) * min(1.0, (1.0 - local) / 0.10))
            title = "WHAT DOES IT MEAN\nTO AVERAGE A NATION?"
            draw.multiline_text(
                (margin, self.height * 0.14),
                title,
                font=self.statement_title,
                fill=(246, 249, 249, alpha),
                spacing=max(5, self.height // 120),
            )
            statements = (
                "251 POLITICAL SYMBOLS ENTER AN ARCHIVE.",
                "199 SURVIVE THE METADATA BOUNDARY.",
                "THE SYSTEM CHOOSES A SPACE, ASSIGNS WEIGHTS, ERASES, AND SYNTHESISES.",
                "THE OUTPUT LOOKS SINGULAR. ITS CONDITIONS ARE NOT.",
            )
            body_y = self.height * 0.50
            line_height = max(26, self.height // 24)
            max_width = self.width - margin * 2
            row = 0
            for index, statement in enumerate(statements):
                reveal = min(1.0, max(0.0, (local - 0.18 - index * 0.13) / 0.11))
                if reveal <= 0.0:
                    continue
                lines = self._wrap_lines(draw, statement, self.statement_body, max_width)
                for wrapped in lines:
                    draw.text(
                        (margin, body_y + row * line_height),
                        wrapped,
                        font=self.statement_body,
                        fill=(174, 216, 224, int(alpha * reveal)),
                    )
                    row += 1
            draw.rectangle(
                (margin - 16, self.height * 0.47, margin - 7, body_y + max(1, row) * line_height),
                fill=(255, 58, 94, alpha),
            )
        image.paste(layer, (0, 0), layer)

    def _process_peek_asset(
        self,
        chapter: str,
        local: float,
        time: float,
    ) -> tuple[Image.Image, str] | None:
        if chapter == "encode":
            index = min(len(self.assets.eigenflags) - 1, int(local * len(self.assets.eigenflags)))
            return self.assets.eigenflags[index], f"EIGENFLAG BASIS / COMPONENT {index + 1:02d}"
        if chapter == "contested":
            image = Image.blend(
                self.assets.flags["ps"].convert("RGB"),
                self.assets.flags["il"].convert("RGB"),
                0.5,
            )
            return image, "PIXEL MEAN / PALESTINE 0.5000 + ISRAEL 0.5000"
        if chapter == "reconstruct":
            code_index = min(len(FOCUS_CODES) - 1, int(local * len(FOCUS_CODES)))
            code = FOCUS_CODES[code_index]
            sub = (local * len(FOCUS_CODES)) % 1.0
            k_index = min(len(RECONSTRUCTION_COMPONENTS) - 1, int(sub * len(RECONSTRUCTION_COMPONENTS)))
            k, image = self.assets.reconstructions[code][k_index]
            return image, f"PCA DECODE / {code.upper()} / K={k:02d}"
        if chapter in {"weight", "erase"}:
            if chapter == "erase":
                intent = "cumulative_co2"
            else:
                intent = INTENTS[min(len(INTENTS) - 1, int(local * len(INTENTS)))]
            return self.assets.weighted_averages[intent], f"WEIGHTED PIXEL MEAN / {INTENT_LABELS[intent].upper()}"
        if chapter == "retrieve":
            intent = "equal" if local < 0.48 else "cumulative_co2"
            return self.assets.weighted_averages[intent], "CLIP QUERY IMAGE / INTERMEDIATE INPUT"
        if chapter == "synthesize":
            index = int((time / max(self.config.duration, 1e-9)) * len(self.assets.average_frames))
            return self.assets.average_frames[index % len(self.assets.average_frames)], "LIVE PCA BARYCENTRIC DECODE"
        return None

    def _draw_process_peek(
        self,
        image: Image.Image,
        chapter: str,
        local: float,
        time: float,
    ) -> None:
        asset = self._process_peek_asset(chapter, local, time)
        if asset is None:
            return
        window = max(0.0, min((local - 0.19) / 0.10, (0.73 - local) / 0.10, 1.0))
        if window <= 0.0:
            return
        source, label = asset
        panel_w = max(180, int(self.width * 0.205))
        image_w = panel_w - max(20, self.width // 90)
        image_h = int(image_w * 0.75)
        panel_h = image_h + max(48, self.height // 15)
        x0 = self.width - max(20, self.width // 24) - panel_w
        y0 = int(self.height * 0.54) - panel_h // 2
        panel = Image.new("RGBA", (panel_w, panel_h), (0, 4, 9, 218))
        draw = ImageDraw.Draw(panel, "RGBA")
        inset = max(8, self.width // 180)
        preview = source.convert("RGB").resize((image_w, image_h), Image.Resampling.BILINEAR)
        if int(time * 8.0) % 19 == 0:
            pixels = np.asarray(preview).copy()
            pixels[:, :, 0] = np.roll(pixels[:, :, 0], max(1, image_w // 90), axis=1)
            preview = Image.fromarray(pixels, "RGB")
        panel.paste(preview, (inset, inset))
        for yy in range(inset, inset + image_h, max(3, self.height // 180)):
            draw.line((inset, yy, inset + image_w, yy), fill=(5, 18, 24, 38), width=1)
        accent = (88, 228, 248, 225)
        draw.rectangle((inset, inset, inset + image_w, inset + image_h), outline=accent, width=1)
        label_font = self._fit_font(
            draw,
            label,
            image_w,
            max(9, self.width // 142),
            max(7, self.width // 210),
        )
        draw.text((inset, inset + image_h + 8), "PROCESS APERTURE / TRACE-BOUND", font=self.hud_tiny, fill=accent)
        draw.text((inset, inset + image_h + max(23, self.height // 55)), label, font=label_font, fill=(218, 235, 238, 235))
        panel.putalpha(int(255 * window))
        image.paste(panel, (x0, y0), panel)

    def _draw_verdict_clouds(
        self,
        image: Image.Image,
        camera: Camera,
        time: float,
        local: float,
    ) -> None:
        right, up, forward = camera.basis()
        x_positions = np.linspace(-12.0, 12.0, len(INTENTS))
        for index, intent in enumerate(INTENTS):
            points, colours, phases, sizes = self.verdict_clouds[intent]
            reveal = float(np.clip((local - 0.05 - index * 0.055) / 0.20, 0.0, 1.0))
            if reveal <= 0.0:
                continue
            moved = points.copy() * (0.47 + reveal * 0.13)
            moved[:, 0] += np.sin(phases + time * 0.83) * (0.18 + local * 0.34)
            moved[:, 1] += np.cos(phases * 1.2 - time * 0.71) * (0.16 + local * 0.28)
            moved[:, 2] += np.sin(phases * 0.7 + time * 1.1) * (0.24 + local * 0.52)
            depth = 27.0 + index * 1.9 + math.sin(time * 0.31 + index) * 1.5
            vertical = math.sin(index * 1.7) * 2.8
            world = (
                camera.position
                + forward * depth
                + right * x_positions[index]
                + up * vertical
                + moved[:, 0, None] * right
                + moved[:, 1, None] * up
                + moved[:, 2, None] * forward
            )
            self._composite_particles(
                image,
                world,
                colours,
                sizes * 1.22,
                camera,
                alpha_weights=np.full(len(world), reveal * (1.0 - max(0.0, local - 0.86) * 3.2)),
                far=68.0,
                limit=len(world),
            )

    def _draw_conclusion_overlay(self, image: Image.Image, local: float, time: float) -> None:
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer, "RGBA")
        margin = max(22, self.width // 25)
        alpha = int(255 * min(1.0, local / 0.10) * min(1.0, (1.0 - local) / 0.045))
        draw.text(
            (margin, margin),
            "OUTPUT ARRAY / SAME CORPUS / FIVE WEIGHTINGS",
            font=self.hud_heading,
            fill=(222, 242, 245, alpha),
        )
        labels = []
        for intent in INTENTS:
            stats = self.assets.findings["weightings"][intent]
            top = stats["top_five"][0]
            labels.append(
                f"{INTENT_LABELS[intent].upper()}\n{top['name'].upper()} {top['share'] * 100:05.2f}%"
            )
        label_y = int(self.height * 0.74)
        for index, label in enumerate(labels):
            x = margin + (self.width - margin * 2) * (index + 0.5) / len(labels)
            draw.multiline_text(
                (x, label_y),
                label,
                font=self.hud_tiny,
                fill=(135, 206, 218, int(alpha * 0.82)),
                anchor="ma",
                align="center",
                spacing=3,
            )
        if local > 0.52:
            verdict = min(1.0, (local - 0.52) / 0.16)
            veil_alpha = int(172 * verdict)
            draw.rectangle((0, self.height * 0.28, self.width, self.height * 0.68), fill=(0, 2, 6, veil_alpha))
            statement = "THE AVERAGE IS NOT FOUND.\nIT IS ENFORCED."
            bbox = draw.multiline_textbbox((0, 0), statement, font=self.verdict_title, spacing=6, align="center")
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            x = (self.width - text_w) / 2
            y = self.height * 0.47 - text_h / 2
            offset = max(2, self.width // 650)
            draw.multiline_text((x + offset, y), statement, font=self.verdict_title, fill=(255, 35, 88, int(110 * verdict)), spacing=6, align="center")
            draw.multiline_text((x - offset, y), statement, font=self.verdict_title, fill=(35, 224, 255, int(110 * verdict)), spacing=6, align="center")
            draw.multiline_text((x, y), statement, font=self.verdict_title, fill=(246, 249, 249, int(255 * verdict)), spacing=6, align="center")
            if local > 0.72:
                sub_alpha = int(230 * min(1.0, (local - 0.72) / 0.12))
                draw.text(
                    (self.width / 2, self.height * 0.66),
                    "THE SYSTEM CAN EXECUTE THE CHOICE. IT CANNOT JUSTIFY IT.",
                    font=self.open_kicker,
                    fill=(126, 214, 228, sub_alpha),
                    anchor="ma",
                )
        image.paste(layer, (0, 0), layer)

    def _draw_interface(
        self,
        image: Image.Image,
        camera: Camera,
        time: float,
        frame_index: int,
    ) -> None:
        chapter_index, chapter, local = self._chapter(time)
        headline, subtitle, lines, accent = self._chapter_content(chapter, local, time)
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer, "RGBA")
        margin = max(15, self.width // 52)
        arm = max(22, self.width // 36)
        line_width = max(1, self.width // 1200)

        # Sparse military/cyber frame: orientation without turning the image into
        # a dashboard.
        frame_colour = (*accent, 155)
        for cx, cy, dx, dy in (
            (margin, margin, 1, 1),
            (self.width - margin, margin, -1, 1),
            (margin, self.height - margin, 1, -1),
            (self.width - margin, self.height - margin, -1, -1),
        ):
            draw.line((cx, cy, cx + dx * arm, cy), fill=frame_colour, width=line_width)
            draw.line((cx, cy, cx, cy + dy * arm), fill=frame_colour, width=line_width)

        panel_x = margin + max(7, self.width // 180)
        panel_y = margin + max(8, self.height // 42)
        panel_w = min(int(self.width * 0.54), max(390, int(self.width * 0.54)))
        inner_width = panel_w - max(12, self.width // 120)
        header_text = f"SEARCH / {chapter_index + 1:02d}.{len(self.chapters):02d} / {CHAPTER_LABELS[chapter]}"
        header_font = self._fit_font(
            draw,
            header_text,
            inner_width,
            max(11, self.width // 96),
            max(8, self.width // 150),
        )
        subtitle_lines = self._wrap_lines(draw, subtitle, self.hud_heading, inner_width)
        heading_bbox = draw.textbbox((0, 0), "Ag", font=self.hud_heading)
        heading_step = max(18, heading_bbox[3] - heading_bbox[1] + max(3, self.height // 180))
        heading_y = panel_y + max(20, self.height // 39)
        longest_row = max(lines, key=len)
        body_font = self._fit_font(
            draw,
            longest_row,
            inner_width,
            max(11, self.width // 96),
            max(8, self.width // 150),
        )
        body_bbox = draw.textbbox((0, 0), "Ag", font=body_font)
        row_step = max(15, body_bbox[3] - body_bbox[1] + max(3, self.height // 180))
        row_y = heading_y + len(subtitle_lines) * heading_step + max(8, self.height // 90)
        panel_h = max(122, int(row_y + len(lines) * row_step + 10 - panel_y))
        draw.rectangle(
            (panel_x - 10, panel_y - 8, panel_x + panel_w, panel_y + panel_h),
            fill=(0, 3, 8, 128),
            outline=(*accent, 74),
            width=1,
        )
        draw.rectangle((panel_x - 10, panel_y - 8, panel_x - 4, panel_y + panel_h), fill=(*accent, 205))
        draw.text(
            (panel_x, panel_y),
            header_text,
            font=header_font,
            fill=(*accent, 245),
        )
        for heading_index, heading_line in enumerate(subtitle_lines):
            draw.text(
                (panel_x, heading_y + heading_index * heading_step),
                heading_line,
                font=self.hud_heading,
                fill=(242, 248, 250, 245),
            )
        for row, line in enumerate(lines):
            draw.text(
                (panel_x, row_y + row * row_step),
                line,
                font=body_font,
                fill=(191, 220, 225, 226 if row < 3 else 190),
            )

        # Renderer state is explicitly labelled as navigation, never analysis.
        top_right = self.width - margin - max(5, self.width // 180)
        progress = self._search_progress(time)
        previous_time = max(0.0, time - 1.0 / max(1, self.config.fps))
        previous_camera = self._camera(previous_time)[0]
        speed = max(0.0, camera.position[2] - previous_camera.position[2]) * self.config.fps
        state = (
            "TNA / LATENT-WORLD SEARCH\n"
            f"NAV Z {camera.position[2]:+08.2f}  FORWARD {speed:04.2f}u/s\n"
            f"FRAME {frame_index:06d}  SEARCH {progress * 100:06.2f}%\n"
            "ANALYTICS TRACE-LOCKED / WORLD + CAMERA ARTISTIC"
        )
        draw.multiline_text(
            (top_right, margin + max(8, self.height // 42)),
            state,
            font=self.hud_tiny,
            fill=(164, 218, 229, 210),
            anchor="ra",
            align="right",
            spacing=max(2, self.height // 260),
        )

        # A large chapter command punches in briefly, with deliberate RGB
        # displacement at the cut while the smaller data panel stays readable.
        command_alpha = int(255 * max(0.0, 1.0 - local / 0.23))
        if command_alpha > 4:
            command_y = int(self.height * 0.48)
            offset = max(2, self.width // 620)
            command_font = self._fit_font(
                draw,
                headline,
                self.width - margin * 2,
                max(24, self.width // 29),
                max(16, self.width // 72),
            )
            draw.text(
                (margin + offset, command_y),
                headline,
                font=command_font,
                fill=(255, 35, 93, command_alpha // 2),
            )
            draw.text(
                (margin - offset, command_y),
                headline,
                font=command_font,
                fill=(35, 230, 255, command_alpha // 2),
            )
            draw.text(
                (margin, command_y),
                headline,
                font=command_font,
                fill=(246, 249, 247, command_alpha),
            )

        self._draw_world_labels(draw, camera, chapter, local)

        # Route legend doubles as the film's argumentative structure.
        route_y = self.height - margin - max(19, self.height // 31)
        route_x0 = margin + arm
        route_x1 = self.width - margin - arm
        draw.line((route_x0, route_y, route_x1, route_y), fill=(83, 171, 187, 120), width=1)
        count = len(self.chapters)
        for index, (key, _, _) in enumerate(self.chapters):
            x = route_x0 + (route_x1 - route_x0) * index / max(1, count - 1)
            active = index == chapter_index
            colour = (*accent, 255) if active else (94, 149, 161, 150)
            radius = max(3, self.width // (360 if active else 520))
            draw.rectangle((x - radius, route_y - radius, x + radius, route_y + radius), fill=colour)
            draw.text(
                (x, route_y + radius + max(4, self.height // 180)),
                f"{index + 1:02d} {ROUTE_LABELS[key]}",
                font=self.hud_tiny,
                fill=colour,
                anchor="ma",
            )
        # Moving scan trace and centre acquisition mark establish restless
        # instrumentation without attaching invented values to the scene.
        scan_y = int((time * 83.0) % self.height)
        draw.line((margin, scan_y, self.width - margin, scan_y), fill=(*accent, 24), width=1)
        cx, cy = self.width // 2, self.height // 2
        gap, reach = max(5, self.width // 250), max(14, self.width // 95)
        draw.line((cx - reach, cy, cx - gap, cy), fill=(*accent, 115), width=1)
        draw.line((cx + gap, cy, cx + reach, cy), fill=(*accent, 115), width=1)
        draw.line((cx, cy - reach, cx, cy - gap), fill=(*accent, 115), width=1)
        draw.line((cx, cy + gap, cx, cy + reach), fill=(*accent, 115), width=1)
        image.paste(layer, (0, 0), layer)
        self._draw_process_peek(image, chapter, local, time)

    def _cut_glitch(self, time: float, key: str, local: float) -> tuple[float, str]:
        edge = min(local, 1.0 - local)
        cut = max(0.0, 1.0 - edge / 0.035)
        micro_phase = (time % 2.35) / 2.35
        micro_cut = max(0.0, 1.0 - min(micro_phase, 1.0 - micro_phase) / 0.045)
        restless = 0.065 + 0.085 * (0.5 + 0.5 * math.sin(time * 3.7))
        if key in {"collision", "axis_snap", "parameter_break", "escape"}:
            restless += 0.15
        amount = min(0.94, restless + cut * 0.68 + micro_cut * 0.52)
        modes = ("channel", "latent", "pixel_sort", "patch_shuffle", "block")
        mode = modes[(int(time * 1.7) + list(dict(FLIGHT_SHOTS)).index(key)) % len(modes)]
        return amount, mode

    def render(self, frame_index: int) -> Image.Image:
        time = frame_index / self.config.fps
        camera, key, local = self._camera(time)
        film_phase, film_local = self._film_phase(time)
        image = self._background(camera, time)
        if film_phase in {"title", "statement"}:
            if film_phase == "statement":
                self._draw_corpus_weather(image, camera, time, "boundary", film_local)
            transition = max(0.0, 1.0 - min(film_local, 1.0 - film_local) / 0.07)
            image = effects.treat(
                image,
                vmask=self.vmask,
                glitch=0.07 + transition * 0.34,
                glitch_mode="channel" if film_phase == "title" else "block",
                seed=self.config.seed + 991,
                frame=frame_index,
                grain_amount=4.1,
                aberration=1,
            )
            self._draw_opening_sequence(image, film_phase, film_local, time)
        elif film_phase == "search":
            _, chapter, chapter_local = self._chapter(time)
            self._draw_corpus_weather(image, camera, time, chapter, chapter_local)
            self._draw_filaments(image, camera, time)
            self._draw_density_world(image, camera, time, chapter, chapter_local)
            self._draw_calm_front(image, camera, time, key)
            self._draw_route(image, camera, time)

            glitch, mode = self._cut_glitch(time, key, local)
            chapter_edge = min(chapter_local, 1.0 - chapter_local)
            chapter_cut = max(0.0, 1.0 - chapter_edge / 0.028)
            acceleration = self._search_progress(time) ** 1.8
            glitch = min(0.96, glitch + chapter_cut * 0.52 + acceleration * 0.08)
            image = effects.treat(
                image,
                vmask=self.vmask,
                glitch=glitch,
                glitch_mode=mode,
                seed=self.config.seed + 991,
                frame=frame_index,
                grain_amount=4.8 + acceleration * 1.2,
                aberration=2 + int(acceleration * 2),
            )
            self._draw_interface(image, camera, time, frame_index)
        else:
            self._draw_corpus_weather(image, camera, time, "unresolved", film_local)
            self._draw_filaments(image, camera, time)
            self._draw_density_world(image, camera, time, "unresolved", film_local)
            self._draw_verdict_clouds(image, camera, time, film_local)
            glitch = min(0.93, 0.16 + film_local * 0.34 + 0.16 * math.sin(time * 4.7) ** 2)
            image = effects.treat(
                image,
                vmask=self.vmask,
                glitch=glitch,
                glitch_mode=("latent", "channel", "block")[int(time * 2.0) % 3],
                seed=self.config.seed + 991,
                frame=frame_index,
                grain_amount=5.6,
                aberration=3,
            )
            self._draw_conclusion_overlay(image, film_local, time)
        fade_in = min(1.0, time / 0.45)
        fade_out = min(1.0, max(0.0, (self.config.duration - time) / 0.55))
        fade = min(fade_in, fade_out)
        if fade < 1.0:
            image = Image.blend(Image.new("RGB", image.size, (0, 0, 0)), image, fade)
        return image


def _write_notes(config: VideoRenderConfig, assets: LatentFlightAssets, output: Path) -> Path:
    path = config.out_dir / "latent_flight_notes.md"
    path.write_text(
        "\n".join(
            (
                f"# {TITLE}",
                "",
                "Independent moving-image study; the canonical and NeurIPS renderers are unchanged.",
                "",
                "The full flag corpus recurs as spatial weather derived from the first three coordinates ",
                "of a 32-component PCA/eigenflag representation. Palestine, Israel, the United States, ",
                "and Germany occupy diffuse colour-density fields made from source and reconstructed ",
                "pixels; no flags or reconstructions are displayed as planes. Eigenflags become spectral ",
                "mist. Stable Diffusion VAE posterior means deform turbulent volumes when enabled.",
                "",
                "A nine-stage route makes the argument legible: corpus boundary, representation, contested ",
                "commensuration, reconstruction, weighting, threshold erasure, foundation-model retrieval, ",
                "a moving synthesis, and non-settlement. Displayed counts, similarities, reconstruction ",
                "errors, contributor concentration, entropy, and retrievals come from computed project records.",
                "",
                "The receding colour front is modulated by a valid barycentric PCA average. Its four weights ",
                "are deterministic artistic parameters that continuously change, so the camera can approach ",
                "but cannot settle. Route geometry, chapter-responsive deformation, lateral cuts, advection, ",
                "fog, glitches, and colour are explicitly labelled artistic treatment rather than analysis.",
                "",
                "The opening identifies Tomas Laurenzo and laurenzo.net, then states the work's operation. ",
                "Transient process apertures expose real intermediate representations. The conclusion turns ",
                "five real pixel-space weighted averages into volumetric apparitions before presenting the ",
                "proposition: The average is not found. It is enforced.",
                "",
                f"Render: {config.width}x{config.height}, {config.fps} fps, {config.duration:.1f}s.",
                f"Video: `{output.name}`.",
                f"SD-VAE: {assets.manifest['sdvae']['status']}.",
                "",
                "See `provenance/latent_flight_provenance.json` and ",
                "`assets-latent-flight/asset_manifest.json`.",
            )
        ),
        encoding="utf-8",
    )
    return path


def render_latent_flight(config: VideoRenderConfig) -> dict[str, Any]:
    """Render the independent latent-flight film and return its output paths."""

    config.out_dir.mkdir(parents=True, exist_ok=True)
    required = config.foundation == "required"
    model_manifest = None
    if config.foundation != "off":
        model_manifest = prepare_foundation_assets(config.out_dir, required=required)
    assets = build_latent_flight_assets(config, model_manifest)

    audio_path = None
    if config.audio:
        from .latent_flight_audio import render_latent_flight_soundtrack

        audio_path = render_latent_flight_soundtrack(assets, config)

    output_path = config.out_dir / "the_national_average_latent_world.mp4"
    renderer = LatentFlightRenderer(config, assets)
    collection_was_enabled = gc.isenabled()
    gc.disable()
    try:
        render_video_stream(
            config,
            renderer,
            output_path,
            config.out_dir / "stills-latent-flight",
            audio_path=audio_path,
            encoder_preset="veryfast",
            crf=18 if config.preset == "production" else 20,
        )
    finally:
        if collection_was_enabled:
            gc.enable()
            gc.collect()
    notes = _write_notes(config, assets, output_path)
    return {
        "video": str(output_path),
        "audio": str(audio_path) if audio_path else None,
        "stills": str(config.out_dir / "stills-latent-flight"),
        "asset_manifest": str(assets.asset_dir / "asset_manifest.json"),
        "provenance": str(config.out_dir / "provenance" / "latent_flight_provenance.json"),
        "notes": str(notes),
    }
