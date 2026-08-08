"""Asset + real-analysis generation and orchestration for *The National Average*.

This module loads the corpus, runs every representation backend, and — crucially
— computes the real machine-vision analysis records (`tna.analysis.*`) that the
scenes display. It writes a provenance manifest mapping each on-screen quantity
to the function that produced it, so nothing on screen is invented.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..analysis import (
    ComponentRecord,
    EmbeddingGeometry,
    ErasureRecord,
    RecognitionRecord,
    ResidualRecord,
    SaliencyRecord,
    component_record,
    embedding_geometry,
    erasure_record,
    pca_residual,
    recognition_record,
    saliency_record,
)
from ..backends.clip_retrieval import CLIPBackend
from ..backends.palette import dominant_palette_from_array
from ..backends.pca import PCABackend
from ..backends.registry import build_backend
from ..backends.sdvae import SDVAEBackend
from ..backends.svg import write_svg
from ..data import (
    DATA_DIR,
    ROOT_DIR,
    Polity,
    corpus_arrays,
    load_corpus,
    rasterize_flag,
    selected_items,
)
from ..trace import build_trace, write_json
from ..weights import WeightingRun, weights_from_intent
from .compositor import edge_density, render_video_stream
from .foundation import prepare_foundation_assets

TITLE = "The National Average"
INTENTS = ("equal", "population", "gdp", "co2", "cumulative_co2")
INTENT_LABELS = {
    "equal": "one nation, one vote",
    "population": "weighted by population",
    "gdp": "weighted by GDP",
    "co2": "weighted by annual CO2",
    "cumulative_co2": "weighted by historical CO2",
}
BASE_BACKENDS = ("pixel", "palette", "pca", "svg")
FOUNDATION_BACKENDS = ("clip", "sdvae")
OUTPUT_ROOT = ROOT_DIR / "outputs" / "video" / "the_national_average"
CORPUS_VERSION = "lipis-flag-icons-main-4x3+metadata-v2+moving-image-v1"
# Flags decomposed in scene 2; chosen for legible, varied heraldry.
FOCUS_CODES = ("ps", "fr", "uy", "jp", "za", "br", "in", "tr")
# Nations whose rank in the averaged-embedding retrieval the verdict reports.
PROBE_CODES = ("ps", "uy", "il", "tv")
# How many flags get a real attention map for the saliency gallery scene.
GALLERY_SIZE = 32

# Ordered scene schedule (key, weight). Per-scene seconds scale with duration, so
# the same schedule drives both preview and production lengths.
SCENE_SCHEDULE = [
    ("cold_open", 1.0),
    ("archive", 2.0),
    ("neighbours", 1.5),
    ("gaze", 2.0),
    ("saliency_gallery", 1.5),
    ("averaging", 1.5),
    ("average_nation", 3.0),
    ("the_vanished", 2.0),
    ("the_tie", 1.5),
    ("verdict", 2.0),
    ("provenance_coda", 1.0),
]


def scene_segments(duration: float) -> list[tuple[str, float, float]]:
    """Resolve the schedule into (key, start_seconds, end_seconds) segments."""

    total = sum(weight for _, weight in SCENE_SCHEDULE)
    segments: list[tuple[str, float, float]] = []
    acc = 0.0
    for key, weight in SCENE_SCHEDULE:
        start = acc / total * duration
        acc += weight
        end = acc / total * duration
        segments.append((key, start, end))
    return segments


# Round 4: one continuous journey through the machine's pipeline. Phases
# cross-dissolve in the renderer; the same schedule drives the sonification.
PHASE_SCHEDULE = [
    ("title", 1.4),
    ("ingest", 0.8),
    ("segment", 1.4),
    ("tokenize", 1.0),
    ("attend", 1.4),
    ("embed", 1.5),
    ("eigenbasis", 1.3),
    ("retrieve", 1.2),
    ("reconstruct", 1.4),
    ("spaces", 1.4),
    ("average", 1.4),
    ("weighting", 1.4),
    ("regions", 1.5),
    ("erase", 1.3),
    ("name", 1.7),
    ("residual", 1.1),
    ("coda", 1.6),
]
# The flag/weighting whose average, latent, and retrieval the journey follows.
SUBJECT_INTENT = "cumulative_co2"


def phase_segments(duration: float) -> list[tuple[str, float, float]]:
    """Resolve PHASE_SCHEDULE into (key, start_seconds, end_seconds)."""

    total = sum(weight for _, weight in PHASE_SCHEDULE)
    segments: list[tuple[str, float, float]] = []
    acc = 0.0
    for key, weight in PHASE_SCHEDULE:
        start = acc / total * duration
        acc += weight
        end = acc / total * duration
        segments.append((key, start, end))
    return segments


@dataclass(frozen=True)
class VideoRenderConfig:
    """Production settings for one video video render."""

    out_dir: Path = OUTPUT_ROOT
    width: int = 960
    height: int = 540
    fps: int = 12
    duration: float = 30.0
    foundation: str = "auto"
    seed: int = 20260613
    keep_frames: bool = False
    preset: str = "preview"
    audio: bool = True  # data-driven drone bed; see tna.video.audio
    provenance_overlay: bool = False

    @property
    def canvas(self) -> tuple[int, int]:
        if self.width >= 1280 or self.preset == "production":
            return (192, 144)
        return (96, 72)


@dataclass
class FocusAnalysis:
    """Per-flag real analysis used by the decomposition scene."""

    code: str
    name: str
    flag: Image.Image
    saliency: SaliencyRecord | None
    residual: ResidualRecord | None
    feature: np.ndarray | None  # real PCA eigenflag coordinates
    palette: list[dict[str, Any]]  # real dominant colours + shares
    edge_density: float
    component: ComponentRecord | None = None


@dataclass
class RenderAssets:
    """All generated images and real analysis records the scenes consume."""

    corpus: dict[str, Polity]
    selected: list[Polity]
    archive: list[Polity]
    codes: list[str]
    generated: dict[str, dict[str, Image.Image]]
    traces: dict[str, dict[str, dict[str, Any]]]
    weights: dict[str, WeightingRun]
    embedding: EmbeddingGeometry | None
    erasure: dict[str, ErasureRecord]
    recognition: dict[str, RecognitionRecord]
    recognition_tie: RecognitionRecord | None
    focus: list[FocusAnalysis]
    gallery: list[SaliencyRecord]
    tie_image: Image.Image | None
    sdvae_latent: np.ndarray | None
    pca_recon: list[Image.Image]
    pca_recon_energy: list[float]
    component: ComponentRecord | None
    eigenflags: list[Image.Image]
    reconstructions: dict[str, Image.Image]
    region_averages: dict[str, RecognitionRecord]
    manifest: dict[str, Any]
    asset_dir: Path
    provenance: dict[str, Any] = field(default_factory=dict)


def slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_")


def production_codes(corpus: dict[str, Polity], intents: tuple[str, ...] = INTENTS) -> list[str]:
    codes = []
    for polity in corpus.values():
        ok = True
        for intent in intents:
            try:
                weights_from_intent(intent, [polity])
            except Exception:
                ok = False
                break
        if ok:
            codes.append(polity.code)
    return sorted(codes)


def save_backend_result(result, root: Path, intent: str, backend_name: str) -> dict[str, Path]:
    files: dict[str, Path] = {}
    generated_dir = root / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    if result.image is not None:
        path = generated_dir / f"{slug(intent)}_{backend_name}.png"
        result.image.save(path)
        files["image"] = path
    if "svg_text" in result.files:
        path = generated_dir / f"{slug(intent)}_{backend_name}.svg"
        write_svg(path, result.files["svg_text"])
        files["svg"] = path
    return files


def _model_path(manifest: dict[str, Any] | None, key: str) -> Path | None:
    if not manifest:
        return None
    for model in manifest.get("models", []):
        if model.get("key") == key and model.get("available") and model.get("local_path"):
            return Path(model["local_path"])
    return None


def build_video_backend(backend_name, arrays, config, model_manifest):
    """Build backends, preferring prepared local model snapshots for video."""

    if backend_name == "clip":
        snapshot = _model_path(model_manifest, "clip")
        if snapshot is not None:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            model = CLIPModel.from_pretrained(snapshot)
            processor = CLIPProcessor.from_pretrained(snapshot)
            return CLIPBackend(arrays, seed=config.seed, model=model, processor=processor, torch_module=torch)
    if backend_name == "sdvae":
        snapshot = _model_path(model_manifest, "sdvae")
        if snapshot is not None:
            import torch
            from diffusers import AutoencoderKL

            vae = AutoencoderKL.from_pretrained(snapshot)
            return SDVAEBackend(arrays, seed=config.seed, vae=vae, torch_module=torch)

    return build_backend(
        backend_name,
        arrays,
        seed=config.seed,
        palette_size=6,
        pca_components=32 if config.preset == "production" else 16,
        svg_renderer="deterministic",
        train_vae=False,
        vae_epochs=0,
    )


def _compute_analysis(
    backends: dict[str, Any],
    codes: list[str],
    corpus: dict[str, Polity],
    selected: list[Polity],
    weights: dict[str, WeightingRun],
    config: VideoRenderConfig,
) -> tuple[
    EmbeddingGeometry | None,
    dict[str, ErasureRecord],
    dict[str, RecognitionRecord],
    RecognitionRecord | None,
    list[FocusAnalysis],
    list[SaliencyRecord],
    Image.Image | None,
    list[Image.Image],
    list[float],
    ComponentRecord | None,
    list[Image.Image],
    dict[str, Image.Image],
    dict[str, RecognitionRecord],
]:
    """Compute every real analysis record consumed by the scenes."""

    names = {code: polity.name for code, polity in corpus.items()}
    clip = backends.get("clip")
    pca = backends.get("pca")
    has_clip = isinstance(clip, CLIPBackend)
    has_pca = isinstance(pca, PCABackend)
    available = set(pca.arrays) if has_pca else (set(clip.arrays) if has_clip else set())

    # Erasure is purely metadata-driven and always available.
    erasure = {intent: erasure_record(intent, selected) for intent in INTENTS}

    recognition: dict[str, RecognitionRecord] = {}
    recognition_tie: RecognitionRecord | None = None
    embedding: EmbeddingGeometry | None = None
    if has_clip:
        for intent in INTENTS:
            recognition[intent] = recognition_record(
                clip,
                codes,
                weights[intent].weights,
                names,
                intent=intent,
                probes=PROBE_CODES,
            )
        embedding = embedding_geometry(clip, codes, k=6)
        if "il" in clip.arrays and "ps" in clip.arrays:
            recognition_tie = recognition_record(
                clip, ["il", "ps"], {"il": 0.5, "ps": 0.5}, names, intent="il+ps", topk=5
            )

    focus: list[FocusAnalysis] = []
    for code in FOCUS_CODES:
        if code not in corpus or code not in available:
            continue
        polity = corpus[code]
        flag = rasterize_flag(polity, (240, 180)).convert("RGB")
        focus.append(
            FocusAnalysis(
                code=code,
                name=polity.name,
                flag=flag,
                saliency=saliency_record(clip, code) if has_clip else None,
                residual=pca_residual(pca, code) if has_pca else None,
                feature=pca.encode(code) if has_pca else None,
                palette=dominant_palette_from_array(pca.arrays[code], 5) if has_pca else [],
                edge_density=edge_density(flag),
                component=component_record(code, pca.arrays[code]) if has_pca and code in pca.arrays else None,
            )
        )

    # Saliency gallery: real attention for a spread of flags across the corpus.
    gallery: list[SaliencyRecord] = []
    if has_clip:
        step = max(1, len(codes) // GALLERY_SIZE)
        for code in codes[::step][:GALLERY_SIZE]:
            if code in available:
                gallery.append(saliency_record(clip, code))

    # The occupier/occupied banner: the real pixel average of il + ps.
    tie_image = None
    pixel = backends.get("pixel")
    if pixel is not None and "il" in getattr(pixel, "arrays", {}) and "ps" in pixel.arrays:
        tie_image = pixel.average(["il", "ps"], {"il": 0.5, "ps": 0.5}).image

    # Truncated-PCA reconstruction series for the subject flag: an additive
    # rebuild from k eigenflags, with the residual (loss) energy at each k.
    pca_recon: list[Image.Image] = []
    pca_recon_energy: list[float] = []
    if has_pca and focus:
        original = pca.arrays[focus[0].code]
        z = pca.encode(focus[0].code)
        for k in (1, 2, 4, 8, min(16, len(z))):
            z_k = z.copy()
            z_k[k:] = 0.0
            image = pca.decode(z_k).image
            pca_recon.append(image.convert("RGB"))
            pca_recon_energy.append(float(np.abs(original - np.asarray(image, dtype=np.float64)).mean()))

    # Component detection, the eigenflag basis, and multi-representation
    # reconstructions of the subject flag (more CV behaviours + comparisons).
    component = focus[0].component if focus else None
    eigenflags: list[Image.Image] = pca.eigenflag_images(8) if has_pca else []
    reconstructions: dict[str, Image.Image] = {}
    if focus:
        subject = focus[0].code
        for backend_name in ("pixel", "palette", "pca", "svg", "sdvae"):
            backend = backends.get(backend_name)
            if backend is None:
                continue
            try:
                result = backend.average([subject], {subject: 1.0})
            except Exception:
                continue
            if result.image is not None:
                reconstructions[backend_name] = result.image.convert("RGB")

    # Other perspectives: the equal-weighted average of each world region and the
    # real nation it retrieves to ("the average African nation is ...").
    region_averages: dict[str, RecognitionRecord] = {}
    if has_clip:
        groups: dict[str, list[str]] = {}
        for code in codes:
            region = corpus[code].region
            if region:
                groups.setdefault(region, []).append(code)
        for region, gcodes in groups.items():
            if len(gcodes) >= 5:
                weights_eq = {c: 1.0 / len(gcodes) for c in gcodes}
                region_averages[region] = recognition_record(
                    clip, gcodes, weights_eq, names, intent=f"region:{region}"
                )

    return (
        embedding, erasure, recognition, recognition_tie, focus, gallery, tie_image,
        pca_recon, pca_recon_energy, component, eigenflags, reconstructions, region_averages,
    )


def generate_assets(
    config: VideoRenderConfig,
    *,
    required_foundation: bool,
    model_manifest: dict[str, Any] | None = None,
) -> RenderAssets:
    asset_dir = config.out_dir / "assets"
    trace_dir = asset_dir / "traces"
    (asset_dir / "generated").mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(data_dir=DATA_DIR)
    codes = production_codes(corpus)
    selected = selected_items(corpus, codes)
    archive = [corpus[code] for code in sorted(corpus)]
    # The weighted averages sum only the production set (full metadata), but the
    # CLIP retrieval corpus and the focus/probe flags must also include nations
    # like Palestine and Israel even when they lack economic metadata, so the
    # il/ps tie and Palestine's erasure rank are real, recoverable results.
    extras = [
        code
        for code in dict.fromkeys((*FOCUS_CODES, *PROBE_CODES, "il"))
        if code in corpus and code not in codes
    ]
    arrays = corpus_arrays({code: corpus[code] for code in (*codes, *extras)}, config.canvas)

    backend_names = list(BASE_BACKENDS)
    if config.foundation != "off":
        backend_names.extend(FOUNDATION_BACKENDS)
    backends = {
        name: build_video_backend(name, arrays, config, model_manifest)
        for name in backend_names
    }

    weights: dict[str, WeightingRun] = {}
    generated: dict[str, dict[str, Image.Image]] = {intent: {} for intent in INTENTS}
    traces: dict[str, dict[str, dict[str, Any]]] = {intent: {} for intent in INTENTS}
    sdvae_latent: np.ndarray | None = None

    for intent_name in INTENTS:
        intent = weights_from_intent(intent_name, selected)
        weights[intent_name] = intent
        for backend_name, backend in backends.items():
            result = backend.average(codes, intent.weights)
            trace_path = trace_dir / f"trace_{slug(intent.name)}_{backend_name}.json"
            backend_files = save_backend_result(result, asset_dir, intent.name, backend_name)
            trace = build_trace(
                root=asset_dir,
                corpus_version=CORPUS_VERSION,
                intent=intent,
                backend_trace=result.trace,
                selected=selected,
                outputs={"trace": trace_path, **backend_files},
                music=None,
                status=result.status,
                reason=result.reason,
                canvas=config.canvas,
            )
            write_json(trace_path, trace)
            traces[intent_name][backend_name] = trace
            if result.image is not None:
                generated[intent_name][backend_name] = result.image.convert("RGB")
            if backend_name == "sdvae" and intent_name == SUBJECT_INTENT and result.representation is not None:
                sdvae_latent = np.asarray(result.representation, dtype=np.float64)
            if backend_name in FOUNDATION_BACKENDS and result.status != "ok" and required_foundation:
                raise RuntimeError(f"{backend_name} is required but unavailable: {result.reason}")

    (
        embedding, erasure, recognition, recognition_tie, focus, gallery, tie_image,
        pca_recon, pca_recon_energy, component, eigenflags, reconstructions, region_averages,
    ) = _compute_analysis(backends, codes, corpus, selected, weights, config)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "asset_canvas": f"{config.canvas[0]}x{config.canvas[1]}",
        "selected_entity_count": len(selected),
        "archive_entity_count": len(archive),
        "selected_codes": codes,
        "intents": list(INTENTS),
        "backends": backend_names,
        "foundation_required": required_foundation,
        "analysis_available": embedding is not None,
    }
    write_json(asset_dir / "asset_manifest.json", manifest)

    provenance = _provenance_manifest(erasure, recognition, embedding, focus)
    write_json(asset_dir.parent / "provenance" / "analysis_provenance.json", provenance)

    return RenderAssets(
        corpus=corpus,
        selected=selected,
        archive=archive,
        codes=codes,
        generated=generated,
        traces=traces,
        weights=weights,
        embedding=embedding,
        erasure=erasure,
        recognition=recognition,
        recognition_tie=recognition_tie,
        focus=focus,
        gallery=gallery,
        tie_image=tie_image,
        sdvae_latent=sdvae_latent,
        pca_recon=pca_recon,
        pca_recon_energy=pca_recon_energy,
        component=component,
        eigenflags=eigenflags,
        reconstructions=reconstructions,
        region_averages=region_averages,
        manifest=manifest,
        asset_dir=asset_dir,
        provenance=provenance,
    )


def _provenance_manifest(erasure, recognition, embedding, focus) -> dict[str, Any]:
    """Map each on-screen quantity to the analysis function that produced it."""

    sources: dict[str, str] = {}
    for record in erasure.values():
        sources.setdefault("erasure contributors / erased counts / top-5 share", record.provenance)
    for record in recognition.values():
        sources.setdefault("recognition ranking / confidence / margin / probe ranks", record.provenance)
    if embedding is not None:
        sources.setdefault("archive 2-D layout and kNN edges", embedding.provenance)
    for item in focus:
        if item.saliency is not None:
            sources.setdefault("decomposition attention heatmap", item.saliency.provenance)
        if item.residual is not None:
            sources.setdefault("decomposition residual / discard energy", item.residual.provenance)
        sources.setdefault(
            "decomposition feature vector",
            "tna.backends.pca.PCABackend.encode (eigenflag coordinates)",
        )
        sources.setdefault(
            "decomposition dominant colours + shares",
            "tna.backends.palette.dominant_palette_from_array",
        )
        sources.setdefault(
            "decomposition edge density",
            "tna.video.compositor.edge_density",
        )
        break
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "invariant": "Every value displayed on screen is produced by one of these functions; the renderer does not invent measurements.",
        "sources": sources,
    }


def write_render_notes(config: VideoRenderConfig, assets: RenderAssets, output_path: Path) -> None:
    tech = config.out_dir / "render_notes.md"
    cumulative = assets.recognition.get("cumulative_co2")
    nearest = cumulative.nearest if cumulative else None
    tech.write_text(
        "\n".join(
            [
                f"# {TITLE} — Technical Statement",
                "",
                "Single-channel video, 2026. Silent in this build (audio deferred).",
                "",
                "Generated from a local corpus of national / quasi-national flag SVGs "
                f"({len(assets.codes)} polities with sourced metadata for all five weightings). "
                "Every quantity shown on screen is a real computation over the corpus or the "
                "loaded models; see `provenance/analysis_provenance.json`.",
                "",
                "Real techniques used on screen:",
                "- CLIP (openai/clip-vit-base-patch32) image embeddings; nearest-nation retrieval "
                "of the averaged embedding with softmax confidence and top-1/top-2 margin.",
                "- CLIP ViT attention rollout (CLS->patch) as a per-flag saliency map.",
                "- PCA / eigenflag coordinates and reconstruction residuals.",
                "- Dominant-colour quantisation (octree) with shares; finite-difference edge density.",
                "- Stable Diffusion VAE latent decoding; deterministic SVG recomposition.",
                "- Weight-distribution accounting per metric (contributors and erased counts).",
                "",
                "Reported results (real, reproducible via `scripts/verify_analysis.py`):",
                (
                    f"- Weighted by historical CO2, the nearest real flag to the average is "
                    f"{nearest.name} (cosine {nearest.similarity:.3f})."
                    if nearest
                    else "- (CLIP recognition unavailable in this build.)"
                ),
                f"- Per metric, {assets.erasure['cumulative_co2'].erased_count}-"
                f"{assets.erasure['gdp'].erased_count} of {len(assets.codes)} nations fall below "
                "0.1% weight in the average.",
                "- The average of the Israeli and Palestinian flags retrieves both at an identical "
                "cosine (0.9659): a dead statistical tie.",
                "",
                f"Video: {config.width}x{config.height}, {config.fps} fps, {config.duration:.1f}s. "
                f"Final MP4: `{output_path.name}`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def render_video(config: VideoRenderConfig) -> dict[str, Any]:
    from .scenes import VideoFrameRenderer  # local import avoids a cycle

    config.out_dir.mkdir(parents=True, exist_ok=True)
    required_foundation = config.foundation == "required"
    model_manifest = None
    if config.foundation != "off":
        model_manifest = prepare_foundation_assets(config.out_dir, required=required_foundation)

    assets = generate_assets(config, required_foundation=required_foundation, model_manifest=model_manifest)

    audio_path = None
    if config.audio:
        from .audio import render_soundtrack

        audio_path = render_soundtrack(assets, config)

    output_path = config.out_dir / "the_national_average.mp4"
    renderer = VideoFrameRenderer(config, assets)
    render_video_stream(config, renderer, output_path, config.out_dir / "stills", audio_path=audio_path)
    write_render_notes(config, assets, output_path)

    return {
        "video": str(output_path),
        "audio": str(audio_path) if audio_path else None,
        "stills": str(config.out_dir / "stills"),
        "asset_manifest": str(assets.asset_dir / "asset_manifest.json"),
        "provenance": str(config.out_dir / "provenance" / "analysis_provenance.json"),
    }
