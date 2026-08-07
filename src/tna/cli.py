"""Command-line interface for the representation ladder prototype."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

from PIL import Image

from .backends.base import BackendResult
from .backends.pca import PCABackend
from .backends.registry import AVAILABLE_BACKENDS, build_backend
from .backends.svg import write_svg
from .data import DATA_DIR, DEFAULT_CANVAS, ROOT_DIR, corpus_arrays, load_corpus, parse_canvas, rasterize_flag, selected_items
from .figures import comparison_figure, eigenflag_figure, embedding_diagnostic, ladder_figure, strip_figure
from .music import averaged_music_profile, render_wav, write_score_trace
from .trace import build_trace, write_json
from .weights import INTENT_ALIASES, WeightingRun, parse_csv, parse_manual_weights, weights_from_intent


DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs"
CORPUS_VERSION = "lipis-flag-icons-main-4x3+metadata-v2"


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate representation-ladder critical averages.")
    parser.add_argument("--entities", default="fr,uy,ps",
                        help="Comma-separated entity codes. Default: fr,uy,ps")
    parser.add_argument("--intents", default="equal,population",
                        help="Comma-separated weighting intents.")
    parser.add_argument("--backends", default="pixel,palette,pca,svg",
                        help="Comma-separated visual backends.")
    parser.add_argument("--canvas", type=parse_canvas, default=DEFAULT_CANVAS,
                        help="Canvas WIDTHxHEIGHT. Default: 96x72")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory. Default: outputs/")
    parser.add_argument("--seed", type=int, default=0,
                        help="Deterministic seed recorded in traces. Default: 0")
    parser.add_argument("--manual-weights", default=None, help="Manual code=value weights.")
    parser.add_argument("--palette-size", type=int, default=5,
                        help="Palette/spec size for palette and SVG backends.")
    parser.add_argument("--pca-components", type=int, default=32,
                        help="PCA components for eigenflag backend.")
    parser.add_argument("--audio-duration", type=float, default=8.0,
                        help="Symbolic WAV duration in seconds.")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Symbolic WAV sample rate.")
    parser.add_argument("--train-vae", action="store_true",
                        help="Train/use optional PyTorch VAE backend if dependencies exist.")
    parser.add_argument("--vae-epochs", type=int, default=1000,
                        help="Optional VAE training epochs (full-batch). Default: 1000")
    parser.add_argument("--include-unavailable", action="store_true",
                        help="Emit traces for unavailable optional backends.")
    parser.add_argument("--svg-renderer", choices=["deterministic", "llm"], default="deterministic")
    parser.add_argument("--alpha-values", default="0,0.25,0.5,1,2",
                        help="Population-alpha strip values.")
    parser.add_argument("--list-entities", action="store_true")
    parser.add_argument("--list-intents", action="store_true")
    parser.add_argument("--list-backends", action="store_true")
    return parser


def output_dirs(root: Path) -> dict[str, Path]:
    """Create the run directory layout used by generated artefacts."""

    dirs = {
        "generated": root / "generated",
        "figures": root / "figures",
        "traces": root / "traces",
        "music": root / "music",
        "bundles": root / "bundles",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_backend_result(result: BackendResult, dirs: dict[str, Path], intent_name: str, backend_name: str) -> dict[str, Path]:
    """Persist backend artefacts and return paths for provenance traces."""

    files: dict[str, Path] = {}
    if result.image is not None:
        image_path = dirs["generated"] / f"{slug(intent_name)}_{backend_name}.png"
        result.image.save(image_path)
        files["image"] = image_path
    if "svg_text" in result.files:
        svg_path = dirs["generated"] / f"{slug(intent_name)}_{backend_name}.svg"
        write_svg(svg_path, result.files["svg_text"])
        files["svg"] = svg_path
    return files


def write_bundle(bundle_path: Path, trace_path: Path, files: dict[str, Path]) -> None:
    """Write the default shareable export: artefact and provenance together."""

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(trace_path, arcname="trace.json")
        for role, item_path in sorted(files.items()):
            if item_path.exists():
                archive.write(item_path, arcname=f"{role}{item_path.suffix}")


def write_music(intent: WeightingRun, selected, dirs: dict[str, Path], duration: float, sample_rate: int) -> tuple[dict[str, object], dict[str, Path]]:
    """Generate the symbolic score and WAV once per weighting intent."""

    music = averaged_music_profile(selected, intent.weights, duration)
    score_path = dirs["music"] / f"score_{slug(intent.name)}.txt"
    wav_path = dirs["music"] / f"anthem_{slug(intent.name)}.wav"
    write_score_trace(score_path, intent.name, intent.weights, selected, music)
    render_wav(music["notes"], wav_path, sample_rate)
    return music, {"score": score_path, "wav": wav_path}


def unavailable_trace(
    args,
    dirs: dict[str, Path],
    intent: WeightingRun,
    backend,
    selected,
    music,
    music_files: dict[str, Path],
) -> dict[str, object]:
    """Write an auditable trace for an optional backend that cannot run."""

    trace_path = dirs["traces"] / f"trace_{slug(intent.name)}_{backend.name}.json"
    trace = build_trace(
        root=args.out_dir,
        corpus_version=CORPUS_VERSION,
        intent=intent,
        backend_trace=backend.average([polity.code for polity in selected], intent.weights).trace,
        selected=selected,
        outputs={"trace": trace_path, **music_files},
        music=music,
        status="unavailable",
        reason=backend.reason,
        canvas=args.canvas,
    )
    write_json(trace_path, trace)
    return trace


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    """Execute one CLI run and return the per-output trace payloads."""

    corpus = load_corpus(data_dir=DATA_DIR)
    if args.list_entities:
        for code, polity in corpus.items():
            print(f"{code:8s} {polity.name:24s} {polity.metadata_quality}")
        return []
    if args.list_intents:
        print("equal")
        print("manual")
        print("population_alpha:<alpha>")
        for alias in sorted(INTENT_ALIASES):
            print(alias)
        return []
    if args.list_backends:
        for backend in AVAILABLE_BACKENDS:
            print(backend)
        return []

    entity_codes = parse_csv(args.entities)
    selected = selected_items(corpus, entity_codes)
    arrays = corpus_arrays(corpus, args.canvas)
    # All visual backends receive the same rasterised corpus. Backends that
    # need learned spaces fit from this shared corpus rather than from only the
    # selected flags, which keeps embeddings comparable across selections.
    source_images = [(polity.name, rasterize_flag(polity, args.canvas)) for polity in selected]

    intents = parse_csv(args.intents)
    if args.manual_weights and "manual" not in intents:
        intents.append("manual")
    manual = parse_manual_weights(args.manual_weights, selected) if args.manual_weights else None
    backend_names = parse_csv(args.backends)
    dirs = output_dirs(args.out_dir)

    traces: list[dict[str, object]] = []
    all_averages: dict[str, dict[str, Image.Image]] = {}
    pca_backend: PCABackend | None = None

    # Build each backend once (fits PCA / trains the VAE a single time), then
    # reuse it across every weighting intent.
    backends = {
        backend_name: build_backend(
            backend_name,
            arrays,
            seed=args.seed,
            palette_size=args.palette_size,
            pca_components=args.pca_components,
            svg_renderer=args.svg_renderer,
            train_vae=(args.train_vae and backend_name == "vae"),
            vae_epochs=args.vae_epochs,
        )
        for backend_name in backend_names
    }

    for intent_name in intents:
        intent = weights_from_intent(intent_name, selected, manual)
        music, music_files = write_music(
            intent, selected, dirs, args.audio_duration, args.sample_rate)
        intent_averages: dict[str, Image.Image] = {}

        for backend_name in backend_names:
            backend = backends[backend_name]
            result = backend.average(entity_codes, intent.weights)
            if result.status == "unavailable":
                if args.include_unavailable:
                    traces.append(unavailable_trace(args, dirs, intent,
                                  backend, selected, music, music_files))
                continue

            backend_files = save_backend_result(result, dirs, intent.name, backend.name)
            trace_path = dirs["traces"] / f"trace_{slug(intent.name)}_{backend.name}.json"
            bundle_path = dirs["bundles"] / f"{slug(intent.name)}_{backend.name}.zip"
            outputs = {"trace": trace_path, "bundle": bundle_path, **backend_files, **music_files}
            trace = build_trace(
                root=args.out_dir,
                corpus_version=CORPUS_VERSION,
                intent=intent,
                backend_trace=result.trace,
                selected=selected,
                outputs=outputs,
                music=music,
                canvas=args.canvas,
            )
            write_json(trace_path, trace)
            write_bundle(bundle_path, trace_path, {**backend_files, **music_files})
            traces.append(trace)
            if result.image is not None:
                intent_averages[backend.name] = result.image
            if isinstance(backend, PCABackend):
                pca_backend = backend

        all_averages[intent.name] = intent_averages
        ladder_figure(source_images, intent_averages, intent.name,
                      dirs["figures"] / f"ladder_{slug(intent.name)}.png")

    populated = {name: images for name, images in all_averages.items() if images}
    if len(populated) > 1:
        weight_labels = {}
        for intent_name in populated:
            intent = weights_from_intent(intent_name, selected, manual)
            shares = ", ".join(f"{value:.2f}" for value in intent.weights.values())
            weight_labels[intent.name] = f"{intent.name}\n({shares})"
        comparison_figure(source_images, populated, weight_labels,
                          dirs["figures"] / "comparison_all_intents.png")

    if pca_backend is not None:
        eigenflag_figure(pca_backend.eigenflag_images(), dirs["figures"] / "eigenflags_pca.png")
        points = pca_backend.embedding_points(list(arrays))
        avg_points = {}
        for intent_name in intents:
            intent = weights_from_intent(intent_name, selected, manual)
            z = pca_backend.average(entity_codes, intent.weights).representation
            avg_points[intent.name] = (float(z[0]), float(z[1] if len(z) > 1 else 0.0))
        embedding_diagnostic(points, avg_points, dirs["figures"] / "embedding_pca.png")

    alpha_values = [float(value) for value in parse_csv(args.alpha_values)]
    alpha_backends = [name for name in backend_names if name in {
        "pixel", "palette", "pca", "svg", "vae", "sdvae"}]
    for backend_name in alpha_backends:
        backend = backends[backend_name]
        strip_images = {}
        for alpha in alpha_values:
            intent = weights_from_intent(f"population_alpha:{alpha}", selected, manual)
            result = backend.average(entity_codes, intent.weights)
            if result.status == "ok" and result.image is not None:
                strip_images[f"a={alpha:g}"] = result.image

        if strip_images:
            strip_figure(
                strip_images,
                f"Population-alpha morph in {backend_name} space",
                dirs["figures"] / f"alpha_morph_{backend_name}.png",
            )

    summary = {"corpus_version": CORPUS_VERSION, "runs": traces}
    write_json(args.out_dir / "run_summary.json", summary)
    write_summary_md(args.out_dir / "run_summary.md", traces)
    print(f"Wrote {len(traces)} trace(s) to {args.out_dir}")
    return traces


def write_summary_md(path: Path, traces: list[dict[str, object]]) -> None:
    """Write a lightweight human index for the JSON traces."""

    lines = [
        "# Representation Ladder Run Summary",
        "",
        "| Intent | Backend | Status | Image | Trace | Bundle |",
        "|---|---|---|---|---|---|",
    ]
    for trace in traces:
        outputs = trace["outputs"]
        lines.append(
            f"| {trace['intent']} | {trace['backend']['backend']} | {trace['status']} | "
            f"{outputs.get('image', '')} | {outputs.get('trace', '')} | {outputs.get('bundle', '')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
