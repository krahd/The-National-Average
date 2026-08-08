# Moving-image work

**Status: active, work in progress (August 2026).**

The National Average includes an ongoing moving-image work generated from the same corpus, weighting operations, representation spaces, and analysis records as the rest of the project. It is not tied to a particular venue. The present source is a working baseline for continued artistic development and future exhibition or submission.

The renderer is part of the artwork's source. Rendered MP4 files are derivative outputs; losing a rendered master must never imply losing the system that produced it.

## Canonical source

The moving-image source is distributed across:

- `src/tna/video/pipeline.py`: asset generation, orchestration, provenance, and render configuration;
- `src/tna/video/scenes.py`: frame and scene construction;
- `src/tna/video/compositor.py`: frame composition and ffmpeg integration;
- `src/tna/video/effects.py`: image effects;
- `src/tna/video/audio.py`: data-driven soundtrack generation;
- `src/tna/video/foundation.py`: foundation-model preparation and model-manifest provenance;
- `src/tna/analysis/`: measurements and records displayed by the moving image;
- `scripts/render_video.py`: canonical command-line rendering entry point;
- `scripts/prepare_video_assets.py`: model-asset preparation;
- `presets/video-2026.json`: versioned render parameters.

None of these files should be treated as temporary application or submission material.

## Reproducing the current baseline

Install the project and development dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Install the foundation-model dependencies for a full production render:

```bash
pip install -e ".[foundation]"
python scripts/prepare_video_assets.py
```

Render a lower-resolution working version:

```bash
python scripts/render_video.py --preset preview
```

Render the preserved production baseline:

```bash
python scripts/render_video.py --preset production
```

The `production` configuration in `presets/video-2026.json` records the recovered three-minute baseline: 1920×1080, 24 fps, 180 seconds, deterministic seed `20260613`, with CLIP and Stable-Diffusion-VAE representations required. The `preview` configuration uses the same duration and seed at lower spatial and temporal resolution and permits unavailable foundation models.

Command-line options can override individual preset values for experiments without altering the preserved baseline.

## Epistemic invariant

The audiovisual system distinguishes analytical quantities from visual treatment. Values shown as measurements, rankings, residuals, saliency, embedding relations, contribution shares, or retrieval results must originate in computed analysis records. The renderer may animate, compose, grade, or distort imagery, but it must not fabricate values presented as analysis.

`src/tests/test_no_fabrication.py` guards this distinction. `src/tests/test_video_smoke.py` exercises the render chain at very low resolution so that changes to dependencies, imports, paths, scene code, or ffmpeg integration cannot silently make the artwork unrenderable.

## Outputs and preservation

Generated material belongs under `outputs/video/` and is not source. Each render writes its video, stills, generated analytical assets, and provenance records into its output directory. Model snapshot revisions are recorded in `provenance/model_manifest.json` when foundation assets are prepared.

Large rendered masters should not be committed to ordinary Git history. Meaningful finished or submitted versions should be attached to a GitHub Release or another archival store, while the exact source state and render preset remain tagged in Git. The baseline established here is tagged `video-2026-v1`; subsequent materially distinct versions should receive new tags rather than overwriting this one.

## Development direction

The moving-image work remains open. Its present renderer should be understood as material for further artistic development rather than a finished or venue-specific edition. Future changes may alter pacing, scene structure, sound, representational spaces, or visual language, but should preserve explicit provenance and the ability to reconstruct earlier tagged versions.
