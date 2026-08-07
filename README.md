# The National Average

**The National Average** is an inspectable implementation of **critical averaging**: selected national symbols are encoded into several representation spaces, combined with the same weighting operation, and decoded or reported with machine-readable provenance.

The central operation is deliberately simple:

```text
source flags -> encode -> z_avg = sum(w_i * z_i) -> decode or report
```

The project asks what changes when the entities and weights are held fixed but the representation changes, and what changes when the representation is held fixed but the weighting criterion changes.

This repository is the public implementation and reproducibility companion for the IBERAMIA work on critical averaging. Manuscript sources, submission files, reviews, and publication-version artefacts are intentionally maintained separately and are not duplicated here.

## Installation

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
```

Optional backends:

```bash
pip install -e ".[ml]"          # local PyTorch VAE
pip install -e ".[foundation]"  # Stable Diffusion VAE and CLIP
```

## Quick start

Run the default worked example:

```bash
python -m tna.cli
```

Run several weighting criteria across the base representation ladder:

```bash
python -m tna.cli \
  --entities fr,uy,ps \
  --intents equal,population,area,gdp,co2,cumulative_co2 \
  --backends pixel,palette,pca,svg \
  --out-dir outputs/tutorial
```

Manual weights are explicit:

```bash
python -m tna.cli \
  --entities fr,uy,ps \
  --intents manual \
  --manual-weights fr=2,uy=1,ps=1 \
  --backends pixel,palette,pca,svg \
  --out-dir outputs/manual-example
```

The original entry point remains supported:

```bash
python src/critical_averaging.py --backends pixel,palette,pca
```

Discover the current corpus and implementation from the CLI itself:

```bash
python -m tna.cli --list-entities
python -m tna.cli --list-intents
python -m tna.cli --list-backends
```

## Outputs and provenance

Generated files are written under `outputs/` by default and are ignored by Git.

```text
outputs/
  generated/          PNG and SVG backend artefacts
  figures/            ladder comparisons, PCA diagnostics, alpha morphs
  music/              symbolic score traces and WAV files
  traces/             per-output JSON provenance records
  bundles/            shareable ZIPs containing an artefact and its trace
  run_summary.json    machine-readable run index
  run_summary.md      human-readable run index
```

Each successful visual result has a JSON trace recording the selected entities, raw and normalised weights, representation space, backend, learned/non-learned status, model/checkpoint information where relevant, deterministic seed, synthesis method, symbolic-music parameters, and output paths. Optional backends can emit explicit `unavailable` traces instead of silently substituting another method.

The default shareable unit is a ZIP in `outputs/bundles/` containing the generated artefact, its provenance trace, and associated symbolic-music outputs.

## Representation ladder

| Backend | Representation | Learned | Output | Status |
|---|---|---:|---|---|
| `pixel` | Per-pixel RGB tensor | No | Image | Implemented |
| `palette` | Dominant colours and area shares | No | Image | Implemented |
| `svg` | Structured SVG-like program specification | No | SVG + image | Implemented |
| `pca` | Linear eigenflag coordinates | Yes, local linear model | Image | Implemented |
| `vae` | Nonlinear latent vector | Yes, locally trained | Image | Optional `ml` extra |
| `sdvae` | Stable Diffusion VAE latent | Yes, pretrained | Image | Optional `foundation` extra |
| `clip` | CLIP image embedding | Yes, pretrained | Nearest-corpus retrieval | Optional `foundation` extra |
| `concept` | Learned concept-token embedding | Yes | Depends on renderer | Scaffolded |
| `music_vae` | Symbolic music latent | Yes | Music | Scaffolded; corpus not supplied |

The shared backend contract is defined in `src/tna/backends/base.py`. The project treats learnedness as a property of a representation, not as a quality ranking.

## Weighting intents

Raw weights express a criterion; normalised weights sum to one and are what the averaging operation uses.

| Intent | Metadata / operation |
|---|---|
| `equal` | equal raw weight for every selected entity |
| `population` | `population_millions` |
| `area` | `area_km2` |
| `recognition_year` | `recognition_year` with confidence/source guards |
| `gdp` | `gdp_current_usd` |
| `co2` / `carbon` | annual `co2_mt` |
| `consumption_co2` | `consumption_co2_mt` |
| `cumulative_co2` | `cumulative_co2_mt` |
| `ghg` | `total_ghg_mt` |
| `energy` | `primary_energy_consumption_twh` |
| `population_alpha:<a>` | `population_millions ** a`; `a=0` is equal weighting |
| `manual` | explicit `code=value` pairs |

The implementation refuses a sourced-metadata weighting when any selected entity lacks a usable value or only has a fallback placeholder. It does not manufacture missing measurements.

## Data

`data/metadata.csv` currently contains 271 rows and 52 columns. It combines public-source metadata used for entity description and weighting. `data/metadata_coverage.json` records field coverage and quality categories, while `data/metadata_sources.md` documents the rebuild procedure, source roles, units, and field-level caveats.

The corpus also contains the SVG flag assets used for rasterisation. The default analytical nation corpus filters supranational, subnational, placeholder, and other out-of-scope codes in `src/tna/data.py`.

The metadata and flag assets are third-party or derivative materials and are **not relicensed under the project MIT license**. See `LICENSE-data.md`, `LICENSE-flags.txt`, and `data/metadata_sources.md`.

## Analysis and audiovisual pipeline

The repository also contains the analysis and audiovisual implementation used to turn the same project data into exhibition/video outputs. These modules are part of the project implementation, not publication manuscripts.

| Path | Purpose |
|---|---|
| `src/tna/analysis/components.py` | connected-component and region analysis |
| `src/tna/analysis/embedding.py` | embedding geometry measurements |
| `src/tna/analysis/erasure.py` | contribution/erasure records |
| `src/tna/analysis/recognition.py` | corpus-retrieval records |
| `src/tna/analysis/residual.py` | reconstruction residuals |
| `src/tna/analysis/saliency.py` | saliency measurements |
| `src/tna/video/pipeline.py` | audiovisual orchestration and provenance manifest |
| `src/tna/video/scenes.py` | scene rendering from analysis records |
| `src/tna/video/compositor.py` | frame composition and ffmpeg integration |
| `src/tna/video/audio.py` | data-driven audio rendering |
| `scripts/prepare_eccv_assets.py` | retained production entry point for foundation assets |
| `scripts/render_eccv_video.py` | retained production entry point for the video artwork |
| `scripts/verify_analysis.py` | cross-check analysis records and invariants |

The video tests enforce the project invariant that displayed analytical quantities originate in computed records rather than procedural fabrication.

## Source map

| Path | Purpose |
|---|---|
| `src/tna/cli.py` | CLI orchestration, output bundles, summaries, and figures |
| `src/tna/data.py` | metadata loading, corpus filtering, flag rasterisation |
| `src/tna/weights.py` | weighting intents, source guards, normalisation |
| `src/tna/trace.py` | JSON provenance records |
| `src/tna/figures.py` | comparison and diagnostic figures |
| `src/tna/backends/` | representation-space implementations and registry |
| `src/tna/music/` | transparent symbolic music and optional scaffold |
| `src/tna/analysis/` | computed machine-vision analysis records |
| `src/tna/video/` | audiovisual production pipeline |
| `scripts/build_metadata.py` | deterministic metadata rebuild from source snapshots |
| `src/critical_averaging.py` | backwards-compatible wrapper around `tna.cli` |
| `src/tests/` | smoke and provenance/fabrication regression tests |

## Validation

Compile-check all tracked Python modules:

```bash
python -m py_compile \
  src/critical_averaging.py \
  src/tna/*.py \
  src/tna/backends/*.py \
  src/tna/music/*.py \
  src/tna/analysis/*.py \
  src/tna/video/*.py \
  scripts/*.py
```

Run the tests:

```bash
pytest -q src/tests
```

Run the base smoke path independently:

```bash
python -m tna.cli \
  --entities fr,uy,ps \
  --intents equal,population \
  --backends pixel,palette,pca,svg \
  --out-dir outputs/smoke-base
```

## Generated/local files

Ignored local material includes virtual environments, `.env`, caches, raster caches, model checkpoints, generated outputs, and local `ignore/` content. Nothing under those paths is required to understand the tracked source tree.

## Licensing

Original project source code and documentation are licensed under the MIT License; see `LICENSE`.

Flag SVGs originate from `lipis/flag-icons` and retain that project's MIT notice; see `LICENSE-flags.txt`. The enriched metadata database contains material under several source licenses and is distributed subject to the notices in `LICENSE-data.md` and the provenance in `data/metadata_sources.md`.
