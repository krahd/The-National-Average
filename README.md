# The National Average

**The National Average** is a research project and artwork by Tomas Laurenzo about the politics of averaging, statistical representation, and generative systems.

The project starts from a simple proposition: an average is not discovered; it is manufactured. Before anything can be averaged, a system has already decided what enters the operation, how each object is represented, how much each contributes, how differences are normalised, and how the result is synthesised. These choices are often hidden behind the apparent neutrality of a final number or image.

The National Average makes them explicit. National flags are encoded in different representation spaces and combined under different weighting criteria. The same averaging operation can therefore produce materially different results depending on whether a flag is treated as pixels, colours, structured geometry, a point in a locally learned latent space, or an embedding in a foundation model. Population, area, GDP, emissions, and other weighting criteria alter the result again.

National symbols are deliberately difficult material for this operation. Flags carry histories of sovereignty, identity, colonialism, occupation, and conflict; they are designed not to be interchangeable. Averaging them stages the violence and absurdity of commensuration in a form that remains immediately legible. The project does not seek a correct average flag. The divergence between its possible averages is the work.

At the centre of the project is **critical averaging**, a method that decomposes aggregation into five sites of decision:

1. **Selection**: what is included and excluded.
2. **Representation**: in what space the selected objects become comparable.
3. **Weighting**: with what relative force each object contributes.
4. **Normalisation**: how unlike objects are coerced into a common frame.
5. **Synthesis**: how the resulting representation is rendered, reconstructed, or retrieved.

The implementation records these operations in machine-readable provenance traces so that the apparatus remains inspectable as the representational space becomes less transparent.

## Publication

The project is discussed in:

> Tomas Laurenzo. **“Critical Averaging and the Politics of Statistical Representation.”** To appear in the proceedings of IBERAMIA 2026, *Advances in Artificial Intelligence*, Springer Lecture Notes in Artificial Intelligence (LNAI), 2026.

IBERAMIA 2026: https://www.iberamia.org/iberamia/iberamia2026/

The paper articulates the method and its critical argument; this repository is the working research and artistic system: code, data, representation backends, experiments, provenance mechanisms, and audiovisual production tools.

## The representation ladder

Every visual backend performs the same basic operation:

```text
source flags -> encode -> z_avg = sum(w_i * z_i) -> decode or report
```

What changes is the space in which `z` exists.

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

The ladder is ordered from direct appearance, through hand-specified reductions and locally learned spaces, to externally learned foundation-model representations. Learnedness is not treated as a quality ranking. It identifies where the coordinate system comes from, and therefore which assumptions enter the averaging operation.

## Weighting

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

A sourced-metadata weighting is refused when any selected entity lacks a usable value or only has a fallback placeholder. Missing measurements are not silently fabricated.

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

## Running the project

Run the default France–Uruguay–Palestine example:

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

Use explicit manual weights:

```bash
python -m tna.cli \
  --entities fr,uy,ps \
  --intents manual \
  --manual-weights fr=2,uy=1,ps=1 \
  --backends pixel,palette,pca,svg \
  --out-dir outputs/manual-example
```

Inspect the current corpus and available operations:

```bash
python -m tna.cli --list-entities
python -m tna.cli --list-intents
python -m tna.cli --list-backends
```

The earlier entry point remains available:

```bash
python src/critical_averaging.py --backends pixel,palette,pca
```


Render the moving-image work from its versioned preset:

```bash
python scripts/render_video.py --preset preview
python scripts/render_video.py --preset production
```

## Outputs and provenance

Generated material is written under `outputs/` by default:

```text
outputs/
  generated/          PNG and SVG backend artefacts
  figures/            ladder comparisons, PCA diagnostics, alpha morphs
  music/              symbolic score traces and WAV files
  traces/             per-output JSON provenance records
  bundles/            ZIPs containing an artefact and its trace
  run_summary.json    machine-readable run index
  run_summary.md      human-readable run index
```

Each result carries a JSON trace recording the selected entities, raw and normalised weights, representation space, backend, model or checkpoint information where relevant, deterministic seed, synthesis method, symbolic-music parameters, and output paths. Optional backends can emit explicit `unavailable` traces rather than substituting another method.

The per-output bundle joins the generated or retrieved artefact with its provenance trace and associated symbolic-music outputs. This is important conceptually as well as technically: the image is not treated as separable from the operations that produced it.

## Data

`data/metadata.csv` contains 271 rows and 52 columns assembled from public data sources. The metadata supports weighting by demographic, geographical, economic, environmental, and historical variables. `data/metadata_coverage.json` records coverage and quality categories; `data/metadata_sources.md` documents provenance, rebuild procedures, units, and field-level caveats.

The flag corpus contains country and territory entries. Supranational bodies, administrative subdivisions, placeholders, and other out-of-scope codes are filtered from the analytical corpus in `src/tna/data.py`. Corpus membership is an operational taxonomy for the project, not a claim about sovereign status.

Data provenance is part of the research question. A weighting criterion only appears neutral if the origin, units, exclusions, temporal frame, and missing values of the underlying data are ignored.

The metadata and flag assets retain their source licences; see `LICENSE-data.md`, `LICENSE-flags.txt`, and `data/metadata_sources.md`.

## Audiovisual work

**Status: active, work in progress (August 2026).** The moving-image work is an ongoing manifestation of The National Average and is being developed independently of any particular venue. Its renderer and versioned presets are canonical artwork source, not disposable production infrastructure. See [`docs/video.md`](docs/video.md) for reproduction and preservation details.

The repository also contains the audiovisual system used to develop moving-image manifestations of the project. These modules transform the same corpus, weighting operations, learned representations, and analysis records into video and sound.

The video pipeline follows a strict invariant: analytical quantities shown on screen originate in computed records. Values, rankings, residuals, saliency measurements, and embedding relations are not invented for visual effect.

| Path | Purpose |
|---|---|
| `src/tna/analysis/components.py` | connected-component and region analysis |
| `src/tna/analysis/embedding.py` | embedding geometry measurements |
| `src/tna/analysis/erasure.py` | contribution and erasure records |
| `src/tna/analysis/recognition.py` | corpus-retrieval records |
| `src/tna/analysis/residual.py` | reconstruction residuals |
| `src/tna/analysis/saliency.py` | saliency measurements |
| `src/tna/video/pipeline.py` | audiovisual orchestration and provenance |
| `src/tna/video/scenes.py` | scene rendering from analysis records |
| `src/tna/video/compositor.py` | frame composition and ffmpeg integration |
| `src/tna/video/audio.py` | data-driven audio rendering |
| `scripts/prepare_video_assets.py` | foundation-model asset preparation |
| `scripts/render_video.py` | video rendering entry point |
| `scripts/verify_analysis.py` | independent checks of analysis records and invariants |
| `presets/video-2026.json` | preserved preview and production render settings |
| `docs/video.md` | moving-image status, reproduction, provenance, and preservation policy |

A secondary symbolic-music pipeline applies the same logic to hand-authored anthem-like profiles, including tempo, pitch, mode, metre, motifs, and durations.

## Repository structure

| Path | Purpose |
|---|---|
| `src/tna/cli.py` | CLI orchestration, output bundles, summaries, and figures |
| `src/tna/data.py` | metadata loading, corpus filtering, flag rasterisation |
| `src/tna/weights.py` | weighting intents, source guards, normalisation |
| `src/tna/trace.py` | provenance records |
| `src/tna/figures.py` | comparison and diagnostic figures |
| `src/tna/backends/` | representation-space implementations and registry |
| `src/tna/music/` | symbolic music system and optional scaffold |
| `src/tna/analysis/` | machine-vision analysis records |
| `src/tna/video/` | moving-image source: orchestration, scenes, composition, effects, and sound |
| `scripts/render_video.py` | canonical moving-image rendering entry point |
| `presets/video-2026.json` | versioned moving-image render settings |
| `docs/video.md` | moving-image documentation and preservation policy |
| `scripts/build_metadata.py` | metadata rebuild from source snapshots |
| `src/critical_averaging.py` | backwards-compatible CLI wrapper |
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

Run a base smoke path:

```bash
python -m tna.cli \
  --entities fr,uy,ps \
  --intents equal,population \
  --backends pixel,palette,pca,svg \
  --out-dir outputs/smoke-base
```

## Licensing

Original source code and documentation are licensed under the MIT License; see `LICENSE`.

Flag SVGs originate from `lipis/flag-icons` and retain that project's MIT notice; see `LICENSE-flags.txt`. The metadata database contains material under several source licences and is distributed subject to the notices in `LICENSE-data.md` and the provenance information in `data/metadata_sources.md`.
