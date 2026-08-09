# The National Average — NeurIPS 2026 implementation plan

**Goal:** produce a substantially stronger NeurIPS Creative AI edition while preserving the existing `video-2026-v1` baseline and the analytical/provenance invariant.

## 1. Artistic problem

The present renderer is technically strong but still carries traces of a system demonstration: dense HUD telemetry, a military/gaze visual language, frequent glitch treatment, and a long sequence of machine-processing stages. For the NeurIPS edition, computation should remain visible without becoming explanatory spectacle.

The edition should move from **source political symbols → incompatible representations → incompatible weightings → erasure → unresolved plurality**. The audience should first understand that the objects are flags, then perceive that different legitimate computational operations produce materially incompatible national averages. No final image should be allowed to appear as the answer.

## 2. Non-negotiable invariants

1. No displayed analytical quantity may be invented for visual effect.
2. Source data, weights, representation backends, checkpoint information, and deterministic seeds remain in provenance.
3. The preserved `production` preset and `video-2026-v1` semantics must not be silently changed.
4. NeurIPS-specific changes live in a distinct `neurips` preset and may branch stylistically without redefining the canonical baseline.
5. The submission video must be ≤180 seconds and <100 MB; the thumbnail must be <100 MB.

## 3. NeurIPS dramaturgy

### A. Sources

A quiet field/grid of national flags establishes the political material before analysis. Avoid reticles or target-designator aesthetics. The point is plurality, not surveillance.

### B. Representation

Move quickly to the fact that the same corpus can be represented as pixels, palettes, structured geometry, PCA, Stable Diffusion VAE latents, and CLIP embeddings. The most important visual event is disagreement among outputs, not the names of the algorithms.

### C. Weighting

Hold long enough on equal, population, GDP, annual CO2, and cumulative CO2 averages that the differences can be perceived. Labels remain sparse. Retrieval/recognition diagnostics are secondary.

### D. Erasure

Make contribution below the threshold visibly disappear. This is the political centre of the film: statistical inclusion can coexist with practical disappearance.

### E. Residual / unresolved ending

Show what representation discards, then end with several incompatible averages coexisting. Do not state or select a correct average. The final gesture should be plural rather than declarative.

## 4. Technical implementation

### New preset

Add `neurips` to `presets/video-2026.json`:

- 1920×1080;
- 24 fps;
- 180 seconds;
- deterministic seed `20260613`;
- foundation models required;
- audio enabled;
- provenance overlay disabled.

### New phase schedule

Use a NeurIPS-specific phase schedule that removes tutorial-like stages and creates longer perceptual holds:

1. `title`
2. `sources` — new quiet source-grid phase
3. `spaces`
4. `embed`
5. `weighting`
6. `average`
7. `erase`
8. `residual`
9. `coda`

The existing production schedule remains untouched.

### Visual treatment

For `neurips`:

- disable the dense global HUD;
- drastically reduce glitch amplitude;
- keep restrained grain/grade;
- eliminate the global 15% phase acceleration;
- make labels sparse and functional;
- implement a new final coda showing multiple incompatible averages rather than a single decoded result and explanatory slogan.

### Sound

Retain the existing data-dependent synthesis architecture but make the NeurIPS edition less continuously cinematic:

- reduce stochastic glitch density;
- reduce high-frequency shimmer;
- preserve phase-dependent tonal movement;
- retain stronger accents only for weighting/erasure transitions;
- leave space around the final unresolved images.

## 5. Validation

Run:

1. Python compile check.
2. Existing test suite.
3. Low-resolution short smoke render using `neurips` preset overrides.
4. Full foundation-asset preparation.
5. Full NeurIPS render.
6. `ffprobe` duration/audio/video validation.
7. Package using `scripts/package-neurips-2026.sh`.
8. Inspect a generated contact sheet from beginning/middle/end and manually select the strongest thumbnail.
9. Confirm packaged preview <100 MB.

## 6. Submission package

Produce:

- `the-national-average-preview.mp4`;
- `the-national-average-thumbnail.png`;
- `the-national-average-neurips-2026.pdf`;
- machine-readable package manifest recording source commit, preset, seed, duration, checksums, and files.

The repository remains the canonical source; submission files are derivatives.
