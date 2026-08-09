# The National Average — NeurIPS 2026 implementation plan

**Goal:** produce a substantially stronger Creative AI edition while preserving the existing `video-2026-v1` baseline and the analytical/provenance invariant.

**Implementation state:** the planned NeurIPS recut plus two additional expansion/test iterations are implemented. Remaining work is full render, visual/sound inspection, evidence-based correction, and submission packaging.

## 1. Artistic problem

The preserved renderer is technically strong but carries the aesthetics of a system demonstration: dense HUD telemetry, military/gaze language, frequent glitch treatment, and a long chain of machine-processing stages. The NeurIPS edition removes that layer without hiding computation.

The work now moves through:

**political corpus → representation → weighting distribution → representation × weighting → erasure → concrete equal pair → unresolved plurality**.

The central artistic operation is no longer “show the pipeline.” It is to make the consequences of computational choices remain simultaneously visible.

## 2. Non-negotiable invariants

1. No displayed analytical quantity is invented for visual effect.
2. Source data, weights, representation backends, checkpoint information, deterministic seeds, and analysis provenance remain recoverable.
3. The preserved `production` preset and `video-2026-v1` semantics do not change silently.
4. NeurIPS-specific changes live in a distinct `neurips` preset/renderer.
5. The submission target is **176 seconds**, leaving margin under the ≤180-second requirement.
6. Preview and thumbnail must each remain `<100 MB`.
7. Sonification can map real measurements but must never be described as a measurement itself.

## 3. Final NeurIPS dramaturgy

1. `title`
2. `sources` — quiet source field establishing the political corpus
3. `spaces` — incompatible representational outputs
4. `embed` — learned representational field, without model fetishism
5. `distribution` — real concentration/effective-contributor measures for five weighting regimes
6. `weighting` — perceptual comparison of weighting outputs
7. `matrix` — representation × weighting as orthogonal operations
8. `thresholds` — progressive practical disappearance under a changing contribution threshold
9. `erase` — project-native erasure view
10. `pair` — equal Israel/Palestine pixel average and CLIP retrieval relation
11. `average`
12. `residual` — what a representation discards
13. `coda` — same corpus / different weightings, no privileged answer

The original tutorial-heavy `tokenize`, `attend`, and `name` phases remain in the baseline work but are not part of the NeurIPS edition.

## 4. Additional iteration 1 — distribution

Implemented in `src/tna/video/neurips_metrics.py` and the NeurIPS renderer.

For each existing weighting record:

- normalised Shannon entropy;
- entropy effective contributor count `exp(H)`;
- maximum normalised contribution;
- number of contributors below an arbitrary threshold.

The threshold sweep is explicitly a visual operation over a chosen threshold. It does not claim that 0.1%, 1%, or any other cut has intrinsic political authority. Its purpose is to make the dependence of disappearance on the chosen criterion perceptible.

The soundtrack uses the same records with deliberately shallow mappings: concentration narrows the spectral field; threshold/erasure counts change sparse-event density.

## 5. Additional iteration 2 — orthogonality and political pair

### Representation × weighting matrix

The matrix uses actual `RenderAssets.generated` outputs. Rows correspond to available representation backends; columns correspond to weighting intentions. It prevents the film from treating representation choice and weighting choice as one operation.

### Equal Israel/Palestine pair

The scene uses:

- the actual archived Israeli and Palestinian flag images;
- `RenderAssets.tie_image`, the real 50/50 pixel average;
- `RenderAssets.recognition_tie`, the real CLIP retrieval/ranking/margin calculated by the existing analysis pipeline.

No explanatory political caption is added. The work presents the operation and its numerical relation.

The soundtrack maps the real retrieval margin to the interval between two tones. Near-zero margin becomes near-unison. This mapping is documented as sound design.

## 6. Validation already implemented

GitHub Actions lightweight tests cover:

- NeurIPS schedule continuity;
- 176-second master target;
- presence/order of the two new iteration families;
- uniform-distribution entropy/effective count;
- concentration behaviour;
- threshold normalisation;
- metric scale invariance.

The expanded NeurIPS test workflow passes.

GPU/foundation-model execution is intentionally outside CI.

## 7. Remaining render/inspection loop

### Pass A — smoke

```bash
python scripts/render_neurips_2026.py \
  --width 640 --height 360 --fps 6 --duration 24 \
  --foundation auto --no-audio \
  --out-dir outputs/video/neurips-smoke
```

Inspect only structural failures: layout collisions, unreadable matrix, source-grid density, invalid phase assumptions.

### Pass B — full

```bash
python scripts/prepare_video_assets.py
python scripts/render_neurips_2026.py
pytest -q src/tests
bash scripts/package-neurips-2026.sh
```

Then inspect complete preview and contact sheet. Do not change the work because a different aesthetic could also be imagined; change it only for observed failures.

## 8. Final visual questions

- Does the matrix read as two independent operations rather than as another dashboard?
- Does the distribution scene produce perceptual concentration, not merely numbers?
- Does the threshold sweep reveal disappearance without moralising the arbitrary cut?
- Is the Israel/Palestine scene formally severe enough to avoid sentimentality or sensationalism?
- Does the sound remain subordinate to the visual/procedural argument?
- Does the coda leave incompatible averages unresolved?

## 9. Submission package

Produce:

- `the-national-average-preview.mp4`;
- `the-national-average-thumbnail.png`;
- `the-national-average-contact-sheet.jpg`;
- `the-national-average-neurips-2026.pdf`;
- machine-readable package manifest with source, seed, duration, and checksums.

The repository remains canonical; submission files are derivatives.
