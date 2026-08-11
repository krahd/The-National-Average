# NeurIPS 2026 execution status - The National Average

**State:** NeurIPS artwork submission package ready for OpenReview upload; canonical full foundation-model render remains available as a later replacement only if completed before upload.  
**Date:** 11 August 2026.

## Preserved boundary

The canonical `video-2026-v1` / production renderer remains untouched. The NeurIPS edition is a separate 1920×1080 / 24 fps / **176-second** derivative with foundation models required, giving explicit margin under the three-minute upload ceiling.

The opportunity-specific package is canonical in:

```text
krahd/professional-opportunities/
  artistic-submissions/2026-08-10_neurips-creative-ai_the-national-average/
```

That package now contains the final description source, compliance record, media manifest and exact checksums. The delivery binaries have also been generated locally for portal upload.

## Submission-ready derivative - 11 August

Because the repository contains the full NeurIPS renderer but no committed rendered master, a separate programmatic submission-preview derivative was generated and audited for the portal package. It is explicitly not represented as the canonical full foundation-model render.

Current upload assets:

- official NeurIPS 2026 `creativeai` description PDF: **2 pages**, US Letter, embedded Type 1 fonts, official author kit, successful GitHub Actions compile and page-by-page visual preflight;
- thumbnail PNG: **161,855 bytes**;
- preview MP4: **176.000 seconds**, H.264/AAC, 1280×720, 24 fps, **5,676,183 bytes**;
- exact SHA-256 values are recorded in the opportunity package.

The submission preview displays only measurements that are either recomputed directly in that derivative or recorded in the accepted IBERAMIA paper. It does not fabricate foundation-model measurements to imitate an unavailable render.

The final statement foregrounds the work's strongest Agency claim: agency exists upstream of the final image, distributed across corpus definition, representation, weighting, normalisation, model architecture, pretrained checkpoints, synthesis and retrieval. The text also records explicitly that the moving-image work and its soundtrack are produced **programmatically from end to end**, from corpus ingestion through submission validation.

## NeurIPS edition now implemented

### Base NeurIPS recut

- separate `src/tna/video/neurips.py` renderer;
- complete-source-field opening rather than target-designator ingest;
- no global military/gaze HUD;
- no glitch spectacle;
- restrained global grain;
- longer perceptual holds;
- unresolved coda retaining five incompatible weighting outcomes;
- separate procedural soundtrack;
- package generation with source/encoded duration checks, `<100 MiB` checks, thumbnail, contact sheet and SHA-256 manifest.

### Additional iteration 1 - concentration / erasure

New real measures in `src/tna/video/neurips_metrics.py`:

- normalised Shannon entropy of each weighting distribution;
- entropy effective number of contributors;
- largest contribution share;
- contributor count below an arbitrary displayed threshold.

New moving-image operations:

- `distribution`: concentration of equal/population/GDP/annual-CO2/cumulative-CO2 weightings becomes directly comparable;
- `thresholds`: a logarithmic 0.01%→1% threshold sweep makes practical disappearance visible while reporting the real number of contributors below the current threshold.

Sound:

- spectral breadth responds to the real weighting entropy;
- sparse distribution events are mapped from each weighting's entropy;
- erasure/threshold event density is tied to real contribution counts.

No sonification is presented as a measurement. It is a documented mapping from existing measurements.

### Additional iteration 2 - representation × weighting / concrete pair

New moving-image operations:

- `matrix`: actual generated results are shown as a representation × weighting matrix, making the two independent choices visible in one field;
- `pair`: an equal Israel/Palestine example uses the project's real pixel average and real CLIP retrieval/ranking/margin without adding an editorial slogan.

Sound:

- representation/weighting axes produce two sparse interlocking temporal grids;
- the real equal-pair CLIP retrieval margin controls the separation of two tones; a near-zero margin produces near-unison.

The coda remains deliberately unresolved: same corpus, different weightings, no privileged average.

## Tests

Lightweight NeurIPS tests run in GitHub Actions and pass after the second iteration. They cover:

- schedule contiguity / exact 176-second target;
- presence/order of distribution, matrix, threshold and pair phases;
- entropy/effective-count behaviour;
- threshold counting on normalised shares;
- scale invariance of distribution metrics.

GPU/foundation-dependent rendering is deliberately excluded from CI and still requires the full local/project environment.

## Execute canonical full renderer

From the repository root:

```bash
python scripts/prepare_video_assets.py
python scripts/render_neurips_2026.py
pytest -q src/tests
bash scripts/package-neurips-2026.sh
```

Cheap structural smoke render:

```bash
python scripts/render_neurips_2026.py \
  --width 640 --height 360 --fps 6 --duration 24 \
  --foundation auto --no-audio \
  --out-dir outputs/video/neurips-smoke
```

## Visual review gate for canonical full renderer

Inspect:

```text
outputs/submissions/neurips-2026/the-national-average-contact-sheet.jpg
outputs/submissions/neurips-2026/the-national-average-preview.mp4
```

Specific questions:

1. Are the distribution bars readable without returning to a dashboard aesthetic?
2. Does the representation × weighting matrix remain legible at exhibition distance?
3. Is the threshold sweep perceptible as disappearance rather than simply fading thumbnails?
4. Does the equal Israel/Palestine pair remain severe and factual rather than illustrative or sensational?
5. Is the coda genuinely unresolved, or does one backend/weighting visually dominate as an implied answer?
6. Does the new data-linked sound remain sparse enough that the image carries the argument?

Only revise after observing a concrete failure in the rendered work.

## Submission derivatives

Canonical full-render outputs, when executed, remain:

```text
outputs/submissions/neurips-2026/
  the-national-average-preview.mp4
  the-national-average-thumbnail.png
  the-national-average-contact-sheet.jpg
  media-manifest.json
```

The final NeurIPS description and exact submission-package record are now owned by `krahd/professional-opportunities`; project-level conceptual and implementation source remains here.
