# NeurIPS 2026 execution status — The National Average

**State:** implementation complete; render/visual approval pending.  
**Date:** 9 August 2026.

## Implemented

- separate `neurips` preset at 1920×1080 / 24 fps / 176 seconds;
- separate `src/tna/video/neurips.py` renderer, leaving `video-2026-v1` untouched;
- complete-source-field opening rather than target-designator ingest;
- reduced phase set centred on representation, weighting, erasure and residual;
- no global HUD or glitch spectacle in the NeurIPS renderer;
- plural unresolved coda containing the five principal weighting outcomes;
- separate restrained procedural soundtrack in `neurips_audio.py`;
- regression test for schedule continuity and political-core phases;
- packaging script with source and encoded-duration checks, <100 MiB checks, thumbnail, contact sheet and SHA-256 manifest;
- artwork statement and detailed implementation plan.

## Execute

From the repository root with the foundation-model dependencies available:

```bash
python scripts/prepare_video_assets.py
python scripts/render_neurips_2026.py
bash scripts/package-neurips-2026.sh
```

For a cheap structural smoke render before the full render:

```bash
python scripts/render_neurips_2026.py \
  --width 640 --height 360 --fps 6 --duration 18 \
  --foundation auto --no-audio \
  --out-dir outputs/video/neurips-smoke
```

Then run the repository tests:

```bash
pytest -q src/tests
```

## Review gate

Inspect:

```text
outputs/submissions/neurips-2026/the-national-average-contact-sheet.jpg
outputs/submissions/neurips-2026/the-national-average-preview.mp4
```

Reject/regenerate only for concrete visual failures: unreadably dense source grid, backend images too small to compare, text collisions, overly abrupt transitions, or a coda in which one result visually dominates enough to imply that it is the answer.

Do not reintroduce the baseline HUD merely to make the computation look more technical.

## Submission derivatives

Expected after successful execution:

```text
outputs/submissions/neurips-2026/
  the-national-average-preview.mp4
  the-national-average-thumbnail.png
  the-national-average-contact-sheet.jpg
  media-manifest.json
```

The ≤3-page PDF remains a textual/typesetting derivative. Its canonical prose is `docs/neurips-2026-creative-ai.md`.
