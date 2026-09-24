# NeurIPS 2026 execution status - The National Average

**State:** accepted to the NeurIPS 2026 Creative AI Track, Artwork track.
**Submission date:** 11 August 2026.
**Decision:** Accept, 18 September 2026; OpenReview decision modified 22 September 2026.
**Acceptance notification supplied:** 23 September 2026.
**Submission number:** **197**.
**OpenReview:** `https://openreview.net/forum?id=MnitJ1jTdQ`.

## Authoritative submission edition

The artwork submitted to NeurIPS 2026 is the latent-world moving-image edition implemented on `main` and rendered by:

```text
scripts/render_latent_flight.py
presets/latent-flight-2026.json
```

Production preset:

- 1920×1080;
- 24 fps;
- 96 seconds;
- deterministic seed `20260810`;
- foundation representations required;
- procedural soundtrack enabled.

Canonical production output:

```text
outputs/video/the_national_average_latent_world/
  the_national_average_latent_world.mp4
```

The earlier 176-second representation/weighting-matrix renderer remains in the repository as historical implementation material, but it is **superseded for the NeurIPS artwork submission**. It must not be used for the description, thumbnail or preview.

## Artwork structure

The latent-world film converts the critical-averaging apparatus into a continuously traversed volumetric environment. Its route is:

1. selection;
2. representation;
3. commensuration;
4. reconstruction;
5. weight / power;
6. threshold / erasure;
7. foundation-model retrieval;
8. moving synthesis;
9. no settlement.

The opening exposes the real corpus boundary: 251 political symbols enter the archive and 199 survive the metadata boundary.

Spatial material includes:

- weather systems derived from the first three coordinates of a 32-component PCA/eigenflag representation;
- colour-density fields from source flags and truncated PCA reconstructions;
- eigenflags expressed as spectral mist;
- Stable Diffusion VAE posterior means expressed as turbulent local geometry;
- a continuously reweighted PCA barycentre rendered as a receding colour front.

Trace-bound process apertures expose real intermediate artefacts, including eigenflags, PCA decodes, the equal Palestine/Israel pixel mean, weighting-specific averages, CLIP query material and the live barycentric decode.

The conclusion places five real weighted pixel-space averages in the same field: equal contribution, population, GDP, annual CO2 and cumulative historical CO2. The film does not select a privileged result.

## Computed / artistic boundary

Computed project records provide corpus membership, PCA coordinates and residuals, weighting statistics, concentration and effective-contributor measures, threshold erasure, retrieval findings and intermediate images.

Camera movement, route geometry, fog, advection, deformation, chromatic fracture and glitches are artistic operations. The provenance manifest keeps that distinction explicit. The renderer may dramatise a computed relation but may not invent one.

The soundtrack follows the same rule: artistic focus weights and real PCA-coordinate energies modulate an industrial sound field, while visual cuts produce corresponding ruptures. It is not presented as a measurement or national sonification.

## NeurIPS submission package

The canonical opportunity package is:

```text
krahd/professional-opportunities/
  artistic-submissions/2026-08-10_neurips-creative-ai_the-national-average/
```

The submitted description describes only the 96-second latent-world artwork and uses genuine frames regenerated from this repository's `main` branch. The portal thumbnail is likewise a genuine latent-world frame rather than the obsolete 176-second submission derivative.

The description was compiled through the official NeurIPS 2026 `creativeai` template workflow. Tomas Laurenzo directly confirmed submission on 11 August 2026 and supplied submission number **197** on 10 September 2026. The persistent OpenReview forum is `MnitJ1jTdQ` (`https://openreview.net/forum?id=MnitJ1jTdQ`). The Program Chairs recorded an **Accept** decision on 18 September 2026, modified 22 September 2026; Tomas supplied the acceptance notification and final reviews on 23 September 2026. The exact portal receipt timestamp and byte identity of the uploaded video remain unarchived.

## Review outcome

The submission was accepted to the Artwork track. The complete decision and reviewer record is preserved in `docs/NEURIPS-2026-REVIEWS.md`.

The two official reviews identify two distinct revision problems. Reviewer 18wG found the moving image visually compelling but too cryptic to communicate the computational and political relations without the paper, making visual legibility itself a future design question. Reviewer YPBT judged the work conceptually and technically rich and recommended inclusion, while requesting clearer definitions of the search and field, an explicit account of the metadata boundary, justification for the four-nation focus, and clarification of the binary division discussed around line 75 in relation to distributed agency.

These comments apply to future development; they do not alter the frozen submitted edition.

## Programmatic production

The moving image and soundtrack are produced programmatically from end to end. Corpus ingestion, weighting, representation, analysis, world construction, frame composition, procedural sound synthesis, encoding and provenance are executed from source code and versioned parameters.

## Historical renderer

The separate `src/tna/video/neurips.py` 176-second renderer, its concentration/erasure metrics, representation × weighting matrix, equal Israel/Palestine sequence and associated tests remain useful project history. They are not the current NeurIPS artwork deliverable and must not be treated as the submission master in future packaging work.

## Post-submission record

- Preserve the frozen submitted description, thumbnail and video manifestations without substituting the superseded 176-second derivative.
- Preserve the accepted OpenReview record at `https://openreview.net/forum?id=MnitJ1jTdQ` and the full reviews in `docs/NEURIPS-2026-REVIEWS.md`.
- Archive exact uploaded binaries and checksums if recovered.
- Await the organisers' further details and next steps; record them separately from the frozen submission.
- Keep later production or exhibition versions separate from the frozen NeurIPS submission.
