# NeurIPS 2026 Creative AI - artwork package

**Artwork:** *The National Average*  
**Artist:** Tomas Laurenzo  
**Track:** NeurIPS 2026 Creative AI - Agency  
**Submission artwork:** latent-world moving-image edition  
**Canonical branch:** `main`  
**Canonical renderer:** `scripts/render_latent_flight.py`  
**Canonical production output:** `outputs/video/the_national_average_latent_world/the_national_average_latent_world.mp4`

## Authoritative submission edition

The work being submitted to NeurIPS 2026 Creative AI is the accelerating volumetric latent-world film implemented on `main`, not the earlier 176-second representation/weighting-matrix cut.

The production preset in `presets/latent-flight-2026.json` defines:

- 1920x1080;
- 24 fps;
- 96 seconds;
- deterministic render seed `20260810`;
- foundation representations required;
- procedural stereo soundtrack enabled.

The earlier NeurIPS-specific 176-second renderer remains historical project material but is **superseded for this submission**. It must not be used to prepare the artwork description, thumbnail or video preview.

## Work

*The National Average* is a generative moving-image work in which a camera accelerates through a software-rendered volumetric world built from computational representations of national and quasi-national flags. It appears to search for a centre: a barycentric average that remains ahead of it and changes as the conditions of averaging change. The search never settles. The conclusion places five incompatible weighted averages in the same field and states: **THE AVERAGE IS NOT FOUND. IT IS ENFORCED.**

The source material is a corpus of political symbols. The work treats averaging as a sequence of consequential operations rather than a neutral calculation: selection, representation, normalisation, weighting and synthesis. This strategy is developed as *critical averaging* in the related accepted IBERAMIA 2026 paper *Critical Averaging and the Politics of Statistical Representation*.

The film turns the apparatus into duration and spatial pursuit. Its opening makes the corpus boundary explicit: 251 political symbols enter the archive and 199 survive the metadata boundary. The traversal then moves through a route of selection, representation, commensuration, reconstruction, weight/power, threshold/erasure, foundation-model retrieval, moving synthesis and no settlement.

The volumetric material includes:

- weather systems derived from the first three coordinates of a 32-component PCA/eigenflag representation of the full corpus;
- non-planar colour-density fields from Palestine, Israel, the United States and Germany and their truncated PCA reconstructions;
- spectral mist derived from eigenflags;
- Stable Diffusion VAE posterior means diffused into turbulent spatial material;
- a continuously reweighted PCA barycentre rendered as a receding colour front ahead of the camera.

Trace-bound process apertures expose real intermediate artefacts: eigenflags, truncated PCA decodes, the equal Palestine/Israel pixel mean, weighting-specific pixel averages, CLIP query images and live PCA barycentric decodes. The final field samples five real pixel-space weighted averages: equal contribution, population, GDP, annual CO2 and cumulative historical CO2.

## AI and ML

Machine learning is part of the work's representational apparatus rather than a decorative generator. The film uses a PCA basis fitted to the project corpus, `stabilityai/sd-vae-ft-mse` as a pretrained latent image representation, and `openai/clip-vit-base-patch32` for learned semantic relations and retrieval.

The distinction between computed quantities and artistic treatment is strict. Corpus membership, PCA coordinates and residuals, weighting statistics, threshold erasure, retrieval findings and intermediate artefacts originate in project records. Camera movement, route geometry, fog, advection, deformation, chromatic fracture and glitches are artistic operations. The renderer may dramatise a computed relation; it may not fabricate one.

The soundtrack follows the same boundary. Artistic focus weights and real PCA-coordinate energies modulate its industrial sound field, and visual cuts create corresponding sonic ruptures. The soundtrack is not presented as a measurement or national sonification.

## Agency

Agency is located upstream of the final generated image. The moving barycentre depends on prior decisions about corpus membership, missing-data boundaries, weighting criteria, representational spaces, model architectures, checkpoints and reconstruction. Some decisions are authored in the artwork; others are inherited from statistical institutions, public datasets, flag taxonomies, training corpora and pretrained models.

The film therefore does not reduce Agency to a competition between human and machine authorship. It makes visible how the capacity to determine what becomes representative is distributed across the apparatus. The camera can pursue a mathematically valid centre indefinitely because no privileged centre exists before those operations manufacture one.

## Submission package

The canonical opportunity package is maintained in:

`krahd/professional-opportunities/artistic-submissions/2026-08-10_neurips-creative-ai_the-national-average/`

The artwork-description source must describe this 96-second latent-world edition. The thumbnail must be a genuine frame from this renderer. The video asset must be the 96-second production master or a submission encoding made directly from that master, not the obsolete 176-second derivative.
