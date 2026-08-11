# The National Average - Latent World

**Status: NeurIPS 2026 Creative AI submission edition, August 2026.**

The latent-world film is the moving-image edition of *The National Average* being submitted to the NeurIPS 2026 Creative AI Track. The internal renderer has historically been called `Latent Flight`; the submitted artwork title remains *The National Average*. This edition does not overwrite the preserved `video-2026-v1` baseline, but it supersedes the earlier 176-second NeurIPS matrix cut for the 2026 artwork submission.

The film opens with an authored title and a concise statement of the operation, then enters a software-rendered three-dimensional traversal through computed representations of the flag corpus. Its camera accelerates relentlessly through a large volumetric world, searching like a probe for a barycentric average that recedes and changes as four artistic focus parameters move. A sparse annotation layer makes the computational and political route legible without turning the work into a report: selection, representation, commensuration, reconstruction, weighting, erasure, retrieval, synthesis and the refusal of settlement.

The spatial material is built from:

- repeated weather systems made from the first three real PCA/eigenflag coordinates of the complete corpus;
- non-planar colour-density fields sampled from Palestine, Israel, the United States, Germany and their truncated PCA reconstructions;
- sparse coefficient-driven filaments running through those fields;
- the first twelve eigenflags sampled into moving spectral mist;
- real `4x24x32` Stable Diffusion VAE posterior means diffused into turbulent local geometry rather than displayed as grids or images;
- a continuously reweighted barycentric PCA mean expressed as an elongated colour front that remains ahead of the camera.

The annotation is trace-bound. It shows computed archive and comparable-set counts, PCA reconstruction errors, real weighting concentration and effective contributor counts, the equal Palestine/Israel CLIP tie, and the equal-weight versus historical-CO2 retrieval outcomes. Camera coordinates, route geometry and chapter-responsive world deformation are artistic rather than analytical values.

At selected moments, a process aperture briefly exposes a real intermediate artefact inside the same military/cyber register: an eigenflag, truncated PCA decode, equal Palestine/Israel pixel mean, weighting-specific pixel average, CLIP query image or live barycentric decode. These windows glitch shut rather than becoming a parallel report.

The source flags and reconstructions are never rendered as rectangles, meshes or recognisable silhouettes in the volumetric field. Their colours remain locally grouped, but their spatial arrangement is deliberately metaphorical. The world itself changes with the operation: commensuration splits it, reconstruction bands it, weighting exerts leverage, erasure removes mass, retrieval collapses it and synthesis destabilises it. Forward camera movement, lateral viewpoint cuts, turbulent advection, fog, chromatic fracture, glitches and sound remain artistic treatment.

The conclusion samples five real pixel-space weighted averages - equal contribution, population, GDP, annual CO2 and cumulative historical CO2 - into thick particulate flag apparitions. Their leading contributors remain labelled as computed records. The five outputs coexist before the work arrives at its final proposition: **THE AVERAGE IS NOT FOUND. IT IS ENFORCED.** The following line makes the computational limit explicit: **THE SYSTEM CAN EXECUTE THE CHOICE. IT CANNOT JUSTIFY IT.**

## Render

From the repository root:

```bash
python scripts/render_latent_flight.py --preset preview
python scripts/render_latent_flight.py --preset production
```

The preview preset is 960x540, 12 fps and 58 seconds. The NeurIPS production master is 1920x1080, 24 fps and 96 seconds, with deterministic seed `20260810` and foundation representations required.

The canonical production output is:

`outputs/video/the_national_average_latent_world/the_national_average_latent_world.mp4`

A short render without foundation models is useful only for pipeline checks and must not be substituted for the submission master:

```bash
python scripts/render_latent_flight.py \
  --width 640 --height 360 --fps 6 --duration 24 \
  --foundation off --no-audio \
  --out-dir outputs/video/latent-flight-smoke
```

VS Code exposes equivalent `TNA: Render Latent World ...` tasks.

The renderer writes source artefacts, stills, technical notes and a provenance manifest that distinguishes computed geometry from visual treatment.
