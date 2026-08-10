# The National Average — Latent Flight

**Status: independent development cut, August 2026.**

`Latent Flight` is a separate moving-image work. It does not replace or modify the
canonical `video-2026-v1` baseline or the NeurIPS 2026 edition.

The film opens with an authored title and a concise statement of the operation, then
enters a software-rendered 3-D traversal through computed representations of the flag
corpus. Its camera accelerates relentlessly through a large volumetric world, searching
like a probe in a nebula for a barycentric average that recedes and changes as four
artistic parameters move. A tactical annotation layer makes the computational and
political route intelligible without turning the work into a report: selection,
representation, commensuration, reconstruction, weighting, erasure, retrieval,
synthesis, and the refusal of settlement.

The spatial material is built from:

- repeated weather systems made from the first three real PCA/eigenflag coordinates
  of the complete corpus;
- non-planar colour-density fields sampled from Palestine, Israel, the United States,
  Germany, and their truncated PCA reconstructions;
- sparse coefficient-driven filaments running through those fields;
- the first twelve eigenflags sampled into moving spectral mist;
- real `4×24×32` Stable Diffusion VAE posterior means diffused into turbulent local
  geometry rather than displayed as grids or images;
- a continuously reweighted barycentric PCA mean expressed as an elongated colour
  front that remains ahead of the camera.

The annotation is trace-bound. It shows computed archive and comparable-set counts,
PCA reconstruction errors, real weighting concentration and effective contributor
counts, the equal Israel/Palestine CLIP tie, and the equal-weight versus historical-CO2
retrieval outcomes. Camera coordinates, route geometry, and chapter-responsive world
deformation are marked as artistic rather than analytical values.

At selected moments, a process aperture briefly exposes a real intermediate artefact
inside the same military/cyber register: an eigenflag, truncated PCA decode, equal
Palestine/Israel pixel mean, weighting-specific pixel average, CLIP query image, or live
barycentric decode. These windows glitch shut rather than becoming a parallel report.

The source flags and reconstructions are never rendered as rectangles, meshes, or
recognisable silhouettes. Their colours remain locally grouped, but their spatial
arrangement is deliberately metaphorical. The world itself changes with the operation:
commensuration splits it, reconstruction bands it, weighting exerts leverage, erasure
removes mass, retrieval collapses it, and synthesis destabilises it. Forward camera
movement, lateral viewpoint cuts, turbulent advection, fog, chromatic fracture, glitches,
and sound remain artistic treatment.

The conclusion samples five real pixel-space weighted averages—equal contribution,
population, GDP, annual CO2, and cumulative historical CO2—into thick particulate flag
apparitions. Their leading contributors remain labelled as computed records. The five
outputs coexist before the work arrives at its final proposition: “The average is not
found. It is enforced.”

## Render

From the repository root:

```bash
python scripts/render_latent_flight.py --preset preview
python scripts/render_latent_flight.py --preset production
```

The preview is 960×540, 12 fps, and 58 seconds. The production master is 1920×1080,
24 fps, and 96 seconds. Production requires the prepared Stable Diffusion VAE model.

A short render without foundation models is useful for checking the pipeline:

```bash
python scripts/render_latent_flight.py \
  --width 640 --height 360 --fps 6 --duration 24 \
  --foundation off --no-audio \
  --out-dir outputs/video/latent-flight-smoke
```

VS Code exposes equivalent `TNA: Render Latent World …` tasks.

Generated files are written under `outputs/video/the_national_average_latent_world/`.
The renderer also writes its source artefacts, stills, technical notes, and a provenance
manifest that distinguishes computed geometry from visual treatment.
