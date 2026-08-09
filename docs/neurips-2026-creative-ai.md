# NeurIPS 2026 Creative AI — artwork package

**Artwork:** The National Average  
**Artist:** Tomas Laurenzo  
**Track:** NeurIPS 2026 Creative AI — Agency  
**Status:** expanded submission edition implemented; full render/visual approval pending  
**Deadline target:** 10 August 2026 AoE; recheck the live OpenReview form immediately before upload.

## Submission description draft

### The National Average

*The National Average* is a generative moving-image work about averaging as a representational operation.

The source material is a corpus of national and quasi-national flags. Flags are useful here because they already perform an extreme compression: historical processes of sovereignty, occupation, independence, territory and political identity are transformed into a small number of colours and geometries designed to remain recognisable. The work submits these objects to a second compression. It asks what a computational system means when it is asked to produce their average.

There is no unique operation hidden under that word. An image can be averaged directly in pixel space, reduced to colour distributions, reconstructed through deterministic geometry, projected into a corpus-specific PCA basis, or moved into representations produced elsewhere by pretrained systems such as CLIP and a Stable Diffusion VAE. Independently, the same political entities can be given equal weight or weighted by population, GDP, annual CO2 emissions, or cumulative historical CO2 emissions. Representation and weighting are orthogonal decisions, although a final synthetic image can make both disappear.

The NeurIPS edition gives these decisions duration. It begins with the source corpus, before any average is shown. Different representation spaces then produce incompatible objects from the same input. Weighting distributions are shown not only through final images but through their actual concentration: normalised entropy, effective contributor count, maximum contribution, and the gradual disappearance produced by a changing contribution threshold. A representation × weighting matrix places both axes in the same field, making visible that changing either operation changes the object called *the average*.

A later scene takes two flags, Israel and Palestine, with exactly equal contribution. The film presents their real pixel average together with the CLIP retrieval relation computed by the project. Nothing is inferred from this operation beyond what it does: two politically irreducible objects can become statistically commensurable, and the representation can return a precise numerical relation while remaining incapable of deciding what that relation should mean.

The film ends with several incompatible averages derived from the same corpus. None is designated as correct.

### AI/ML

Machine learning is one layer of a heterogeneous representational apparatus rather than a decorative generator placed after the argument.

The NeurIPS edition uses:

- direct pixel averaging;
- dominant-colour/palette representations;
- PCA/eigenflag representations learned from the project corpus;
- deterministic SVG/structured recomposition;
- `openai/clip-vit-base-patch32` for image embeddings, retrieval, saliency and representational relations;
- `stabilityai/sd-vae-ft-mse` as a pretrained latent image representation.

The five weighting intentions used in the film are equal contribution, population, GDP, annual CO2, and cumulative historical CO2. Every displayed contribution, residual, saliency result, retrieval, similarity, weighting, entropy, effective contributor count and threshold count originates in a computed record. The renderer can change duration, scale and visual arrangement; it may not fabricate an analytical result for dramatic effect.

The soundtrack follows the same rule. It is generated procedurally, but selected sonic parameters are mapped from existing records: weighting concentration affects spectral breadth, erasure affects sparse-event density, and the retrieval margin in the equal Israel/Palestine pair controls the separation between two tones. These mappings are sound design; the sound is not presented as another measurement.

### Agency

The artwork's agency does not begin with the final generative operation.

Before an average can be produced, the system must already have a corpus; decide which political entities count as members of it; choose a representation; choose a weighting criterion; normalise values; choose learned or non-learned computational spaces; and decide how a representation will be converted back into an image. Some of these decisions belong to this project. Others are inherited from statistical datasets, model training, checkpoint construction, and prior visual corpora.

The resulting image can nevertheless appear singular. It has the authority of an answer even when its conditions are multiple.

The NeurIPS edition treats this disappearance of prior decisions as the central problem of Agency. Equal weighting produces one object; population another; GDP and carbon histories produce others. Changing representation while holding the weighting fixed produces another set of disagreements. A threshold can leave every country technically included while making many of them practically invisible. The computational system can execute all of these operations automatically. It cannot determine which one ought to represent a nation, a world, or a political relation.

### Technical form

- single-channel generative/data-driven moving image;
- NeurIPS edition target: **176 seconds**, 1920×1080, 24 fps;
- deterministic render seed: `20260613`;
- foundation representations: `openai/clip-vit-base-patch32`, `stabilityai/sd-vae-ft-mse`, with resolved local snapshot revisions retained in render provenance;
- separate NeurIPS renderer; preserved canonical `video-2026-v1` baseline is not overwritten;
- procedural stereo soundtrack linked to computed project records;
- complete source, datasets/provenance, analysis records, render pipeline, tests, packaging scripts and preservation documentation maintained in the project repository.

### Artist biography

Tomas Laurenzo is an artist, computer scientist, and Associate Professor of Critical Media Practices at the University of Colorado Boulder. His work examines the political and aesthetic conditions of computation, artificial intelligence, interaction, and technical media through artworks, software systems, and scholarly research. His work has been presented at venues including NeurIPS, Ars Electronica, SIGGRAPH Asia, ISEA, CVPR, Sónar+D and MUTEK. Recent generative works include *Montevideo, 1983*, *Hommage Numérique*, *Abandoned Future*, and *Ave Imperator*. His earlier research on media appropriation, human–computer ideology, and the user–programmer continuum forms part of the theoretical context of *The National Average*.

## Current production / review gate

The NeurIPS renderer and its two additional expansion/test iterations are implemented and lightweight tests pass. The full foundation-model render has not yet been aesthetically approved. Before submission:

1. render the 176-second 1920×1080 edition;
2. inspect the full film and contact sheet;
3. verify that the representation × weighting matrix is legible without becoming a dashboard;
4. verify that the threshold sweep reads as disappearance rather than merely animation;
5. verify that the Israel/Palestine scene remains severe, factual and non-sensational;
6. verify that the data-linked sound remains subordinate to the image;
7. package and validate preview/thumbnail;
8. typeset this statement to ≤3 NeurIPS pages.

## Required upload assets

- `the-national-average-neurips-2026.pdf` — ≤3 pages, NeurIPS format;
- `the-national-average-thumbnail.png` — <100 MB;
- `the-national-average-preview.mp4` — ≤3 minutes and <100 MB.

Submission derivatives remain separate from the venue-independent canonical artwork source.
