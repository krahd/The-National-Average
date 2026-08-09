# NeurIPS 2026 Creative AI — artwork package

**Artwork:** The National Average  
**Artist:** Tomas Laurenzo  
**Track:** NeurIPS 2026 Creative AI — Agency  
**Status:** submission package in production  
**Deadline target:** 10 August 2026 AoE; recheck the live OpenReview form immediately before upload because the static NeurIPS call still displays the pre-extension date.

## Submission description draft

### The National Average

*The National Average* is a generative moving-image artwork about a familiar political operation disguised as arithmetic: averaging.

The work begins with national flags. Their apparent simplicity is useful because flags are already compressed political objects. They reduce complicated and contested histories of sovereignty, conquest, independence, occupation, territory, identity, and belonging to deliberately recognisable geometries and colours. They are designed to distinguish one political entity from another. *The National Average* asks what happens when a computational system is instructed to make them commensurable.

The system does not contain a single averaging operation. It implements a ladder of representation spaces. Flags can be treated directly as pixels; reduced to dominant colours and their relative areas; reconstructed as structured geometry; projected into a locally learned PCA basis; encoded through a locally trained variational autoencoder; or represented by pretrained foundation models including a Stable Diffusion VAE and CLIP. The same source objects can also be assigned different weights: equal contribution, population, area, GDP, carbon emissions, energy consumption, recognition year, or explicit manual values. Each result is accompanied by machine-readable provenance recording the selected entities, representation space, weights, normalisation, model/checkpoint information, seed, and synthesis operation.

The procedure is intentionally excessive. There is no neutral answer hidden behind these alternatives. Before an average can exist, a system must decide what counts as an object, how that object will be represented, which differences are preserved, which are normalised away, and how much each entity contributes. The final image can look authoritative precisely because the operations that manufactured it have disappeared into the result.

The moving-image work makes those operations perceptible. It begins close to the visual language of statistical and technical demonstration: ordered flags, weights, coordinates, computed relations, synthetic outputs. As the film progresses, apparently minor representational choices accumulate. Different spaces produce incompatible representative objects. A country can become visually negligible under one weighting and structurally dominant under another. Foundation-model spaces introduce relationships that cannot be inferred from the visible geometry of the flags themselves. What initially appears to be a search for a representative image becomes a demonstration that representation has already taken place before the final image is generated.

### AI/ML

Machine learning is not used as a decorative image generator. It is one layer in a deliberately heterogeneous representational apparatus. The work compares non-learned representations with locally learned spaces and pretrained foundation-model representations. PCA and a local VAE make explicit how a corpus-specific learned space changes the meaning of an average; Stable Diffusion VAE and CLIP place the same objects inside representational spaces learned elsewhere from much larger visual corpora. CLIP is used as an embedding space and retrieval mechanism rather than as an unquestioned semantic authority. Optional model dependencies emit explicit unavailable states rather than silently substituting another method.

The implementation enforces an epistemic invariant: any quantity presented on screen as an analysis result — contribution, residual, saliency, embedding relation, ranking, retrieval or weighting — must originate in computed records. Visual treatment may animate or distort those records but may not fabricate measurements for dramatic effect.

### Agency

The work addresses agency by refusing to locate it solely in either artist or model.

The synthetic national object is produced through a chain of decisions distributed across source political symbols, statistical datasets, inclusion rules, representation spaces, weighting criteria, normalisation procedures, learned models, synthesis methods, and artistic selection. No single stage simply *contains* the final meaning. Yet the finished average can appear inevitable once that chain disappears.

This is the political problem of the work. Statistical systems often present the outcome while concealing the prior decisions that made the outcome computable. *The National Average* gives those decisions duration. It turns them into audiovisual material so that agency can no longer be attributed only to the final generative operation.

The work therefore treats Agency not as a competition between human creativity and machine autonomy, but as an inquiry into where representational decisions become effective and where responsibility can disappear. The system can produce an average automatically. It cannot decide what an average ought to mean.

### Technical form

- generative/data-driven moving image;
- canonical baseline: 180 seconds, 1920×1080, 24 fps;
- deterministic render seed: `20260613`;
- foundation representations: `openai/clip-vit-base-patch32`, `stabilityai/sd-vae-ft-mse`, with resolved model snapshot revisions stored in render provenance;
- soundtrack generated from the same data-driven audiovisual system;
- complete source, data provenance, analysis records, render pipeline and preservation documentation in the public project repository.

### Artist biography

Tomas Laurenzo is an artist, computer scientist, and Associate Professor of Critical Media Practices at the University of Colorado Boulder. His work examines the political and aesthetic conditions of computation, artificial intelligence, interaction, and technical media through artworks, software systems, and scholarly research. His work has been presented at venues including NeurIPS, Ars Electronica, SIGGRAPH Asia, ISEA, CVPR, Sónar+D and MUTEK. Recent generative works include *Montevideo, 1983*, *Hommage Numérique*, *Abandoned Future*, and *Ave Imperator*. His research on media appropriation, human–computer ideology, and the user–programmer continuum provides part of the theoretical background for *The National Average*.

## NeurIPS-specific edit direction

The existing three-minute baseline should be treated as source material, not assumed final. Improvements should preserve the analytical invariant while strengthening the artwork as an artwork:

1. reduce explanatory density wherever the image already demonstrates the operation;
2. strengthen the first transition from apparently neutral statistical procedure to visibly incompatible averages;
3. allow longer holds on materially different outputs so their political/aesthetic difference can be perceived before the next operation appears;
4. avoid interface/demo pacing; use computation as dramaturgy rather than tutorial sequence;
5. make the learned-representation transition legible without fetishising model names;
6. let disappearance/erasure of low-weight entities become visible through composition rather than explanatory captions where possible;
7. audit soundtrack dynamics against the visual escalation; sound should articulate changes of representational regime rather than continuously decorate the image;
8. end without selecting a privileged or 'correct' average.

## Required upload assets

- `the-national-average-neurips-2026.pdf` — ≤3 pages, NeurIPS format, containing the description above in compressed form;
- `the-national-average-thumbnail.png` — representative still, <100 MB;
- `the-national-average-preview.mp4` — ≤3 minutes and <100 MB.

The repository's canonical work should remain venue-independent. Submission-only derivatives belong in `outputs/` or a release package, not as the canonical artwork source.
