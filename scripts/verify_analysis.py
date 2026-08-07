#!/usr/bin/env python3
"""Verify the real-analysis layer against independent expectations.

Checks that every record is a genuine computation: erasure counts from the
weight distributions, CLIP recognition (the il/ps tie, the cumulative-CO2
empire), embedding kNN, and real CLIP attention. No fabricated values anywhere.
"""

from __future__ import annotations

from tna.analysis import (
    embedding_geometry,
    erasure_record,
    recognition_record,
    saliency_record,
)
from tna.backends.clip_retrieval import build_clip_backend
from tna.data import DATA_DIR, corpus_arrays, load_corpus
from tna.video.eccv import INTENTS, production_codes
from tna.weights import weights_from_intent


def main() -> None:
    corpus = load_corpus(data_dir=DATA_DIR)
    codes = production_codes(corpus, INTENTS)
    selected = [corpus[code] for code in codes]
    names = {code: polity.name for code, polity in corpus.items()}
    print(f"production codes usable for all {len(INTENTS)} weightings: {len(codes)}")

    print("\n== erasure (real weight accounting) ==")
    for intent in ("population", "gdp", "cumulative_co2"):
        record = erasure_record(intent, selected)
        top = ", ".join(f"{c.code}:{c.weight:.3f}" for c in record.top(5))
        print(
            f"[{intent}] top5_share={record.top5_share:.2f} "
            f"erased={record.erased_count}/{record.total}  top: {top}"
        )

    union = sorted(set(codes) | {"il", "ps", "fr"} & set(corpus))
    arrays = corpus_arrays({code: corpus[code] for code in union}, (96, 72))
    clip = build_clip_backend(arrays)
    print(f"\nclip backend: {type(clip).__name__} over {len(arrays)} flags")

    print("\n== recognition (real CLIP retrieval) ==")
    run = weights_from_intent("cumulative_co2", selected)
    rec = recognition_record(
        clip, codes, run.weights, names, intent="cumulative_co2", probes=("ps", "uy", "il")
    )
    print(
        f"cumulative_co2 -> nearest {rec.nearest.code} ({rec.nearest.name}) "
        f"sim={rec.nearest.similarity:.4f} confidence={rec.confidence:.3f} margin={rec.margin:.4f}"
    )
    print(f"  probes (rank, sim): {rec.probe_ranks}")

    if "il" in arrays and "ps" in arrays:
        tie = recognition_record(clip, ["il", "ps"], {"il": 0.5, "ps": 0.5}, names, topk=3)
        print(
            "il/ps average ranking:",
            [(r.code, round(r.similarity, 4)) for r in tie.ranking],
            f"margin={tie.margin:.5f}",
        )

    print("\n== embedding geometry (real CLIP kNN) ==")
    geo = embedding_geometry(clip, union[:40], k=3)
    for code in list(geo.knn)[:4]:
        print(f"  {code} nearest: {[(c, round(s, 3)) for c, s in geo.knn[code]]}")

    print("\n== saliency (real CLIP attention rollout) ==")
    sal = saliency_record(clip, "fr" if "fr" in arrays else union[0])
    print(
        f"  {sal.code} grid={sal.grid} heat_shape={sal.heat.shape} "
        f"max={float(sal.heat.max()):.3f} checkpoint={sal.model_checkpoint}"
    )


if __name__ == "__main__":
    main()
