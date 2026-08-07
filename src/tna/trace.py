"""Provenance v2 trace helpers."""

from __future__ import annotations

import json
from pathlib import Path

from .data import Polity, serialise_polity
from .weights import WeightingRun


def relpath(path: Path, root: Path) -> str:
    """Store output paths relative to the run directory when possible."""

    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def build_trace(
    *,
    root: Path,
    corpus_version: str,
    intent: WeightingRun,
    backend_trace: dict[str, object],
    selected: list[Polity],
    outputs: dict[str, Path],
    music: dict[str, object] | None,
    status: str = "ok",
    reason: str | None = None,
    canvas: tuple[int, int] | None = None,
) -> dict[str, object]:
    """Assemble the machine-readable audit record for one backend output."""

    return {
        "status": status,
        "reason": reason,
        "corpus_version": corpus_version,
        "intent": intent.name,
        "weight_source": intent.source,
        "weights": {
            "raw": intent.raw_weights,
            "normalised": intent.weights,
        },
        "entities": [serialise_polity(polity) for polity in selected],
        "backend": backend_trace,
        "music": music,
        "outputs": {key: relpath(value, root) for key, value in outputs.items()},
        "operations": {
            "selection": [polity.code for polity in selected],
            "representation": backend_trace.get("space"),
            "weighting": intent.source,
            "normalisation": (
                f"{canvas[0]}x{canvas[1]} raster canvas, transparent padding flattened to white, "
                "and backend-specific representation"
                if canvas else
                "raster canvas (white background) and backend-specific representation"
            ),
            "synthesis": backend_trace.get("synthesis", backend_trace.get("space")),
        },
    }


def write_json(path: Path, data: dict[str, object]) -> None:
    """Write deterministic, pretty JSON so traces work in diffs and reviews."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
