"""Guard the core invariant: scenes display only real, analysis-derived values.

The renderer's scenes must not synthesise any number, bar, or position from
``sin``/``random``/``lerp``-style generators. Layout easing lives in
``compositor`` and is allowed there; ``scenes.py`` must stay clean.
"""

from __future__ import annotations

import re
from pathlib import Path

SCENES = Path(__file__).resolve().parents[1] / "tna" / "video" / "scenes.py"

# Patterns that would indicate a fabricated (procedurally generated) value
# leaking into a displayed quantity.
FORBIDDEN = (
    r"math\.sin",
    r"np\.sin",
    r"\brandom\b",
    r"\brng\b",
    r"\blerp\s*\(",
    r"np\.random",
)


def test_scenes_contain_no_fabrication_patterns() -> None:
    source = SCENES.read_text(encoding="utf-8")
    lines = source.splitlines()
    offending: dict[str, list[int]] = {}
    for pattern in FORBIDDEN:
        matches = [i + 1 for i, line in enumerate(lines) if re.search(pattern, line)]
        if matches:
            offending[pattern] = matches
    assert not offending, f"fabrication patterns found in scenes.py: {offending}"


def test_scenes_pull_values_from_analysis_records() -> None:
    source = SCENES.read_text(encoding="utf-8")
    # The scenes must reference the real records, not invent equivalents.
    for marker in ("self.assets.recognition", "self.assets.erasure", "self.assets.embedding", "saliency", "corpus_similarity", "pca_recon"):
        assert marker in source, f"expected scenes.py to consume {marker!r}"
