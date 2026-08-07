"""Backwards-compatible re-exports.

The renderer was split into `pipeline` (orchestration + real analysis records),
`scenes` (frame compositor), and `compositor` (shared helpers + ffmpeg). Existing
imports of ``tna.video.eccv`` continue to work through these re-exports.
"""

from __future__ import annotations

from .pipeline import (
    BASE_BACKENDS,
    CORPUS_VERSION,
    FOUNDATION_BACKENDS,
    INTENT_LABELS,
    INTENTS,
    TITLE,
    ECCVRenderConfig,
    generate_assets,
    production_codes,
    render_eccv_video,
)
from .scenes import ECCVFrameRenderer

__all__ = [
    "BASE_BACKENDS",
    "CORPUS_VERSION",
    "FOUNDATION_BACKENDS",
    "INTENT_LABELS",
    "INTENTS",
    "TITLE",
    "ECCVRenderConfig",
    "ECCVFrameRenderer",
    "generate_assets",
    "production_codes",
    "render_eccv_video",
]
