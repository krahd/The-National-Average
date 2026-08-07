"""Video production helpers for exhibition outputs.

Implementation lives in `pipeline` (orchestration + real analysis), `scenes`
(frame compositor), and `compositor` (shared helpers + ffmpeg). `eccv` remains as
a backwards-compatible re-export.
"""

from .foundation import prepare_foundation_assets
from .pipeline import ECCVRenderConfig, render_eccv_video

__all__ = ["ECCVRenderConfig", "prepare_foundation_assets", "render_eccv_video"]
