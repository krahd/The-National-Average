"""Moving-image production for The National Average.

``pipeline`` generates analysis-derived assets and provenance, ``scenes``
renders frames, and ``compositor`` performs final frame/audio composition.
"""

from .foundation import prepare_foundation_assets
from .pipeline import VideoRenderConfig, render_video

__all__ = ["VideoRenderConfig", "prepare_foundation_assets", "render_video"]
