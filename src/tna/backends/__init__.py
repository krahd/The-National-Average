"""Visual representation backends."""

from .base import Backend, BackendResult, UnavailableBackend, barycentric_average
from .palette import PaletteBackend
from .pca import PCABackend
from .pixel import PixelBackend
from .svg import SVGBackend

__all__ = [
    "Backend",
    "BackendResult",
    "UnavailableBackend",
    "barycentric_average",
    "PaletteBackend",
    "PCABackend",
    "PixelBackend",
    "SVGBackend",
]
