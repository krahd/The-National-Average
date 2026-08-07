"""Real machine-vision analysis layer for *The Average Nation*.

Every function here computes a value from the flag corpus or the loaded models
and returns a typed, JSON-serialisable record. The video renderer consumes these
records and is forbidden from inventing measurements: if a number or overlay
appears on screen, it traces back to one of these functions.
"""

from .components import ComponentRecord, Region, component_record
from .embedding import EmbeddingGeometry, embedding_geometry
from .erasure import Contributor, ErasureRecord, erasure_record
from .recognition import RecognitionRecord, Retrieved, recognition_record
from .residual import ResidualRecord, palette_residual, pca_residual
from .saliency import SaliencyRecord, saliency_record

__all__ = [
    "ComponentRecord",
    "Region",
    "component_record",
    "EmbeddingGeometry",
    "embedding_geometry",
    "Contributor",
    "ErasureRecord",
    "erasure_record",
    "RecognitionRecord",
    "Retrieved",
    "recognition_record",
    "ResidualRecord",
    "pca_residual",
    "palette_residual",
    "SaliencyRecord",
    "saliency_record",
]
