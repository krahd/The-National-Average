"""Real CLIP ViT attention — "where the machine looks".

Runs the CLIP vision transformer with attentions exposed and performs attention
rollout over the patch tokens, yielding a per-flag heatmap of the model's
attention from the CLS token to each image patch. This is a genuine model
internal, not a stylised overlay; it replaces the renderer's fabricated
"feature vector" bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from ..backends.clip_retrieval import CLIPBackend


@dataclass(frozen=True)
class SaliencyRecord:
    code: str
    heat: np.ndarray  # (grid, grid) attention normalised to 0..1 (final rollout)
    grid: int
    model_checkpoint: str
    layer_maps: tuple[np.ndarray, ...] = ()  # cumulative rollout after each layer
    provenance: str = (
        "tna.analysis.saliency.saliency_record via CLIP vision_model attention "
        "rollout (CLS->patch tokens)"
    )

    @property
    def n_layers(self) -> int:
        return len(self.layer_maps) or 1

    def _map(self, index: int) -> np.ndarray:
        maps = self.layer_maps or (self.heat,)
        return maps[max(0, min(index, len(maps) - 1))]

    def heatmap(self, size: tuple[int, int]) -> Image.Image:
        small = Image.fromarray(np.clip(self.heat * 255, 0, 255).astype(np.uint8))
        return small.resize(size, Image.Resampling.BILINEAR)

    def layer_heatmap(self, index: int, size: tuple[int, int]) -> Image.Image:
        small = Image.fromarray(np.clip(self._map(index) * 255, 0, 255).astype(np.uint8))
        return small.resize(size, Image.Resampling.BILINEAR)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "grid": self.grid,
            "model_checkpoint": self.model_checkpoint,
            "provenance": self.provenance,
        }


def _attention_vision_model(clip: CLIPBackend):
    """Return a vision model that actually emits attentions.

    The SDPA/flash attention kernels return ``None`` for attentions, so an eager
    implementation is required for rollout. The eager model is cached on the
    backend instance after first use.
    """

    cached = getattr(clip, "_eager_vision_model", None)
    if cached is not None:
        return cached
    from transformers import CLIPModel

    try:
        model = CLIPModel.from_pretrained(clip.model_checkpoint, attn_implementation="eager")
    except Exception:
        model = CLIPModel.from_pretrained(clip.model_checkpoint)
    vision = model.eval().vision_model
    clip._eager_vision_model = vision
    return vision


def saliency_record(clip: CLIPBackend, code: str) -> SaliencyRecord:
    """Compute the real CLS->patch attention rollout heatmap for one flag."""

    torch = clip.torch
    vision = _attention_vision_model(clip)
    image = Image.fromarray(np.clip(clip.arrays[code], 0, 255).astype(np.uint8))
    inputs = clip.processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = vision(pixel_values=inputs["pixel_values"], output_attentions=True)
    attentions = outputs.attentions
    if not attentions or attentions[0] is None:
        raise RuntimeError(
            "CLIP vision model returned no attentions; an eager attention "
            "implementation is required for saliency rollout."
        )
    seq = attentions[0].shape[-1]
    rollout = torch.eye(seq)
    layer_maps: list[np.ndarray] = []
    grid = int(round((seq - 1) ** 0.5))
    for attention in attentions:
        head_mean = attention.mean(dim=1)[0]  # (seq, seq), averaged over heads
        augmented = head_mean + torch.eye(seq)  # account for the residual connection
        augmented = augmented / augmented.sum(dim=-1, keepdim=True)
        rollout = augmented @ rollout
        # Cumulative CLS->patch attention after this layer, for the "attention
        # forming across depth" animation.
        cls = rollout[0, 1:].numpy()[: grid * grid].reshape(grid, grid)
        cls = (cls - cls.min()) / ((cls.max() - cls.min()) or 1.0)
        layer_maps.append(cls.astype(np.float32))
    heat = layer_maps[-1]
    return SaliencyRecord(
        code=code,
        heat=heat,
        grid=grid,
        model_checkpoint=clip.model_checkpoint,
        layer_maps=tuple(layer_maps),
    )
