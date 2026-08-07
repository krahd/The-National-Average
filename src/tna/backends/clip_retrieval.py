"""Rung 7: CLIP image-embedding retrieval (foundation-model rung).

CLIP organises images by a learned, web-scale semantic geometry. This rung
embeds each selected flag, averages the embeddings barycentrically, and then —
because the space has no decoder — *retrieves the real nation whose flag lies
nearest the average* instead of synthesising a new image. The result is a
different and equally pointed demonstration: the "average" of France, Uruguay,
and Palestine is answered by naming an existing flag, a choice made entirely
inside priors the user never authored.

The backend degrades to a trace-only ``UnavailableBackend`` if Transformers is
absent or the weights cannot be loaded.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .base import Backend, BackendResult, UnavailableBackend


CLIP_MODEL = "openai/clip-vit-base-patch32"


def build_clip_backend(arrays, seed: int = 0):
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except Exception:
        return UnavailableBackend(
            "clip",
            "CLIP image embedding retrieval",
            "Transformers/CLIP dependencies are not installed; install the optional foundation extra to enable retrieval.",
            seed=seed,
        )
    try:
        model = CLIPModel.from_pretrained(CLIP_MODEL)
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    except Exception as exc:
        return UnavailableBackend(
            "clip",
            "CLIP image embedding retrieval",
            f"CLIP weights could not be loaded ({CLIP_MODEL}): {exc}",
            seed=seed,
        )
    return CLIPBackend(arrays, seed=seed, model=model, processor=processor, torch_module=torch)


class CLIPBackend(Backend):
    name = "clip"
    space = "CLIP image embedding"
    learned = True
    # Non-decodable: the averaged embedding is answered by retrieval, not
    # synthesis. The returned image is a real corpus flag, labelled as such.
    decodable = False

    def __init__(self, arrays: dict[str, np.ndarray], seed: int, model, processor, torch_module):
        super().__init__(arrays, seed=seed)
        self.torch = torch_module
        self.model = model.eval()
        self.processor = processor
        self.model_checkpoint = CLIP_MODEL
        self.latent_dim = int(model.config.projection_dim)
        self._cache: dict[str, np.ndarray] = {}
        self._corpus_matrix: np.ndarray | None = None
        self._corpus_codes: list[str] = list(arrays)

    def _embed(self, array: np.ndarray) -> np.ndarray:
        image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
        with self.torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            output = self.model.get_image_features(**inputs)
        # Older Transformers returns the projected embedding tensor directly;
        # 5.x wraps it in an output object carrying the projection in
        # image_embeds (or pooler_output for the projected pooled vector).
        if self.torch.is_tensor(output):
            embedding = output
        else:
            embedding = getattr(output, "image_embeds", None)
            if embedding is None:
                embedding = output.pooler_output
        vector = embedding.squeeze(0).numpy().astype(np.float64)
        # L2-normalise so barycentric averaging and cosine retrieval are
        # consistent; the unit sphere is the natural geometry for CLIP.
        return vector / (np.linalg.norm(vector) or 1.0)

    def encode(self, code: str) -> np.ndarray:
        if code not in self._cache:
            self._cache[code] = self._embed(self.arrays[code])
        return self._cache[code]

    def _ensure_corpus(self) -> None:
        # Embed the full corpus once so the average can be matched against every
        # real flag, not just the selected ones.
        if self._corpus_matrix is None:
            self._corpus_matrix = np.stack([self.encode(code) for code in self._corpus_codes])

    def decode(self, z: np.ndarray) -> BackendResult:
        self._ensure_corpus()
        query = z / (np.linalg.norm(z) or 1.0)
        similarities = self._corpus_matrix @ query
        order = np.argsort(-similarities)
        ranking = [
            {"code": self._corpus_codes[i], "similarity": round(float(similarities[i]), 4)}
            for i in order[:5]
        ]
        nearest = self._corpus_codes[int(order[0])]
        retrieved = Image.fromarray(np.clip(self.arrays[nearest], 0, 255).astype(np.uint8))
        return BackendResult(
            image=retrieved,
            representation=z,
            retrieval={"nearest": nearest, "similarity": ranking[0]["similarity"], "ranking": ranking},
            trace={
                "synthesis": "nearest-nation retrieval from averaged CLIP embedding",
                "nearest_nation": nearest,
                "ranking": ranking,
            },
        )
