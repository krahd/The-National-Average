"""Rung 4: Stable Diffusion VAE latent averaging (foundation-model rung).

This is the sharpest rung of the ladder. Unlike the local ``vae`` rung, whose
latent is fit on the small flag corpus, this backend averages in the latent of
a *web-scale* autoencoder (the VAE component of Stable Diffusion). The averaged
flag is therefore filtered through priors about what an image *is* that the user
never authored and cannot inspect from the output alone.

The backend stays optional: if Diffusers/Torch are absent or the pretrained
weights cannot be loaded (e.g. offline), it degrades to a trace-only
``UnavailableBackend`` rather than failing the run.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .base import Backend, BackendResult, UnavailableBackend


# The fine-tuned MSE autoencoder from the Stable Diffusion family. It is ~335MB
# and downloads once into the standard Hugging Face cache.
SDVAE_MODEL = "stabilityai/sd-vae-ft-mse"

# The SD VAE downsamples by a factor of 8 and was trained at hundreds of pixels.
# We run it at a fixed 4:3 working resolution so its learned priors are actually
# exercised, then return the native decode. Both dimensions are multiples of 8.
WORKING_SIZE = (256, 192)  # (width, height)


def build_sdvae_backend(arrays, seed: int = 0):
    # Dependency import is the first availability gate; loading the weights is
    # the second. The base package remains runnable without either.
    try:
        import torch
        from diffusers import AutoencoderKL
    except Exception:
        return UnavailableBackend(
            "sdvae",
            "Stable Diffusion VAE latent",
            "Diffusers/Torch are not installed; install the optional foundation extra to enable this backend.",
            seed=seed,
        )
    try:
        vae = AutoencoderKL.from_pretrained(SDVAE_MODEL)
    except Exception as exc:  # offline, missing weights, hub error
        return UnavailableBackend(
            "sdvae",
            "Stable Diffusion VAE latent",
            f"Stable Diffusion VAE weights could not be loaded ({SDVAE_MODEL}): {exc}",
            seed=seed,
        )
    return SDVAEBackend(arrays, seed=seed, vae=vae, torch_module=torch)


class SDVAEBackend(Backend):
    name = "sdvae"
    space = "Stable Diffusion VAE latent"
    learned = True
    decodable = True

    def __init__(self, arrays: dict[str, np.ndarray], seed: int, vae, torch_module):
        super().__init__(arrays, seed=seed)
        self.torch = torch_module
        self.vae = vae.eval()
        self.model_checkpoint = SDVAE_MODEL
        width, height = WORKING_SIZE
        self.working_size = WORKING_SIZE
        # Latent is (4, H/8, W/8); flatten size is recorded for the trace.
        self.latent_dim = 4 * (height // 8) * (width // 8)
        # Encoding the same flags repeatedly (e.g. across the alpha morph) is
        # the dominant cost, so cache posterior means per code.
        self._cache: dict[str, np.ndarray] = {}

    def _to_tensor(self, array: np.ndarray):
        # SD VAE expects RGB in [-1, 1], shape (1, 3, H, W). The shared raster is
        # upsampled to the fixed working resolution so the prior operates at a
        # scale it understands; this resize is part of the rung's normalisation.
        image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).resize(
            self.working_size, Image.Resampling.BILINEAR)
        tensor = self.torch.tensor(np.asarray(image) / 255.0, dtype=self.torch.float32)
        return tensor.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0

    def encode(self, code: str) -> np.ndarray:
        if code not in self._cache:
            with self.torch.no_grad():
                posterior = self.vae.encode(self._to_tensor(self.arrays[code])).latent_dist
                # Posterior mean rather than a sample keeps the averaged latent
                # deterministic and traceable, mirroring the local VAE rung.
                self._cache[code] = posterior.mean.squeeze(0).numpy()
        return self._cache[code]

    def decode(self, z: np.ndarray) -> BackendResult:
        with self.torch.no_grad():
            latent = self.torch.tensor(z, dtype=self.torch.float32).unsqueeze(0)
            sample = self.vae.decode(latent).sample.squeeze(0)
            image = (sample / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
        out = Image.fromarray(np.clip(image * 255, 0, 255).astype(np.uint8))
        return BackendResult(
            image=out,
            representation=z,
            trace={
                "synthesis": "Stable Diffusion VAE decode from averaged posterior means",
                "working_resolution": f"{self.working_size[0]}x{self.working_size[1]}",
            },
        )
