"""Rung 3: optional PyTorch convolutional VAE backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..data import ROOT_DIR
from .base import Backend, BackendResult, UnavailableBackend


CHECKPOINT = ROOT_DIR / "models" / "flag_vae.pt"


def build_vae_backend(arrays, seed: int = 0, train: bool = False, epochs: int = 10):
    """Construct the optional VAE backend or a trace-only unavailable backend."""

    try:
        import torch
    except Exception:
        return UnavailableBackend(
            "vae",
            "nonlinear learned latent",
            "PyTorch is not installed; install the optional ml extra and rerun with --train-vae.",
            seed=seed,
        )
    if train:
        train_vae(arrays, CHECKPOINT, seed=seed, epochs=epochs)
    if not CHECKPOINT.exists():
        return UnavailableBackend(
            "vae",
            "nonlinear learned latent",
            "No VAE checkpoint exists. Rerun with --train-vae after installing the optional ml extra.",
            seed=seed,
        )
    return VAEBackend(arrays, CHECKPOINT, seed=seed, torch_module=torch)


class VAEBackend(Backend):
    name = "vae"
    space = "nonlinear learned latent"
    learned = True
    decodable = True

    def __init__(self, arrays: dict[str, np.ndarray], checkpoint: Path, seed: int, torch_module):
        super().__init__(arrays, seed=seed)
        # Torch is passed in from build_vae_backend so importing the package
        # does not require PyTorch unless the user asks for the VAE rung.
        self.torch = torch_module
        height, width, _ = next(iter(arrays.values())).shape
        self.model = build_flag_vae(self.torch, in_hw=(height, width), latent=64)
        payload = self.torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.model_checkpoint = str(checkpoint)
        self.latent_dim = 64

    def encode(self, code: str) -> np.ndarray:
        with self.torch.no_grad():
            tensor = array_to_tensor(self.arrays[code], self.torch)
            # Use the posterior mean rather than sampling. That makes repeated
            # runs deterministic and makes the averaged latent traceable.
            mu, _ = self.model.encode(tensor)
            return mu.squeeze(0).numpy()

    def decode(self, z: np.ndarray) -> BackendResult:
        with self.torch.no_grad():
            tensor = self.torch.tensor(z, dtype=self.torch.float32).unsqueeze(0)
            recon = self.model.decode(tensor).squeeze(0).permute(1, 2, 0).numpy()
        image = Image.fromarray(np.clip(recon * 255, 0, 255).astype(np.uint8))
        return BackendResult(
            image=image,
            representation=z,
            trace={"synthesis": "PyTorch VAE decode from averaged posterior means"},
        )


def train_vae(arrays: dict[str, np.ndarray], checkpoint: Path, seed: int, epochs: int) -> None:
    import torch
    import torch.nn.functional as functional

    torch.manual_seed(seed)
    height, width, _ = next(iter(arrays.values())).shape
    model = build_flag_vae(torch, in_hw=(height, width), latent=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # This is intentionally full-batch training over a small visual corpus. It
    # is a demonstrator of nonlinear latent averaging, not a production model.
    data = torch.cat([array_to_tensor(array, torch) for array in arrays.values()], dim=0)
    total = max(1, epochs)
    print(f"  [vae] training on {data.size(0)} flags for {total} full-batch epochs ...")
    for epoch in range(total):
        recon, mu, logvar = model(data)
        reconstruction = functional.mse_loss(recon, data, reduction="sum") / data.size(0)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.size(0)
        # A small KL weight keeps reconstructions legible while still forcing
        # the latent space to behave like a VAE rather than a plain autoencoder.
        loss = reconstruction + 0.001 * kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % max(1, total // 5) == 0:
            print(f"  [vae] epoch {epoch + 1}/{total}  loss={loss.item():.2f}  recon={reconstruction.item():.2f}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "seed": seed, "epochs": epochs}, checkpoint)


def array_to_tensor(array: np.ndarray, torch_module):
    tensor = torch_module.tensor(array / 255.0, dtype=torch_module.float32)
    return tensor.permute(2, 0, 1).unsqueeze(0)


def build_flag_vae(torch_module, in_hw: tuple[int, int], latent: int = 64):
    """Create a tiny convolutional VAE sized for the requested flag canvas."""

    nn = torch_module.nn

    class FlagVAE(nn.Module):
        def __init__(self):
            super().__init__()
            height, width = in_hw
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 4, 2, 1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.ReLU(),
            )
            self.feature_h = height // 8
            self.feature_w = width // 8
            flat = 128 * self.feature_h * self.feature_w
            self.fc_mu = nn.Linear(flat, latent)
            self.fc_logvar = nn.Linear(flat, latent)
            self.fc_decode = nn.Linear(latent, flat)
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 3, 4, 2, 1),
                nn.Sigmoid(),
            )

        def encode(self, x):
            hidden = self.encoder(x).flatten(1)
            return self.fc_mu(hidden), self.fc_logvar(hidden)

        def decode(self, z):
            hidden = self.fc_decode(z).view(-1, 128, self.feature_h, self.feature_w)
            return self.decoder(hidden)

        def forward(self, x):
            mu, logvar = self.encode(x)
            # During training we sample so the model learns a distribution. At
            # inference time VAEBackend.encode uses mu directly for stability.
            z = mu + torch_module.randn_like(mu) * torch_module.exp(0.5 * logvar)
            return self.decode(z), mu, logvar

    return FlagVAE()
