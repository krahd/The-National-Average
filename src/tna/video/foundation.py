"""Foundation-model asset preparation for the ECCV video renderer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CLIP_MODEL = "openai/clip-vit-base-patch32"
SDVAE_MODEL = "stabilityai/sd-vae-ft-mse"


@dataclass(frozen=True)
class FoundationAssetRequest:
    """One Hugging Face model that must be available for production renders."""

    key: str
    repo_id: str
    loader: str


FOUNDATION_ASSETS = (
    FoundationAssetRequest("clip", CLIP_MODEL, "transformers.CLIPModel/CLIPProcessor"),
    FoundationAssetRequest("sdvae", SDVAE_MODEL, "diffusers.AutoencoderKL"),
)


class FoundationAssetError(RuntimeError):
    """Raised when a required model cannot be downloaded or verified."""


def _snapshot_revision(path: Path) -> str | None:
    # snapshot_download returns .../snapshots/<commit>. The directory name is
    # the most stable revision identifier available without hitting the network.
    if path.name and len(path.name) >= 8:
        return path.name
    return None


def _load_clip(snapshot: Path) -> None:
    from transformers import CLIPModel, CLIPProcessor

    CLIPModel.from_pretrained(snapshot)
    CLIPProcessor.from_pretrained(snapshot)


def _load_sdvae(snapshot: Path) -> None:
    from diffusers import AutoencoderKL

    AutoencoderKL.from_pretrained(snapshot)


def _verify_asset(request: FoundationAssetRequest, snapshot: Path) -> dict[str, Any]:
    if request.key == "clip":
        _load_clip(snapshot)
    elif request.key == "sdvae":
        _load_sdvae(snapshot)
    else:
        raise FoundationAssetError(f"unknown foundation asset: {request.key}")
    return {
        "key": request.key,
        "repo_id": request.repo_id,
        "loader": request.loader,
        "local_path": str(snapshot),
        "resolved_revision": _snapshot_revision(snapshot),
        "available": True,
        "error": None,
    }


def prepare_foundation_assets(
    out_dir: Path,
    *,
    required: bool = True,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Download and verify the CLIP and SD-VAE assets used by the video.

    The model files stay in the standard Hugging Face cache. This function only
    writes a small manifest into the production output directory.
    """

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - depends on optional deps
        raise FoundationAssetError(
            "huggingface_hub is required to prepare ECCV foundation assets; "
            "install the foundation optional dependencies."
        ) from exc

    provenance_dir = out_dir / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = provenance_dir / "model_manifest.json"

    models = []
    errors = []
    for request in FOUNDATION_ASSETS:
        try:
            snapshot = Path(
                snapshot_download(
                    repo_id=request.repo_id,
                    local_files_only=local_files_only,
                )
            )
            models.append(_verify_asset(request, snapshot))
        except Exception as exc:
            message = f"{request.repo_id}: {exc}"
            errors.append(message)
            models.append(
                {
                    "key": request.key,
                    "repo_id": request.repo_id,
                    "loader": request.loader,
                    "local_path": None,
                    "resolved_revision": None,
                    "available": False,
                    "error": str(exc),
                }
            )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "required": required,
        "local_files_only": local_files_only,
        "models": models,
        "all_available": not errors,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if errors and required:
        raise FoundationAssetError(
            "Required foundation assets could not be prepared. "
            "See provenance/model_manifest.json. Errors: " + " | ".join(errors)
        )
    return manifest
