"""Optional symbolic music VAE scaffold."""

from __future__ import annotations


def unavailable_trace() -> dict[str, object]:
    # A music VAE would need a real symbolic anthem corpus. The current music
    # backend uses hand-authored profiles, so the honest behaviour is a trace
    # explaining why no latent model was trained.
    return {
        "backend": "music_vae",
        "space": "symbolic music latent",
        "status": "unavailable",
        "reason": "The corpus contains hand-authored example profiles, not enough symbolic anthem data to train a music VAE.",
    }
