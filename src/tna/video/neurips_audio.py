"""Restrained soundtrack for the NeurIPS edition of *The National Average*.

The sound remains procedurally generated and uses real analysis-derived scalars,
but it avoids the dense glitch/supersaw character of the preserved baseline.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .neurips import neurips_segments

SR = 48_000


def _lowpass(sig: np.ndarray, cutoff: float) -> np.ndarray:
    n = len(sig)
    f = np.fft.rfftfreq(n, 1 / SR)
    h = 1.0 / (1.0 + (f / max(cutoff, 1.0)) ** 4)
    return np.fft.irfft(np.fft.rfft(sig) * h, n)


def _place(buf: np.ndarray, at: float, sig: np.ndarray, gain: float = 1.0) -> None:
    a = max(0, int(at * SR))
    b = min(len(buf), a + len(sig))
    if b > a:
        buf[a:b] += sig[: b - a] * gain


def _pulse(freq: float, duration: float = 0.8) -> np.ndarray:
    n = max(1, int(duration * SR))
    t = np.arange(n) / SR
    env = np.exp(-t * 4.0)
    return (np.sin(2 * np.pi * freq * t) + 0.18 * np.sin(2 * np.pi * freq * 2.01 * t)) * env


def render_neurips_soundtrack(assets, config) -> Path:
    rng = np.random.default_rng(config.seed + 2026)
    duration = config.duration
    n = int(duration * SR)
    t = np.arange(n) / SR

    era = assets.erasure.get("cumulative_co2")
    erased_frac = era.erased_count / max(1, era.total) if era else 0.5

    # Low continuous air/traffic-like bed: deliberately synthetic, never claimed
    # as field recording or as a measurement.
    noise_l = _lowpass(rng.normal(0, 1, n), 170.0)
    noise_r = _lowpass(rng.normal(0, 1, n), 190.0)
    noise_l /= np.max(np.abs(noise_l)) or 1.0
    noise_r /= np.max(np.abs(noise_r)) or 1.0

    root = 36.71
    phase = 2 * np.pi * np.cumsum(np.full(n, root)) / SR
    sub = np.sin(phase * 0.5) * 0.11 + np.sin(phase) * 0.045
    slow = 0.72 + 0.12 * np.sin(2 * np.pi * 0.018 * t)
    left = noise_l * 0.055 * slow + sub
    right = noise_r * 0.055 * np.roll(slow, 900) + np.roll(sub, 137)

    # Sparse structural accents; only representational regime changes and erasure
    # receive explicit sonic punctuation.
    for key, start, end in neurips_segments(duration):
        phase_dur = end - start
        if key in {"spaces", "weighting", "average"}:
            tone = _pulse(48.0 if key == "spaces" else 42.0, 1.2)
            _place(left, start + phase_dur * 0.08, tone, 0.12)
            _place(right, start + phase_dur * 0.08 + 0.025, tone, 0.12)
        elif key == "erase":
            count = max(3, min(9, int(3 + erased_frac * 7)))
            for i in range(count):
                at = start + (i + 0.5) / count * phase_dur
                tone = _pulse(31.0 - i * 0.7, 0.65)
                _place(left, at, tone, 0.14)
                _place(right, at + 0.018, tone, 0.14)

    # Long fade into the unresolved coda.
    fade_in = np.clip(t / 3.0, 0.0, 1.0)
    fade_out = np.clip((duration - t) / 8.0, 0.0, 1.0)
    env = fade_in * fade_out
    left *= env
    right *= env

    peak = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1e-8)
    stereo = np.stack([left / peak * 0.68, right / peak * 0.68], axis=1)
    pcm = (np.clip(stereo, -1.0, 1.0) * 32767).astype("<i2")

    path = config.out_dir / "audio" / "the_national_average_neurips_2026.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(pcm.tobytes())
    return path
