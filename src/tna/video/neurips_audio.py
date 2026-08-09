"""Restrained soundtrack for the NeurIPS edition of *The National Average*.

The sound is procedural and structurally coupled to the same weighting records shown
by the film. Concentration narrows the spectral field; erasure changes event density;
the equal Israel/Palestine CLIP margin becomes the interval between two tones.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .neurips import neurips_segments
from .neurips_metrics import count_below, stats_for
from .pipeline import SUBJECT_INTENT

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

    stats = {intent: stats_for(intent, run.weights) for intent, run in assets.weights.items()}
    subject_run = assets.weights.get(SUBJECT_INTENT)
    subject_stats = stats.get(SUBJECT_INTENT)
    entropy = subject_stats.normalised_entropy if subject_stats else 0.5
    concentration = 1.0 - entropy

    cutoff_l = 115.0 + entropy * 120.0
    cutoff_r = cutoff_l * 1.08
    noise_l = _lowpass(rng.normal(0, 1, n), cutoff_l)
    noise_r = _lowpass(rng.normal(0, 1, n), cutoff_r)
    noise_l /= np.max(np.abs(noise_l)) or 1.0
    noise_r /= np.max(np.abs(noise_r)) or 1.0

    root = 34.0 - concentration * 5.0
    phase = 2 * np.pi * np.cumsum(np.full(n, root)) / SR
    sub = np.sin(phase * 0.5) * 0.105 + np.sin(phase) * 0.04
    slow = 0.70 + 0.11 * np.sin(2 * np.pi * 0.016 * t)
    left = noise_l * 0.052 * slow + sub
    right = noise_r * 0.052 * np.roll(slow, 900) + np.roll(sub, 137)

    for key, start, end in neurips_segments(duration):
        phase_dur = end - start
        if key == "distribution":
            intents = [intent for intent in assets.weights if intent in stats]
            for i, intent in enumerate(intents):
                s = stats[intent]
                at = start + (i + 0.5) / max(1, len(intents)) * phase_dur
                tone = _pulse(30.0 + s.normalised_entropy * 22.0, 0.9)
                _place(left, at, tone, 0.10)
                _place(right, at + 0.021, tone, 0.10)
        elif key in {"spaces", "weighting", "average"}:
            tone = _pulse(45.0 if key == "spaces" else 39.0, 1.0)
            _place(left, start + phase_dur * 0.08, tone, 0.085)
            _place(right, start + phase_dur * 0.08 + 0.025, tone, 0.085)
        elif key == "matrix":
            # Two axes are made audible as two sparse, interlocking pulse grids.
            for i in range(5):
                _place(left, start + (i + 0.5) / 5 * phase_dur, _pulse(43.0, 0.42), 0.065)
            for i in range(6):
                _place(right, start + (i + 0.5) / 6 * phase_dur, _pulse(47.0, 0.36), 0.055)
        elif key == "thresholds" and subject_run is not None:
            removed = count_below(subject_run.weights, 0.001)
            count = max(3, min(12, round(3 + removed / max(1, len(subject_run.weights)) * 10)))
            for i in range(count):
                at = start + (i + 0.5) / count * phase_dur
                tone = _pulse(30.0 - i * 0.32, 0.55)
                _place(left, at, tone, 0.10)
                _place(right, at + 0.018, tone, 0.10)
        elif key == "erase":
            era = assets.erasure.get(SUBJECT_INTENT)
            erased_frac = era.erased_count / max(1, era.total) if era else 0.5
            count = max(3, min(9, int(3 + erased_frac * 7)))
            for i in range(count):
                at = start + (i + 0.5) / count * phase_dur
                tone = _pulse(28.0 - i * 0.5, 0.6)
                _place(left, at, tone, 0.11)
                _place(right, at + 0.018, tone, 0.11)
        elif key == "pair":
            rec = assets.recognition_tie
            margin = abs(float(rec.margin)) if rec is not None else 0.0
            # A zero/near-zero retrieval margin produces near-unison. The mapping is
            # intentionally simple and documented as sonification, not measurement.
            separation = min(5.0, margin * 180.0)
            centre = 41.0
            a = _pulse(centre - separation / 2.0, 2.2)
            b = _pulse(centre + separation / 2.0, 2.2)
            at = start + phase_dur * 0.38
            _place(left, at, a, 0.13)
            _place(right, at, b, 0.13)

    fade_in = np.clip(t / 3.0, 0.0, 1.0)
    fade_out = np.clip((duration - t) / 8.0, 0.0, 1.0)
    env = fade_in * fade_out
    left *= env
    right *= env

    peak = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1e-8)
    stereo = np.stack([left / peak * 0.62, right / peak * 0.62], axis=1)
    pcm = (np.clip(stereo, -1.0, 1.0) * 32767).astype("<i2")

    path = config.out_dir / "audio" / "the_national_average_neurips_2026.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(pcm.tobytes())
    return path
