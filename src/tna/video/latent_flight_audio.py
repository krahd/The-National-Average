"""Procedural industrial soundtrack for the latent-flight film.

The changing focus weights and real PCA coordinate energies modulate an artistic
sound design.  They are not presented as measurements or national sonifications.
The mix keeps proposing a stable drone, then dislocating it with detuning, cuts,
buffer-like repeats, and short noise ruptures.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .latent_flight import FOCUS_CODES, film_phase_segments, flight_segments, parameter_weights


SR = 48_000


def _place(buffer: np.ndarray, sample: int, signal: np.ndarray, gain: float = 1.0) -> None:
    start = max(0, sample)
    end = min(len(buffer), start + len(signal))
    if end > start:
        buffer[start:end] += signal[: end - start] * gain


def _smoothed_noise(rng: np.random.Generator, count: int, window: int) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, count + window).astype(np.float32)
    cumulative = np.cumsum(noise, dtype=np.float64)
    smooth = (cumulative[window:] - cumulative[:-window]) / np.sqrt(window)
    return smooth[:count].astype(np.float32)


def _weight_curves(duration: float, count: int) -> np.ndarray:
    samples = max(512, int(duration * 18))
    coarse_t = np.linspace(0.0, 1.0, samples, endpoint=True)
    coarse = np.stack([parameter_weights(float(value)) for value in coarse_t])
    target = np.linspace(0.0, 1.0, count, endpoint=True)
    return np.stack(
        [np.interp(target, coarse_t, coarse[:, i]) for i in range(4)],
        axis=1,
    ).astype(np.float32)


def render_latent_flight_soundtrack(assets, config) -> Path:
    rng = np.random.default_rng(config.seed + 0x1A7E17)
    count = max(1, int(config.duration * SR))
    time = np.arange(count, dtype=np.float32) / SR
    weights = _weight_curves(config.duration, count)

    energies = np.array(
        [np.linalg.norm(assets.focus_latents[code][:8]) for code in FOCUS_CODES],
        dtype=np.float64,
    )
    energies /= energies.max() or 1.0
    roots = 31.0 + energies * np.array((10.0, 15.0, 7.0, 18.0))
    root_curve = np.sum(weights * roots[None, :], axis=1).astype(np.float32)

    # A near-stable centre whose phase is constantly bent by the moving average.
    drift = (
        0.31 * np.sin(2 * np.pi * time * 0.071)
        + 0.19 * np.sin(2 * np.pi * time * 0.137 + 1.7)
        + (weights[:, 0] - weights[:, 1]) * 1.9
    ).astype(np.float32)
    phase = (2 * np.pi / SR * np.cumsum(root_curve + drift, dtype=np.float64)).astype(np.float32)
    sub = np.sin(phase * 0.5) * 0.18 + np.sin(phase) * 0.085
    pressure = np.sin(phase * 1.997 + np.sin(time * 0.19) * 1.3) * 0.035

    low_noise = _smoothed_noise(rng, count, 420)
    low_noise /= np.max(np.abs(low_noise)) or 1.0
    metal_noise = rng.normal(0, 1, count).astype(np.float32)
    metal_noise -= np.roll(metal_noise, 1)
    metal_noise *= 0.010 + 0.014 * (weights[:, 2] + weights[:, 3])

    left = (sub + pressure + low_noise * 0.042 + metal_noise).astype(np.float32)
    right = (
        np.roll(sub, 193)
        + np.roll(pressure, -311)
        + np.roll(low_noise, 857) * 0.042
        - np.roll(metal_noise, 37)
    ).astype(np.float32)

    # Nervous, non-metric impacts. The interval contracts toward calm, but each
    # arrival is displaced by the current parameter field.
    at = 0.72
    pulse_index = 0
    while at < config.duration:
        local_weight = parameter_weights(at / max(config.duration, 1e-9))
        interval = 0.22 + 0.27 * float(local_weight[pulse_index % 4])
        length = int(SR * (0.055 + 0.08 * float(local_weight[(pulse_index + 1) % 4])))
        tt = np.arange(length, dtype=np.float32) / SR
        frequency = 47.0 + energies[pulse_index % 4] * 56.0
        envelope = np.exp(-tt * (28.0 + pulse_index % 5)).astype(np.float32)
        hit = (
            np.sin(2 * np.pi * frequency * tt + 10.0 * np.exp(-tt * 55.0))
            + rng.normal(0, 0.7, length).astype(np.float32)
        ) * envelope
        sample = int(at * SR)
        _place(left, sample, hit, 0.085)
        _place(right, sample + int((0.004 + 0.011 * local_weight[3]) * SR), hit, 0.075)
        at += interval
        pulse_index += 1

    # Every edit tears the audio buffer. These cuts correspond exactly to the
    # visual hard-cut schedule, while their spectra come from deterministic noise.
    film_phases = film_phase_segments(config.duration)
    search_start, search_end = film_phases[2][1], film_phases[2][2]
    segments = [
        (key, start + search_start, end + search_start)
        for key, start, end in flight_segments(search_end - search_start)
    ]
    for index, (_, start, _) in enumerate(segments[1:], start=1):
        burst_duration = 0.09 + 0.025 * (index % 4)
        n = int(burst_duration * SR)
        tt = np.arange(n, dtype=np.float32) / SR
        tear = rng.normal(0, 1, n).astype(np.float32)
        tear *= np.exp(-tt * (18.0 + index % 3)).astype(np.float32)
        tear *= 0.14
        sample = int(start * SR)
        _place(left, sample, tear)
        _place(right, sample + (index % 5) * 91, -tear if index % 2 else tear)
        if index % 3 == 0:
            repeat = tear[: max(1, n // 5)].copy()
            for repeat_index in range(5):
                offset = sample + int((0.12 + repeat_index * 0.043) * SR)
                _place(left, offset, repeat, 0.72 - repeat_index * 0.09)
                _place(right, offset + 127, repeat[::-1], 0.60 - repeat_index * 0.07)

    # The late false calm is conspicuously clean, but two almost-unison tones
    # continue sliding past each other and are broken by the final escape cut.
    calm_segment = next((item for item in segments if item[0] == "almost_rest"), None)
    if calm_segment:
        _, start, end = calm_segment
        a, b = int(start * SR), min(count, int(end * SR))
        local_t = np.arange(b - a, dtype=np.float32) / SR
        sine_window = np.clip(
            np.sin(np.linspace(0, np.pi, b - a, dtype=np.float32)),
            0.0,
            1.0,
        )
        window = sine_window ** 1.5
        detune = 0.18 + 0.12 * np.sin(local_t * 0.7)
        calm_l = np.sin(2 * np.pi * (73.4 - detune) * local_t) * window * 0.115
        calm_r = np.sin(2 * np.pi * (73.4 + detune) * local_t) * window * 0.115
        left[a:b] = left[a:b] * (1.0 - window * 0.62) + calm_l
        right[a:b] = right[a:b] * (1.0 - window * 0.62) + calm_r

    fade_in = np.clip(time / 1.2, 0.0, 1.0)
    fade_out = np.clip((config.duration - time) / 0.22, 0.0, 1.0)
    envelope = fade_in * fade_out
    left *= envelope
    right *= envelope

    peak = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1e-8)
    stereo = np.stack((left / peak * 0.78, right / peak * 0.78), axis=1)
    pcm = (np.clip(stereo, -1.0, 1.0) * 32767).astype("<i2")
    path = config.out_dir / "audio" / "the_national_average_latent_flight.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(pcm.tobytes())
    return path
