"""Dark-ambient synthesiser soundtrack for *The Average Nation*.

A thick, multi-band drone built in numpy: heavy sub-bass, detuned-unison supersaw
pads (low + mid), a high noise shimmer, slow FFT-filtered movement, a reverb tail,
and layered glitch textures — with stark accents at the collapse, the erasure, and
the naming. The chord/brightness/glitch shift per pipeline phase, driven by the real
records, but the bed stays continuous, brooding, and rich. No melody/composition.
The synthesis is sound design, not a claimed measurement.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .pipeline import phase_segments

SR = 48_000

# Low, dark roots per phase (a slow brooding progression; no melody).
_PHASE_ROOT = {
    "ingest": 32.70, "segment": 34.65, "tokenize": 36.71, "attend": 38.89,
    "embed": 32.70, "eigenbasis": 41.20, "retrieve": 36.71, "reconstruct": 38.89,
    "average": 30.87, "weighting": 34.65, "erase": 27.50, "name": 32.70,
    "residual": 36.71, "coda": 27.50,
    "title": 30.87, "spaces": 38.89, "regions": 34.65,
}
_PHASE_BRIGHT = {
    "ingest": 0.30, "segment": 0.45, "tokenize": 0.50, "attend": 0.50,
    "embed": 0.62, "eigenbasis": 0.66, "retrieve": 0.55, "reconstruct": 0.60,
    "average": 0.40, "weighting": 0.52, "erase": 0.24, "name": 0.30,
    "residual": 0.42, "coda": 0.20,
    "title": 0.18, "spaces": 0.55, "regions": 0.48,
}


def _scalars(assets) -> dict[str, float]:
    era = assets.erasure.get("cumulative_co2")
    rec = assets.recognition.get("cumulative_co2")
    out = {"erased_frac": 0.5, "confidence": 0.5}
    if era:
        out["erased_frac"] = min(1.0, era.erased_count / (era.total or 1))
    if rec:
        out["confidence"] = rec.confidence
    return out


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k < 2:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    sm = (c[k:] - c[:-k]) / k
    pad = k // 2
    return np.concatenate([np.full(pad, sm[0]), sm, np.full(len(x) - len(sm) - pad, sm[-1])])


def _phase_curve(duration: float, mapping: dict, default: float) -> np.ndarray:
    n = int(duration * SR)
    curve = np.full(n, default, dtype=np.float64)
    for key, s, e in phase_segments(duration):
        a, b = int(s * SR), min(n, int(e * SR))
        curve[a:b] = mapping.get(key, default)
    return _smooth(curve, int(1.2 * SR))


def _lowpass(sig: np.ndarray, cutoff: float, order: int = 4) -> np.ndarray:
    n = len(sig)
    f = np.fft.rfftfreq(n, 1 / SR)
    H = 1.0 / (1.0 + (f / max(cutoff, 1.0)) ** order)
    return np.fft.irfft(np.fft.rfft(sig) * H, n)


def _highpass(sig: np.ndarray, cutoff: float, order: int = 2) -> np.ndarray:
    n = len(sig)
    f = np.fft.rfftfreq(n, 1 / SR)
    ratio = (f / max(cutoff, 1.0)) ** order
    H = ratio / (1.0 + ratio)
    return np.fft.irfft(np.fft.rfft(sig) * H, n)


def _supersaw(root_t: np.ndarray, rng: np.random.Generator, voices: int = 7, detune: float = 0.014) -> np.ndarray:
    out = np.zeros(len(root_t))
    half = (voices - 1) / 2 or 1
    for i in range(voices):
        spread = detune * ((i - (voices - 1) / 2) / half)
        ph = np.cumsum(root_t * (1.0 + spread)) / SR + rng.random()
        out += 2.0 * (ph - np.floor(ph + 0.5))
    return out / voices


def _reverb(sig: np.ndarray, rng: np.random.Generator, decay: float = 1.8, mix: float = 0.28) -> np.ndarray:
    m = int(decay * SR)
    ir = rng.normal(0, 1, m) * np.exp(-np.linspace(0, 6, m))
    ir /= np.abs(ir).max() or 1.0
    length = len(sig) + m - 1
    wet = np.fft.irfft(np.fft.rfft(sig, length) * np.fft.rfft(ir, length), length)[: len(sig)]
    wet /= np.abs(wet).max() or 1.0
    return (1 - mix) * sig + mix * wet


def _env(nn: int, attack: float = 0.05, release: float = 0.5) -> np.ndarray:
    env = np.ones(nn)
    ai, ri = max(1, int(nn * attack)), max(1, int(nn * release))
    env[:ai] = np.linspace(0, 1, ai)
    env[-ri:] = np.linspace(1, 0, ri)
    return env


def _tone(freq: float, dur: float, attack: float, release: float) -> np.ndarray:
    nn = max(1, int(dur * SR))
    tt = np.arange(nn) / SR
    w = np.sin(2 * np.pi * freq * tt) + 0.25 * np.sin(2 * np.pi * 2 * freq * tt)
    return w * _env(nn, attack, release)


def _place(buf: np.ndarray, at: float, sig: np.ndarray, gain: float = 1.0) -> None:
    a = max(0, int(at * SR))
    b = min(len(buf), a + len(sig))
    if b > a:
        buf[a:b] += sig[: b - a] * gain


def render_soundtrack(assets, config) -> Path:
    rng = np.random.default_rng(config.seed + 777)
    duration = config.duration
    n = int(duration * SR)
    t = np.arange(n) / SR
    sc = _scalars(assets)

    root_t = _phase_curve(duration, _PHASE_ROOT, 32.70)
    bright_t = _phase_curve(duration, _PHASE_BRIGHT, 0.4)

    # Heavy sub-bass: sub-octave + fundamental sine via phase accumulation.
    sub_ph = 2 * np.pi * np.cumsum(root_t * 0.5) / SR
    fund_ph = 2 * np.pi * np.cumsum(root_t) / SR
    sub = (np.sin(sub_ph) * 0.9 + np.sin(fund_ph) * 0.5)
    sub *= 1.0 + 0.05 * np.sin(2 * np.pi * 0.03 * t)

    # Detuned-unison pads.
    low = _lowpass(_supersaw(root_t, rng, voices=7, detune=0.012), 320.0) * 0.6
    mid = _lowpass(_supersaw(root_t * 1.5, rng, voices=6, detune=0.02)
                   + _supersaw(root_t * 2.0, rng, voices=6, detune=0.02), 1100.0) * 0.30
    mid *= 0.5 + bright_t
    # High shimmer: filtered noise, gated by brightness.
    shimmer = _highpass(rng.normal(0, 1, n), 2500.0) * (0.03 * bright_t)

    left = sub * 0.55 + low + mid + shimmer
    left *= 1.0 + 0.08 * np.sin(2 * np.pi * 0.018 * t)  # slow swell
    # Stereo: re-voice the pads slightly for width.
    low_r = _lowpass(_supersaw(root_t, rng, voices=7, detune=0.013), 320.0) * 0.6
    right = sub * 0.55 + low_r + np.roll(mid, 311) + np.roll(shimmer, 173)
    right *= 1.0 + 0.08 * np.sin(2 * np.pi * 0.018 * t + 0.7)

    # Glitch textures: short band-passed noise stutters, denser when unsure.
    glitch_rate = 0.5 + 1.5 * (1.0 - sc["confidence"])
    for _ in range(int(duration * glitch_rate)):
        at = rng.random() * duration
        length = int(rng.integers(int(0.02 * SR), int(0.12 * SR)))
        burst = _highpass(rng.normal(0, 1, length), 800.0) * np.linspace(1, 0, length) * 0.06
        _place(left, at, burst)
        _place(right, at + 0.003, burst)

    # Accents tied to the real operations.
    for key, s, e in phase_segments(duration):
        dur = e - s
        if key == "average":
            _place(left, s + dur * 0.5, _tone(34, 0.9, 0.005, 0.85) * 0.22)
            _place(right, s + dur * 0.5, _tone(34, 0.9, 0.005, 0.85) * 0.22)
        elif key == "erase":
            hits = max(3, int(8 * sc["erased_frac"]))
            for i in range(hits):
                imp = _tone(46 * (1 - 0.3 * i / hits), 0.3, 0.005, 0.85) * 0.12
                _place(left, s + i / hits * dur, imp)
                _place(right, s + i / hits * dur, imp)
        elif key == "name":
            for at in (0.0, 0.4, 0.6):
                _place(left, s + at * dur, _tone(41, 0.7, 0.004, 0.85) * 0.20)
                _place(right, s + at * dur, _tone(41, 0.7, 0.004, 0.85) * 0.20)

    # Space + soft saturation, fuller master.
    left = _reverb(left, rng, decay=2.0, mix=0.26)
    right = _reverb(right, rng, decay=2.0, mix=0.26)
    peak = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1e-6)
    left = np.tanh(left / peak * 1.4) * 0.72
    right = np.tanh(right / peak * 1.4) * 0.72
    stereo = np.clip(np.stack([left, right], axis=1), -1.0, 1.0)
    pcm = (stereo * 32767).astype("<i2")
    path = config.out_dir / "audio" / "the_average_nation.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(pcm.tobytes())
    return path
