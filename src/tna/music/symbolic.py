"""Symbolic music averaging and WAV rendering."""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

from ..data import Polity


MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def weighted_choice_distribution(items: dict[object, float]) -> tuple[object, dict[str, float]]:
    """Choose the highest-weight categorical value and keep its distribution."""

    total = sum(items.values()) or 1.0
    distribution = {str(key): value / total for key, value in items.items()}
    choice = max(distribution.items(), key=lambda item: item[1])[0]
    return choice, distribution


def midi_from_degree(root_midi: int, mode: str, degree: int) -> int:
    scale = MINOR_SCALE if mode == "minor" else MAJOR_SCALE
    return root_midi + scale[(degree - 1) % 7]


def midi_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{(midi // 12) - 1}"


def midi_to_frequency(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def averaged_music_profile(selected: list[Polity], weights: dict[str, float], target_duration: float) -> dict[str, object]:
    """Average symbolic musical descriptors into an inspectable score plan.

    Continuous values such as tempo are averaged directly. Categorical values
    such as mode keep a full distribution and use the highest-weight category.
    Motif scale degrees and durations are averaged step-by-step.
    """

    tempo = sum(weights[polity.code] * polity.music.tempo_bpm for polity in selected)
    root_midi = int(round(sum(weights[polity.code] * polity.music.root_midi for polity in selected)))
    max_len = max(len(polity.music.motif_degrees) for polity in selected)

    mode_weights: dict[str, float] = {}
    meter_weights: dict[int, float] = {}
    for polity in selected:
        weight = weights[polity.code]
        mode_weights[polity.music.mode] = mode_weights.get(polity.music.mode, 0.0) + weight
        meter_weights[polity.music.meter_beats] = meter_weights.get(polity.music.meter_beats, 0.0) + weight

    mode, mode_distribution = weighted_choice_distribution(mode_weights)
    meter, meter_distribution = weighted_choice_distribution(meter_weights)

    averaged_degrees: list[int] = []
    averaged_durations: list[float] = []
    for index in range(max_len):
        # Repeating shorter motifs lets all selected profiles contribute to
        # every generated note without inventing hidden interpolation rules.
        degree_value = 0.0
        duration_value = 0.0
        for polity in selected:
            profile = polity.music
            weight = weights[polity.code]
            degree_value += weight * profile.motif_degrees[index % len(profile.motif_degrees)]
            duration_value += weight * profile.motif_durations[index % len(profile.motif_durations)]
        averaged_degrees.append(int(min(7, max(1, round(degree_value)))))
        averaged_durations.append(max(0.25, round(duration_value * 4) / 4))

    beat_seconds = 60.0 / tempo
    notes = []
    elapsed = 0.0
    index = 0
    while elapsed < target_duration - 0.05:
        degree = averaged_degrees[index % len(averaged_degrees)]
        beats = averaged_durations[index % len(averaged_durations)]
        seconds = min(beats * beat_seconds, target_duration - elapsed)
        midi = midi_from_degree(root_midi, str(mode), degree)
        notes.append(
            {
                "index": index + 1,
                "degree": degree,
                "beats": beats,
                "start_seconds": round(elapsed, 4),
                "duration_seconds": round(seconds, 4),
                "midi": midi,
                "pitch": midi_name(midi),
                "frequency_hz": round(midi_to_frequency(midi), 3),
            }
        )
        elapsed += seconds
        index += 1

    return {
        "backend": "symbolic",
        "space": "tempo/mode/meter/motif symbolic profile",
        "tempo_bpm": round(tempo, 3),
        "root_midi": root_midi,
        "root_pitch": midi_name(root_midi),
        "mode": mode,
        "mode_distribution": mode_distribution,
        "meter_beats": int(meter),
        "meter_distribution": meter_distribution,
        "motif_degrees": averaged_degrees,
        "motif_durations_beats": averaged_durations,
        "target_duration_seconds": target_duration,
        "actual_duration_seconds": round(elapsed, 4),
        "notes": notes,
    }


def render_wav(notes: list[dict[str, object]], path: Path, sample_rate: int) -> None:
    """Render a simple monophonic WAV using only the Python standard library."""

    samples: list[int] = []
    amplitude = 0.28
    for note in notes:
        frequency = float(note["frequency_hz"])
        sample_count = max(1, int(round(float(note["duration_seconds"]) * sample_rate)))
        attack = max(1, int(0.012 * sample_rate))
        release = max(1, int(0.035 * sample_rate))
        for sample_index in range(sample_count):
            time = sample_index / sample_rate
            envelope = 1.0
            if sample_index < attack:
                envelope = sample_index / attack
            elif sample_index > sample_count - release:
                envelope = max(0.0, (sample_count - sample_index) / release)
            # A few low harmonics make the tone easier to hear while keeping
            # synthesis deterministic and inspectable.
            value = math.sin(2 * math.pi * frequency * time)
            value += 0.22 * math.sin(2 * math.pi * frequency * 2 * time)
            value += 0.08 * math.sin(2 * math.pi * frequency * 3 * time)
            samples.append(int(max(-1.0, min(1.0, value * amplitude * envelope)) * 32767))

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def write_score_trace(path: Path, intent: str, weights: dict[str, float], selected: list[Polity], music: dict[str, object]) -> None:
    """Write a human-readable score trace beside the machine JSON trace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Critical Averaging symbolic score trace",
        "",
        f"intent: {intent}",
        f"tempo_bpm: {music['tempo_bpm']}",
        f"mode: {music['mode']}",
        f"meter: {music['meter_beats']}/4",
        f"root: {music['root_pitch']}",
        "",
        "weights:",
    ]
    for polity in selected:
        lines.append(f"- {polity.code} {polity.name}: {weights[polity.code]:.6f}")
    lines.extend(["", "notes:"])
    for note in music["notes"]:
        lines.append(
            "- {index:02d}. {pitch:>3s} degree={degree} beats={beats} "
            "start={start_seconds:.3f}s duration={duration_seconds:.3f}s freq={frequency_hz:.3f}Hz".format(**note)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
