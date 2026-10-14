#!/usr/bin/env python3
"""Sample every narrative phase of the portrait edition at its true timeline."""

from __future__ import annotations

import json
from pathlib import Path

from tna.video.foundation import prepare_foundation_assets
from tna.video.latent_flight import LatentFlightRenderer, build_latent_flight_assets
from tna.video.pipeline import VideoRenderConfig


def main() -> None:
    preset_path = Path("presets/we-the-artists-portrait-2026.json")
    preset = json.loads(preset_path.read_text(encoding="utf-8"))["production"]
    output = Path("outputs/video/we-the-artists-portrait-diagnostic")
    output.mkdir(parents=True, exist_ok=True)
    config = VideoRenderConfig(
        out_dir=output,
        width=540,
        height=960,
        fps=int(preset["fps"]),
        duration=float(preset["duration"]),
        foundation="required",
        seed=int(preset["seed"]),
        keep_frames=True,
        preset="preview",
        audio=False,
        provenance_overlay=False,
    )
    models = prepare_foundation_assets(output, required=True, local_files_only=True)
    assets = build_latent_flight_assets(config, models)
    renderer = LatentFlightRenderer(config, assets)
    # Title, introduction, all nine chapters, five-average reveal and both propositions.
    times = [3, 11, 17, 25, 32, 39, 47, 55, 63, 70, 78, 83, 87, 91, 94]
    for second in times:
        destination = output / f"portrait_{second:02d}s.png"
        renderer.render(round(second * config.fps)).save(destination)
        print(destination, flush=True)


if __name__ == "__main__":
    main()
