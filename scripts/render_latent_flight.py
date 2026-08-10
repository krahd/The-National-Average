#!/usr/bin/env python3
"""Render the independent 3-D latent-flight edition of The National Average."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tna.video.latent_flight import render_latent_flight
from tna.video.pipeline import VideoRenderConfig


DEFAULT_PRESET_FILE = Path("presets/latent-flight-2026.json")
DEFAULT_OUT_DIR = Path("outputs/video/the_national_average_latent_world")


def load_preset(path: Path, name: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    preset = payload.get(name)
    if not isinstance(preset, dict):
        raise SystemExit(f"Missing {name!r} preset in {path}")
    return preset


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the 3-D latent-flight edition.")
    parser.add_argument("--preset", choices=("preview", "production"), default="preview")
    parser.add_argument("--preset-file", type=Path, default=DEFAULT_PRESET_FILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--foundation", choices=("required", "auto", "off"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    preset = load_preset(args.preset_file, args.preset)
    config = VideoRenderConfig(
        out_dir=args.out_dir,
        width=args.width if args.width is not None else int(preset["width"]),
        height=args.height if args.height is not None else int(preset["height"]),
        fps=args.fps if args.fps is not None else int(preset["fps"]),
        duration=args.duration if args.duration is not None else float(preset["duration"]),
        foundation=args.foundation if args.foundation is not None else str(preset["foundation"]),
        seed=args.seed if args.seed is not None else int(preset["seed"]),
        keep_frames=args.keep_frames,
        preset=args.preset,
        audio=bool(preset.get("audio", True)) and not args.no_audio,
        provenance_overlay=False,
    )
    print(json.dumps(render_latent_flight(config), indent=2))


if __name__ == "__main__":
    main()
