#!/usr/bin/env python3
"""Render the moving-image work for The National Average."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tna.video import VideoRenderConfig, render_video


DEFAULT_OUT_DIR = Path("outputs/video/the_national_average")
DEFAULT_PRESET_FILE = Path("presets/video-2026.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render The National Average moving-image work.")
    parser.add_argument("--preset", choices=["preview", "production", "neurips"], default="preview")
    parser.add_argument("--preset-file", type=Path, default=DEFAULT_PRESET_FILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--foundation", choices=["required", "auto", "off"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--provenance-overlay", action="store_true")
    return parser


def load_preset(path: Path, name: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get(name)
    if not isinstance(values, dict):
        raise SystemExit(f"Preset {name!r} not found in {path}")
    return values


def main() -> None:
    args = build_parser().parse_args()
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
        provenance_overlay=bool(preset.get("provenance_overlay", False)) or args.provenance_overlay,
    )
    result = render_video(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
