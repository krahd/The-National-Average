#!/usr/bin/env python3
"""Render the NeurIPS 2026 edition of The National Average."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tna.video.neurips import render_neurips_video
from tna.video.pipeline import VideoRenderConfig

DEFAULT_PRESET_FILE = Path("presets/video-2026.json")
DEFAULT_OUT_DIR = Path("outputs/video/the_national_average_neurips_2026")


def load_neurips_preset(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    preset = payload.get("neurips")
    if not isinstance(preset, dict):
        raise SystemExit(f"Missing neurips preset in {path}")
    return preset


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the NeurIPS 2026 edition.")
    parser.add_argument("--preset-file", type=Path, default=DEFAULT_PRESET_FILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--foundation", choices=["required", "auto", "off"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    p = load_neurips_preset(args.preset_file)
    config = VideoRenderConfig(
        out_dir=args.out_dir,
        width=args.width if args.width is not None else int(p["width"]),
        height=args.height if args.height is not None else int(p["height"]),
        fps=args.fps if args.fps is not None else int(p["fps"]),
        duration=args.duration if args.duration is not None else float(p["duration"]),
        foundation=args.foundation if args.foundation is not None else str(p["foundation"]),
        seed=args.seed if args.seed is not None else int(p["seed"]),
        keep_frames=args.keep_frames,
        preset="neurips",
        audio=bool(p.get("audio", True)) and not args.no_audio,
        provenance_overlay=False,
    )
    print(json.dumps(render_neurips_video(config), indent=2))


if __name__ == "__main__":
    main()
