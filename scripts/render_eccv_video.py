#!/usr/bin/env python3
"""Render the ECCV 2026 video artwork ``The Average Nation``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tna.video import ECCVRenderConfig, render_eccv_video


DEFAULT_OUT_DIR = Path("outputs/eccv/the_average_nation")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render The Average Nation ECCV video package.")
    parser.add_argument("--preset", choices=["preview", "submission"], default="preview")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--foundation", choices=["required", "auto", "off"], default=None)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--keep-frames", action="store_true")
    return parser


def preset_defaults(preset: str) -> dict[str, int | float | str]:
    if preset == "submission":
        return {
            "width": 1920,
            "height": 1080,
            "fps": 24,
            "duration": 180.0,
            "foundation": "required",
        }
    return {
        "width": 960,
        "height": 540,
        "fps": 12,
        "duration": 180.0,
        "foundation": "auto",
    }


def main() -> None:
    args = build_parser().parse_args()
    defaults = preset_defaults(args.preset)
    config = ECCVRenderConfig(
        out_dir=args.out_dir,
        width=args.width or int(defaults["width"]),
        height=args.height or int(defaults["height"]),
        fps=args.fps or int(defaults["fps"]),
        duration=args.duration or float(defaults["duration"]),
        foundation=args.foundation or str(defaults["foundation"]),
        seed=args.seed,
        keep_frames=args.keep_frames,
        preset=args.preset,
    )
    result = render_eccv_video(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
