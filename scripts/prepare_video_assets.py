#!/usr/bin/env python3
"""Prepare the foundation-model assets used by the video renderer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tna.video.foundation import FoundationAssetError, prepare_foundation_assets


DEFAULT_OUT_DIR = Path("outputs/video/the_national_average")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and verify foundation-model assets for the video.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--local-files-only", action="store_true", help="Verify only existing Hugging Face cache files.")
    parser.add_argument("--optional", action="store_true", help="Write a manifest but do not fail if models are missing.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        manifest = prepare_foundation_assets(
  args.out_dir,
  required=not args.optional,
  local_files_only=args.local_files_only,
        )
    except FoundationAssetError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
