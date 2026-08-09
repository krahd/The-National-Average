#!/usr/bin/env bash
set -euo pipefail

# Package the strongest current The National Average render for the NeurIPS 2026
# Creative AI artwork submission. This script never alters the canonical artwork
# source or the preserved production baseline.

INPUT="${1:-outputs/video/the_national_average/the_national_average.mp4}"
OUT_DIR="${2:-outputs/submissions/neurips-2026}"
mkdir -p "$OUT_DIR"

if [[ ! -f "$INPUT" ]]; then
  echo "Input video not found: $INPUT" >&2
  echo "Render the current production baseline first:" >&2
  echo "  python scripts/render_video.py --preset production" >&2
  exit 1
fi

DURATION="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$INPUT")"
python - "$DURATION" <<'PY'
import sys
v=float(sys.argv[1])
if v > 180.05:
    raise SystemExit(f"NeurIPS preview exceeds 3 minutes: {v:.3f}s")
PY

# Conservative bitrate budget: comfortably below the 100 MB upload ceiling for
# a three-minute work while retaining high visual quality for screen display.
ffmpeg -y -i "$INPUT" \
  -c:v libx264 -preset slow -profile:v high -pix_fmt yuv420p \
  -b:v 3500k -maxrate 4200k -bufsize 8400k \
  -c:a aac -b:a 192k -movflags +faststart \
  "$OUT_DIR/the-national-average-preview.mp4"

# Extract a late-middle frame rather than the opening explanatory material.
ffmpeg -y -ss 00:01:40 -i "$INPUT" -frames:v 1 \
  "$OUT_DIR/the-national-average-thumbnail.png"

python - "$OUT_DIR/the-national-average-preview.mp4" <<'PY'
import os, sys
p=sys.argv[1]
size=os.path.getsize(p)
limit=100*1024*1024
print(f"Preview: {size/1024/1024:.2f} MiB")
if size >= limit:
    raise SystemExit("Preview is >=100 MiB; reduce bitrate before submission")
PY

printf '\nNeurIPS media package created in %s\n' "$OUT_DIR"
printf 'Manual remaining step: choose/verify the thumbnail aesthetically and export the <=3-page PDF.\n'
