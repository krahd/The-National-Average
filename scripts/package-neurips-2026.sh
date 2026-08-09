#!/usr/bin/env bash
set -euo pipefail

# Package the NeurIPS-specific edition. The canonical video source and preserved
# production baseline are never altered.

INPUT="${1:-outputs/video/the_national_average_neurips_2026/the_national_average_neurips_2026.mp4}"
OUT_DIR="${2:-outputs/submissions/neurips-2026}"
mkdir -p "$OUT_DIR"

if [[ ! -f "$INPUT" ]]; then
  echo "Input video not found: $INPUT" >&2
  echo "Render the NeurIPS edition first:" >&2
  echo "  python scripts/render_neurips_2026.py" >&2
  exit 1
fi

DURATION="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$INPUT")"
python - "$DURATION" <<'PY'
import sys
v=float(sys.argv[1])
if v > 180.0:
    raise SystemExit(f"NeurIPS source exceeds 3 minutes: {v:.3f}s")
if v < 1:
    raise SystemExit(f"Invalid source duration: {v:.3f}s")
PY

PREVIEW="$OUT_DIR/the-national-average-preview.mp4"
THUMB="$OUT_DIR/the-national-average-thumbnail.png"
CONTACT="$OUT_DIR/the-national-average-contact-sheet.jpg"

ffmpeg -y -i "$INPUT" \
  -c:v libx264 -preset slow -profile:v high -pix_fmt yuv420p \
  -b:v 3500k -maxrate 4200k -bufsize 8400k \
  -c:a aac -b:a 192k -movflags +faststart \
  "$PREVIEW"

# Default thumbnail: the late weighting/erasure region. The contact sheet exists
# specifically so this choice can be replaced after aesthetic inspection.
ffmpeg -y -ss 00:01:48 -i "$INPUT" -frames:v 1 "$THUMB"

# Twelve-frame visual audit across the complete work. 176 seconds / 12 ≈ 14.7.
ffmpeg -y -i "$INPUT" -vf "fps=1/14.6667,scale=480:-1,tile=4x3" -frames:v 1 "$CONTACT"

PREVIEW_DURATION="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$PREVIEW")"
python - "$INPUT" "$PREVIEW" "$THUMB" "$CONTACT" "$DURATION" "$PREVIEW_DURATION" <<'PY'
import hashlib
import json
import pathlib
import subprocess
import sys

source = pathlib.Path(sys.argv[1])
preview = pathlib.Path(sys.argv[2])
thumb = pathlib.Path(sys.argv[3])
contact = pathlib.Path(sys.argv[4])
source_duration = float(sys.argv[5])
preview_duration = float(sys.argv[6])
limit = 100 * 1024 * 1024
if preview_duration > 180.0:
    raise SystemExit(f"Encoded preview exceeds 3 minutes: {preview_duration:.3f}s")
for p in (preview, thumb):
    size = p.stat().st_size
    print(f"{p.name}: {size / 1024 / 1024:.2f} MiB")
    if size >= limit:
        raise SystemExit(f"{p.name} is >=100 MiB")

probe = subprocess.check_output([
    "ffprobe", "-v", "error", "-show_entries",
    "stream=codec_type,codec_name,width,height,r_frame_rate",
    "-of", "json", str(preview),
], text=True)
streams = json.loads(probe).get("streams", [])
if not any(s.get("codec_type") == "video" for s in streams):
    raise SystemExit("Packaged preview has no video stream")

manifest = {
    "submission": "NeurIPS 2026 Creative AI",
    "artwork": "The National Average",
    "source_video": str(source),
    "source_duration_seconds": source_duration,
    "preview_duration_seconds": preview_duration,
    "files": {},
}
for p in (preview, thumb, contact):
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest["files"][p.name] = {
        "bytes": p.stat().st_size,
        "sha256": digest,
    }
preview.parent.joinpath("media-manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
PY

printf '\nNeurIPS media package created in %s\n' "$OUT_DIR"
printf 'Inspect %s and replace the default thumbnail if another frame is stronger.\n' "$CONTACT"
