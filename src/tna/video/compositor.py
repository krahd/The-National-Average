"""Shared PIL compositing helpers and ffmpeg streaming for the video video.

These are presentation utilities only. They never invent measurements: classical
operations such as :func:`sobel_edges` and :func:`colour_masks` are genuine image
transforms of a real flag, and every value rendered as text by the scenes comes
from the analysis layer, not from this module.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Shared palette: a cold, institutional background and a single cyan accent.
# Foreground tones lifted toward white so text stays legible through the grain.
BG = (3, 5, 8)
ACCENT = (120, 214, 224)
EDGE = (150, 236, 244)
INK = (246, 249, 247)
MUTED = (202, 216, 216)
FAINT = (175, 198, 200)


class FrameSource(Protocol):
    def render(self, frame_index: int) -> Image.Image:
        ...


def load_font(size: int, *, mono: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/SFNSMono.ttf" if mono else "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Supplemental/Andale Mono.ttf"
        if mono
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def ease(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def lerp(a: float, b: float, value: float) -> float:
    """Geometric interpolation for layout/animation only — never for displayed data."""

    return a + (b - a) * value


def fit_image(image: Image.Image, size: tuple[int, int], *, fill: tuple[int, int, int] = BG) -> Image.Image:
    target_w, target_h = size
    out = Image.new("RGB", size, fill)
    scale = min(target_w / image.width, target_h / image.height)
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    out.paste(resized, ((target_w - new_size[0]) // 2, (target_h - new_size[1]) // 2))
    return out


def paste_rgba(base: Image.Image, overlay: Image.Image, xy: tuple[int, int]) -> None:
    if overlay.mode != "RGBA":
        overlay = overlay.convert("RGBA")
    base.paste(overlay, xy, overlay)


def rgba_from_rgb(image: Image.Image, alpha: int = 255) -> Image.Image:
    out = image.convert("RGBA")
    out.putalpha(alpha)
    return out


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = INK,
    *,
    anchor: str | None = None,
) -> None:
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def sobel_edges(image: Image.Image, color: tuple[int, int, int] = EDGE, alpha: int = 180) -> Image.Image:
    """Real finite-difference edge magnitude of the image, as an RGBA overlay."""

    gray = np.asarray(image.convert("L"), dtype=np.float32)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    mag = np.sqrt(gx * gx + gy * gy)
    mag = mag / (mag.max() or 1.0)
    mask = np.clip((mag - 0.16) * 4.0, 0.0, 1.0)
    rgba = np.zeros((gray.shape[0], gray.shape[1], 4), dtype=np.uint8)
    rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2] = color
    rgba[:, :, 3] = np.clip(mask * alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def edge_density(image: Image.Image) -> float:
    """Real scalar: fraction of pixels whose normalised edge magnitude exceeds 0.16."""

    gray = np.asarray(image.convert("L"), dtype=np.float32)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    mag = np.sqrt(gx * gx + gy * gy)
    mag = mag / (mag.max() or 1.0)
    return float((mag > 0.16).mean())


def colour_masks(image: Image.Image, k: int = 4) -> list[tuple[Image.Image, tuple[int, int, int], float]]:
    """Real octree colour quantisation: each dominant colour as a mask + share."""

    quantized = image.convert("RGB").quantize(colors=k, method=Image.FASTOCTREE)
    palette = quantized.getpalette() or []
    labels = np.asarray(quantized)
    counts = sorted(quantized.getcolors() or [], reverse=True)
    total = sum(count for count, _ in counts) or 1
    layers = []
    for count, index in counts[:k]:
        rgb = tuple(palette[index * 3 : index * 3 + 3])
        mask = labels == index
        rgba = np.zeros((labels.shape[0], labels.shape[1], 4), dtype=np.uint8)
        rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2] = rgb
        rgba[:, :, 3] = np.where(mask, 175, 0).astype(np.uint8)
        layers.append((Image.fromarray(rgba, "RGBA"), rgb, count / total))
    return layers


def heat_overlay(heat: Image.Image, alpha: int = 200) -> Image.Image:
    """Map a single-channel heat image to a cyan->white RGBA overlay."""

    values = np.asarray(heat.convert("L"), dtype=np.float32) / 255.0
    rgba = np.zeros((values.shape[0], values.shape[1], 4), dtype=np.uint8)
    rgba[:, :, 0] = np.clip(values * 255, 0, 255).astype(np.uint8)
    rgba[:, :, 1] = np.clip(120 + values * 135, 0, 255).astype(np.uint8)
    rgba[:, :, 2] = np.clip(150 + values * 105, 0, 255).astype(np.uint8)
    rgba[:, :, 3] = np.clip(values * alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


# -- sci-fi HUD helpers (decoration only; never render a fabricated number) ----

def wireframe(image: Image.Image, spacing: int, color=(30, 64, 70), alpha: int = 55) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = image.size
    c = (*color, alpha)
    for x in range(0, w, spacing):
        draw.line((x, 0, x, h), fill=c, width=1)
    for y in range(0, h, spacing):
        draw.line((0, y, w, y), fill=c, width=1)
    image.paste(overlay, (0, 0), overlay)


def hud_frame(image: Image.Image, color=ACCENT, alpha: int = 150) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = image.size
    m = max(10, w // 44)
    arm = max(18, w // 30)
    c = (*color, alpha)
    for cx, cy, dx, dy in ((m, m, 1, 1), (w - m, m, -1, 1), (m, h - m, 1, -1), (w - m, h - m, -1, -1)):
        draw.line((cx, cy, cx + dx * arm, cy), fill=c, width=1)
        draw.line((cx, cy, cx, cy + dy * arm), fill=c, width=1)
    for x in range(m + arm, w - m - arm, max(24, w // 22)):
        draw.line((x, m - 4, x, m), fill=(*color, alpha // 2), width=1)
    image.paste(overlay, (0, 0), overlay)


def scan_beam(image: Image.Image, y: int, color=EDGE, alpha: int = 130, band: int | None = None) -> None:
    w, h = image.size
    band = band or max(6, h // 36)
    yy = np.arange(h)
    prof = np.clip(1.0 - np.abs(yy - y) / band, 0.0, 1.0)
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:, :, 0], arr[:, :, 1], arr[:, :, 2] = color
    arr[:, :, 3] = (prof * alpha).astype(np.uint8)[:, None]
    overlay = Image.fromarray(arr, "RGBA")
    image.paste(overlay, (0, 0), overlay)


def bracket_box(draw: ImageDraw.ImageDraw, box, color=EDGE, length: int = 10, width: int = 1) -> None:
    x0, y0, x1, y1 = box
    for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        draw.line((cx, cy, cx + dx * length, cy), fill=color, width=width)
        draw.line((cx, cy, cx, cy + dy * length), fill=color, width=width)


def reticle(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int, color=EDGE) -> None:
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=1)
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.line((cx + dx * r, cy + dy * r, cx + dx * (r + 7), cy + dy * (r + 7)), fill=color, width=1)
    draw.line((cx - 4, cy, cx + 4, cy), fill=color, width=1)
    draw.line((cx, cy - 4, cx, cy + 4), fill=color, width=1)


def particles(draw: ImageDraw.ImageDraw, points, color=EDGE, radius: int = 1) -> None:
    for x, y in points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def crosshair(draw: ImageDraw.ImageDraw, cx: int, cy: int, size: int, gap: int, color=EDGE) -> None:
    draw.line((cx - size, cy, cx - gap, cy), fill=color, width=1)
    draw.line((cx + gap, cy, cx + size, cy), fill=color, width=1)
    draw.line((cx, cy - size, cx, cy - gap), fill=color, width=1)
    draw.line((cx, cy + gap, cx, cy + size), fill=color, width=1)


def dashed_circle(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int, color=EDGE, dashes: int = 24, spin: float = 0.0) -> None:
    step = 360 / dashes
    offset = (spin * 360) % step
    for i in range(dashes):
        if i % 2 == 0:
            a0 = offset + i * step
            draw.arc((cx - r, cy - r, cx + r, cy + r), a0, a0 + step * 0.6, fill=color, width=1)


def target_designator(draw: ImageDraw.ImageDraw, box, color=EDGE, close: float = 1.0) -> None:
    """Lock-on corner brackets that close in as ``close`` goes 0 -> 1."""

    x0, y0, x1, y1 = box
    span = min(x1 - x0, y1 - y0)
    inset = int((1.0 - max(0.0, min(1.0, close))) * span * 0.35)
    bracket_box(draw, (x0 - inset, y0 - inset, x1 + inset, y1 + inset), color, length=max(8, span // 8), width=1)
    # range ticks on the top edge
    for t in range(5):
        tx = int(x0 + (x1 - x0) * t / 4)
        draw.line((tx, y0 - 5, tx, y0 - 1), fill=color, width=1)


def render_video_stream(
    config,
    renderer: FrameSource,
    output_path: Path,
    still_dir: Path,
    audio_path: Path | None = None,
) -> None:
    """Stream rendered frames to ffmpeg. Audio is optional (silent when None)."""

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found on PATH")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if still_dir.exists():
        shutil.rmtree(still_dir)
    still_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = config.out_dir / "frames"
    if config.keep_frames:
        if frame_dir.exists():
            shutil.rmtree(frame_dir)
        frame_dir.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg,
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{config.width}x{config.height}",
        "-r", str(config.fps),
        "-i", "-",
    ]
    if audio_path is not None:
        command += ["-i", str(audio_path)]
    command += [
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium" if config.preset == "production" else "veryfast",
        "-crf", "18" if config.preset == "production" else "20",
    ]
    if audio_path is not None:
        command += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    command += [str(output_path)]

    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin pipe could not be opened")

    total_frames = int(round(config.duration * config.fps))
    still_times = [0.08, 0.24, 0.43, 0.62, 0.80, 0.94]
    still_indices = {min(total_frames - 1, max(0, int(total_frames * value))) for value in still_times}
    try:
        for frame_index in range(total_frames):
            frame = renderer.render(frame_index)
            if frame_index in still_indices:
                frame.save(still_dir / f"still_{frame_index:06d}.png")
            if config.keep_frames:
                frame.save(frame_dir / f"frame_{frame_index:06d}.jpg", quality=92)
            process.stdin.write(frame.convert("RGB").tobytes())
            if frame_index and frame_index % max(1, config.fps * 10) == 0:
                print(f"rendered {frame_index}/{total_frames} frames")
    finally:
        process.stdin.close()
    exit_code = process.wait()
    if exit_code != 0:
        raise RuntimeError(f"ffmpeg exited with code {exit_code}")
