"""NeurIPS 2026 edition of *The National Average*.

This renderer is deliberately separate from the preserved production baseline.  It
uses the same computed assets and provenance records, but replaces the dense
machine-demo dramaturgy with a quieter sequence organised around plurality,
representation, weighting, erasure, and unresolved coexistence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from . import effects
from .compositor import (
    BG,
    EDGE,
    INK,
    MUTED,
    draw_text,
    ease,
    fit_image,
    paste_rgba,
    render_video_stream,
    rgba_from_rgb,
)
from .pipeline import (
    INTENTS,
    TITLE,
    VideoRenderConfig,
    generate_assets,
    prepare_foundation_assets,
    write_render_notes,
)
from .scenes import AUTHOR, WEBSITE, VideoFrameRenderer


NEURIPS_PHASE_SCHEDULE = [
    ("title", 0.8),
    ("sources", 1.8),
    ("spaces", 2.4),
    ("embed", 1.5),
    ("weighting", 2.5),
    ("average", 1.8),
    ("erase", 2.7),
    ("residual", 1.8),
    ("coda", 2.7),
]


def neurips_segments(duration: float) -> list[tuple[str, float, float]]:
    total = sum(weight for _, weight in NEURIPS_PHASE_SCHEDULE)
    out: list[tuple[str, float, float]] = []
    acc = 0.0
    for key, weight in NEURIPS_PHASE_SCHEDULE:
        start = acc / total * duration
        acc += weight
        out.append((key, start, acc / total * duration))
    return out


class NeurIPSFrameRenderer(VideoFrameRenderer):
    """Quiet submission edition; no global HUD and almost no glitch spectacle."""

    XFADE = 1.2

    def __init__(self, config, assets):
        super().__init__(config, assets)
        self.segments = neurips_segments(config.duration)
        self.phase_methods["sources"] = self.phase_sources
        self.phase_methods["coda"] = self.phase_coda_neurips

    def phase_sources(self, p):
        image = self.blank()
        draw = ImageDraw.Draw(image)
        thumbs = list(self.archive_thumbs.items())
        if not thumbs:
            return image

        cols = 14 if self.config.width >= 1280 else 10
        rows = 8 if self.config.width >= 1280 else 6
        n = min(len(thumbs), cols * rows)
        shown = max(1, int(n * ease(min(1.0, p * 1.25))))
        margin_x = int(self.config.width * 0.065)
        top = int(self.config.height * 0.16)
        usable_w = self.config.width - 2 * margin_x
        usable_h = int(self.config.height * 0.65)
        cell_w = usable_w // cols
        cell_h = usable_h // rows

        for i, (_, thumb) in enumerate(thumbs[:shown]):
            r, c = divmod(i, cols)
            tile = fit_image(thumb, (max(8, cell_w - 7), max(8, cell_h - 7)))
            x = margin_x + c * cell_w + (cell_w - tile.width) // 2
            y = top + r * cell_h + (cell_h - tile.height) // 2
            paste_rgba(image, rgba_from_rgb(tile, 224), (x, y))

        draw_text(
            draw,
            (margin_x, int(self.config.height * 0.09)),
            f"SOURCE // {len(self.assets.archive)} political symbols",
            self.label_font,
            MUTED,
        )
        if p > 0.68:
            draw_text(
                draw,
                (margin_x, int(self.config.height * 0.86)),
                "before an average, a corpus",
                self.small,
                EDGE,
            )
        return image

    def phase_coda_neurips(self, p):
        image = self.blank()
        draw = ImageDraw.Draw(image)
        intents = [i for i in ("equal", "population", "gdp", "co2", "cumulative_co2") if i in self.assets.generated]
        items = []
        for intent in intents:
            img = self.preferred_image(intent)
            if img is not None:
                items.append((intent, img))
        if not items:
            return image

        n = len(items)
        gap = max(10, int(self.config.width * 0.008))
        max_w = int((self.config.width * 0.86 - gap * (n - 1)) / n)
        max_h = int(max_w * 0.75)
        y = int(self.config.height * 0.30)
        total_w = n * max_w + (n - 1) * gap
        x0 = (self.config.width - total_w) // 2
        alpha = int(245 * ease(min(1.0, p * 1.6)))

        for i, (intent, source) in enumerate(items):
            x = x0 + i * (max_w + gap)
            tile = fit_image(source, (max_w, max_h))
            paste_rgba(image, rgba_from_rgb(tile, alpha), (x, y))
            if p > 0.36:
                draw_text(draw, (x, y + max_h + 12), intent, self.tiny, MUTED)

        if p > 0.56:
            draw_text(
                draw,
                (self.config.width // 2, int(self.config.height * 0.76)),
                "there is no average before a representation",
                self.h1_font,
                INK,
                anchor="mm",
            )
        if p > 0.78:
            draw_text(
                draw,
                (self.config.width // 2, int(self.config.height * 0.90)),
                f"{TITLE.upper()}  ·  {AUTHOR}  ·  {WEBSITE}",
                self.tiny,
                MUTED,
                anchor="mm",
            )
        return image

    def render(self, frame_index):
        t = frame_index / self.config.fps
        self._t = t
        duration = self.config.duration
        i, key, start, end = self._phase_at(t)
        p = (t - start) / max(1e-6, end - start)
        frame = self.phase_methods[key](min(1.0, p))

        if end - t < self.XFADE and i < len(self.segments) - 1:
            nxt_key = self.segments[i + 1][0]
            nxt = self.phase_methods[nxt_key](0.0)
            amount = (self.XFADE - (end - t)) / self.XFADE
            frame = Image.blend(frame, nxt, min(0.55, amount * 0.55))

        # The original renderer's HUD/glitch aesthetic is intentionally removed.
        # A small amount of grain keeps computationally different source images
        # inside one visual register without pretending the operations are smooth.
        frame = effects.treat(
            frame,
            vmask=self.vmask,
            glitch=0.0,
            glitch_mode="block",
            seed=self.config.seed,
            frame=frame_index,
            grain_amount=2.2,
            aberration=0,
        )
        fade = max(0.0, min(min(1.0, t / 1.6), min(1.0, (duration - t) / 3.5)))
        if fade < 1.0:
            frame = Image.blend(Image.new("RGB", frame.size, BG), frame, fade)
        return frame


def render_neurips_video(config: VideoRenderConfig) -> dict[str, Any]:
    """Render the NeurIPS edition without changing the baseline renderer."""

    config.out_dir.mkdir(parents=True, exist_ok=True)
    required_foundation = config.foundation == "required"
    model_manifest = None
    if config.foundation != "off":
        model_manifest = prepare_foundation_assets(config.out_dir, required=required_foundation)

    assets = generate_assets(
        config,
        required_foundation=required_foundation,
        model_manifest=model_manifest,
    )

    from .neurips_audio import render_neurips_soundtrack

    audio_path = render_neurips_soundtrack(assets, config) if config.audio else None
    output_path = config.out_dir / "the_national_average_neurips_2026.mp4"
    renderer = NeurIPSFrameRenderer(config, assets)
    render_video_stream(
        config,
        renderer,
        output_path,
        config.out_dir / "stills-neurips",
        audio_path=audio_path,
    )
    write_render_notes(config, assets, output_path)
    return {
        "video": str(output_path),
        "audio": str(audio_path) if audio_path else None,
        "stills": str(config.out_dir / "stills-neurips"),
        "asset_manifest": str(assets.asset_dir / "asset_manifest.json"),
        "provenance": str(config.out_dir / "provenance" / "analysis_provenance.json"),
    }
