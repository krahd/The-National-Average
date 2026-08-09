"""NeurIPS 2026 edition of *The National Average*.

This renderer remains separate from the preserved production baseline. It uses the
same computed assets and provenance records but develops a quieter dramaturgy around
plurality, representation, concentration, erasure, and unresolved coexistence.
"""

from __future__ import annotations

from typing import Any

from PIL import Image, ImageDraw

from ..trace import write_json
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
from .neurips_metrics import count_below, stats_for
from .pipeline import (
    INTENTS,
    SUBJECT_INTENT,
    TITLE,
    VideoRenderConfig,
    generate_assets,
    prepare_foundation_assets,
    write_render_notes,
)
from .scenes import AUTHOR, WEBSITE, VideoFrameRenderer


NEURIPS_PHASE_SCHEDULE = [
    ("title", 0.7),
    ("sources", 1.6),
    ("spaces", 2.0),
    ("embed", 1.3),
    ("distribution", 2.0),
    ("weighting", 2.0),
    ("thresholds", 2.3),
    ("average", 1.5),
    ("erase", 2.0),
    ("residual", 1.5),
    ("coda", 2.4),
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
    """Submission edition: computed quantities, restrained text, no global HUD."""

    XFADE = 1.1

    def __init__(self, config, assets):
        super().__init__(config, assets)
        self.segments = neurips_segments(config.duration)
        self.phase_methods["sources"] = self.phase_sources
        self.phase_methods["distribution"] = self.phase_distribution
        self.phase_methods["thresholds"] = self.phase_thresholds
        self.phase_methods["coda"] = self.phase_coda_neurips

    def phase_sources(self, p):
        image = self.blank()
        draw = ImageDraw.Draw(image)
        thumbs = list(self.archive_thumbs.items())
        if not thumbs:
            return image

        cols = 21 if self.config.width >= 1280 else 14
        rows = 13 if self.config.width >= 1280 else 9
        n = min(len(thumbs), cols * rows)
        shown = max(1, int(n * ease(min(1.0, p * 1.2))))
        margin_x = int(self.config.width * 0.055)
        top = int(self.config.height * 0.15)
        usable_w = self.config.width - 2 * margin_x
        usable_h = int(self.config.height * 0.68)
        cell_w = usable_w // cols
        cell_h = usable_h // rows

        for i, (_, thumb) in enumerate(thumbs[:shown]):
            r, c = divmod(i, cols)
            tile = fit_image(thumb, (max(8, cell_w - 5), max(8, cell_h - 5)))
            x = margin_x + c * cell_w + (cell_w - tile.width) // 2
            y = top + r * cell_h + (cell_h - tile.height) // 2
            paste_rgba(image, rgba_from_rgb(tile, 224), (x, y))

        draw_text(
            draw,
            (margin_x, int(self.config.height * 0.08)),
            f"SOURCE // archive: {len(self.assets.archive)} political symbols",
            self.label_font,
            MUTED,
        )
        return image

    def phase_distribution(self, p):
        """Show the concentration introduced by each real weighting record."""

        image = self.blank()
        draw = ImageDraw.Draw(image)
        intents = [intent for intent in INTENTS if intent in self.assets.weights]
        if not intents:
            return image

        margin = int(self.config.width * 0.08)
        top = int(self.config.height * 0.17)
        row_h = int(self.config.height * 0.13)
        bar_x = int(self.config.width * 0.31)
        bar_w = int(self.config.width * 0.48)

        draw_text(draw, (margin, int(self.config.height * 0.08)), "WEIGHT DISTRIBUTION", self.label_font, MUTED)

        shown = max(1, int(len(intents) * min(1.0, p * 1.35)))
        for row, intent in enumerate(intents[:shown]):
            run = self.assets.weights[intent]
            stats = stats_for(intent, run.weights)
            y = top + row * row_h
            draw_text(draw, (margin, y), intent, self.small, INK)
            draw_text(
                draw,
                (margin, y + 22),
                f"H={stats.normalised_entropy:.3f}  N_eff={stats.effective_count:.1f}  max={stats.top_share*100:.1f}%",
                self.tiny,
                MUTED,
            )

            ranked = sorted(run.weights.items(), key=lambda item: item[1], reverse=True)
            total = sum(max(0.0, float(value)) for _, value in ranked) or 1.0
            x = bar_x
            for idx, (code, value) in enumerate(ranked):
                share = max(0.0, float(value)) / total
                width = max(1, int(bar_w * share))
                level = max(38, 198 - min(150, idx * 8))
                draw.rectangle((x, y, min(bar_x + bar_w, x + width), y + 18), fill=(level, level, level))
                x += width
                if x >= bar_x + bar_w:
                    break

            top_codes = "  ".join(f"{code.upper()} {value/total*100:.1f}%" for code, value in ranked[:4])
            draw_text(draw, (bar_x, y + 24), top_codes, self.tiny, EDGE)

        return image

    def phase_thresholds(self, p):
        """Progressively remove contributors below a real contribution threshold."""

        image = self.blank()
        draw = ImageDraw.Draw(image)
        run = self.assets.weights.get(SUBJECT_INTENT)
        if run is None:
            return image

        # Sweep logarithmically from 0.01% to 1% contribution.
        threshold = 10 ** (-4.0 + 2.0 * ease(min(1.0, p)))
        total = sum(max(0.0, float(value)) for value in run.weights.values()) or 1.0
        probabilities = {code: max(0.0, float(value)) / total for code, value in run.weights.items()}
        codes = sorted(probabilities)

        cols = 18 if self.config.width >= 1280 else 12
        margin_x = int(self.config.width * 0.07)
        top = int(self.config.height * 0.18)
        usable_w = self.config.width - 2 * margin_x
        cell_w = usable_w // cols
        cell_h = int(cell_w * 0.78)

        removed = count_below(run.weights, threshold)
        draw_text(draw, (margin_x, int(self.config.height * 0.07)), "ERASURE THRESHOLD", self.label_font, MUTED)
        draw_text(
            draw,
            (margin_x, int(self.config.height * 0.11)),
            f"{SUBJECT_INTENT} // contribution < {threshold*100:.3f}% // {removed}/{len(codes)} below threshold",
            self.small,
            INK,
        )

        for i, code in enumerate(codes):
            thumb = self.archive_thumbs.get(code)
            if thumb is None:
                continue
            r, c = divmod(i, cols)
            x = margin_x + c * cell_w
            y = top + r * cell_h
            tile = fit_image(thumb, (max(8, cell_w - 7), max(8, int((cell_w - 7) * 0.75))))
            alpha = 30 if probabilities[code] < threshold else 220
            paste_rgba(image, rgba_from_rgb(tile, alpha), (x, y))

        return image

    def phase_coda_neurips(self, p):
        image = self.blank()
        draw = ImageDraw.Draw(image)
        intents = [i for i in ("equal", "population", "gdp", "co2", "cumulative_co2") if i in self.assets.generated]
        items = [(intent, self.preferred_image(intent)) for intent in intents]
        items = [(intent, source) for intent, source in items if source is not None]
        if not items:
            return image

        n = len(items)
        gap = max(10, int(self.config.width * 0.008))
        max_w = int((self.config.width * 0.86 - gap * (n - 1)) / n)
        max_h = int(max_w * 0.75)
        y = int(self.config.height * 0.30)
        total_w = n * max_w + (n - 1) * gap
        x0 = (self.config.width - total_w) // 2
        alpha = int(245 * ease(min(1.0, p * 1.5)))

        if p > 0.2:
            draw_text(draw, (x0, int(self.config.height * 0.20)), "SAME CORPUS", self.tiny, MUTED)

        for i, (intent, source) in enumerate(items):
            x = x0 + i * (max_w + gap)
            tile = fit_image(source, (max_w, max_h))
            paste_rgba(image, rgba_from_rgb(tile, alpha), (x, y))
            if p > 0.34:
                draw_text(draw, (x, y + max_h + 12), intent, self.tiny, MUTED)

        if p > 0.58:
            draw_text(draw, (x0, int(self.config.height * 0.74)), "DIFFERENT WEIGHTINGS", self.tiny, EDGE)
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
            frame = Image.blend(frame, nxt, min(0.52, amount * 0.52))

        frame = effects.treat(
            frame,
            vmask=self.vmask,
            glitch=0.0,
            glitch_mode="block",
            seed=self.config.seed,
            frame=frame_index,
            grain_amount=2.0,
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

    assets.provenance.setdefault("sources", {})[
        "NeurIPS weighting entropy / effective contributor count / threshold counts"
    ] = "tna.video.neurips_metrics over tna.weights.WeightingRun.weights"
    write_json(config.out_dir / "provenance" / "analysis_provenance.json", assets.provenance)

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
