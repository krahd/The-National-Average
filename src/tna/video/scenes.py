"""Continuous machine-pipeline journey for *The National Average* (round 6).

One flowing pass through the machine's pipeline under a military/gaze HUD: title,
ingest, segment, tokenize, attend, embed, eigenbasis, retrieve, reconstruct,
spaces, average, weighting, regions, erase, name, residual, coda. Each phase
animates a real CV operation; text is sparse machine-labels plus real telemetry.
Grade/grain/glitch is applied globally in :meth:`render` via ``effects`` (imagery
only) as fast, short bursts driven by per-phase signals. No phase synthesises a
displayed measurement.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from ..data import rasterize_flag
from . import effects
from .compositor import (
    ACCENT,
    BG,
    EDGE,
    FAINT,
    INK,
    MUTED,
    bracket_box,
    crosshair,
    dashed_circle,
    draw_text,
    ease,
    fit_image,
    heat_overlay,
    hud_frame,
    load_font,
    paste_rgba,
    reticle,
    rgba_from_rgb,
    target_designator,
    wireframe,
)
from .pipeline import INTENT_LABELS, INTENTS, SUBJECT_INTENT, TITLE, phase_segments

AUTHOR = "Tomas Laurenzo"
WEBSITE = "laurenzo.net"
SPEED = 1.15  # everything 15% faster


def _bump(p: float, centre: float, width: float = 0.1) -> float:
    return max(0.0, 1.0 - abs(p - centre) / width)


def _fast(p: float, rate: float = 1.6) -> float:
    return ease(min(1.0, p * rate))


class VideoFrameRenderer:
    XFADE = 0.5
    NO_HUD = {"title", "coda"}
    # Per-phase glitch bursts: (fraction-of-phase, amplitude); 0.10s wide -> fast.
    GLITCH = {
        "title": ([], "block"),
        "ingest": ([(0.12, 0.3)], "block"),
        "segment": ([(0.35, 0.4), (0.62, 0.4)], "channel"),
        "tokenize": ([(0.3, 0.5), (0.55, 0.5), (0.8, 0.45)], "patch_shuffle"),
        "attend": ([(0.5, 0.3)], "block"),
        "embed": ([(0.25, 0.35), (0.6, 0.4)], "channel"),
        "eigenbasis": ([(0.5, 0.35)], "block"),
        "retrieve": ([(0.72, 0.5)], "pixel_sort"),
        "reconstruct": ([(0.4, 0.4), (0.72, 0.4)], "latent"),
        "spaces": ([(0.4, 0.4), (0.72, 0.4)], "channel"),
        "average": ([(0.5, 0.6), (0.66, 0.45)], "latent"),
        "weighting": ([(0.4, 0.4), (0.72, 0.4)], "channel"),
        "regions": ([(0.4, 0.4), (0.72, 0.4)], "block"),
        "erase": ([(0.3, 0.4), (0.6, 0.45), (0.85, 0.4)], "block"),
        "name": ([(0.18, 0.45), (0.5, 0.45), (0.8, 0.4)], "channel"),
        "residual": ([(0.2, 0.45)], "latent"),
        "coda": ([], "block"),
    }

    def __init__(self, config, assets):
        self.config = config
        self.assets = assets
        width, height = config.width, config.height
        self.title_font = load_font(max(40, width // 16))
        self.h1_font = load_font(max(20, width // 38))
        self.label_font = load_font(max(13, width // 72), mono=True)
        self.small = load_font(max(11, width // 104), mono=True)
        self.tiny = load_font(max(9, width // 132), mono=True)
        self.names = {code: polity.name for code, polity in assets.corpus.items()}
        self.archive_thumbs = {
            polity.code: rasterize_flag(polity, (72, 54)).convert("RGB")
            for polity in assets.archive
        }
        self.focus = assets.focus
        self.subject = assets.focus[0] if assets.focus else None
        self._t = 0.0
        self.vmask = effects.vignette_mask((width, height))
        self.segments = phase_segments(config.duration)
        self.cloud_tile = (max(10, width // 74), max(8, height // 74))
        self.cloud_thumbs = {}
        if assets.embedding is not None:
            for code in assets.embedding.coords2d:
                thumb = self.archive_thumbs.get(code)
                if thumb is not None:
                    self.cloud_thumbs[code] = thumb.resize(self.cloud_tile, Image.Resampling.BILINEAR)
        self.norm_coords = assets.embedding.normalised_coords() if assets.embedding is not None else {}
        self.phase_methods = {
            "title": self.phase_title,
            "ingest": self.phase_ingest,
            "segment": self.phase_segment,
            "tokenize": self.phase_tokenize,
            "attend": self.phase_attend,
            "embed": self.phase_embed,
            "eigenbasis": self.phase_eigenbasis,
            "retrieve": self.phase_retrieve,
            "reconstruct": self.phase_reconstruct,
            "spaces": self.phase_spaces,
            "average": self.phase_average,
            "weighting": self.phase_weighting,
            "regions": self.phase_regions,
            "erase": self.phase_erase,
            "name": self.phase_name,
            "residual": self.phase_residual,
            "coda": self.phase_coda,
        }

    # -- helpers -----------------------------------------------------------
    def blank(self):
        return Image.new("RGB", (self.config.width, self.config.height), BG)

    def tag(self, draw, xy, text, colour=EDGE, font=None):
        draw_text(draw, xy, text, font or self.label_font, colour)

    def cloud_xy(self, code, margin, top, fw, fh):
        x, y = self.norm_coords[code]
        return margin + int(x * fw), top + int(y * fh)

    def field(self):
        margin = int(self.config.width * 0.1)
        top = int(self.config.height * 0.13)
        fw = self.config.width - 2 * margin - self.cloud_tile[0]
        fh = int(self.config.height * 0.72) - self.cloud_tile[1]
        return margin, top, fw, fh

    def subject_at(self, p):
        if not self.focus:
            return None
        return self.focus[min(len(self.focus) - 1, int(p * len(self.focus)))]

    def gaze_points(self, subj, x, y, w, h):
        """Real CV keypoints: the centroids of the detected colour regions."""

        pts = []
        if subj is not None and subj.component is not None:
            for r in subj.component.regions[:6]:
                cx = x + int((r.bbox[0] + r.bbox[2]) / 2 * w)
                cy = y + int((r.bbox[1] + r.bbox[3]) / 2 * h)
                pts.append((cx, cy))
        return pts or [(x + w // 2, y + h // 2)]

    def histogram(self, draw, subj, ox, oy, width):
        """Real colour-share histogram of the detected regions."""

        if subj is None or subj.component is None:
            return
        x = ox
        for r in subj.component.regions[:6]:
            bw = max(2, int(width * r.share))
            draw.rectangle((x, oy, x + bw, oy + 7), fill=tuple(r.rgb))
            x += bw + 1

    def hud(self, image, key, t):
        draw = ImageDraw.Draw(image)
        hud_frame(image, color=(70, 150, 160), alpha=120)
        m = max(14, int(self.config.width * 0.03))
        draw_text(draw, (m, int(self.config.height * 0.045)), f"NATIONAL-AVERAGE.PROC // {key.upper()} // t={t:05.1f}s", self.tiny, (110, 180, 188))
        rx = int(self.config.width * 0.78)
        ry = int(self.config.height * 0.045)
        rec = self.assets.recognition.get(SUBJECT_INTENT)
        lines = []
        if rec is not None:
            lines += [f"conf {rec.confidence:0.3f}", f"near {rec.nearest.code.upper()} {rec.nearest.similarity:0.3f}"]
        if self.assets.component is not None:
            c = self.assets.component
            lines += [f"sym {c.symmetry_h:0.2f}/{c.symmetry_v:0.2f}", f"edge {c.edge_density:0.3f}"]
        if self.subject is not None and self.subject.feature is not None:
            lines += [f"z0 {self.subject.feature[0]:+0.2f}", f"z1 {self.subject.feature[1]:+0.2f}"]
        for i, line in enumerate(lines[:6]):
            draw_text(draw, (rx, ry + i * 13), line, self.tiny, (90, 150, 158))
        # real CLIP/PCA feature-vector strip (sci-fi telemetry artefact)
        if self.subject is not None and self.subject.feature is not None:
            fy = ry + 6 * 13 + 8
            scale = float(np.abs(self.subject.feature).max()) or 1.0
            for j, v in enumerate(self.subject.feature[:16]):
                bx = rx + j * 5
                draw.line((bx, fy, bx, fy - int((v / scale) * 14)), fill=(80, 140, 148), width=2)

    # -- 0. title ----------------------------------------------------------
    def phase_title(self, p):
        image = self.blank()
        draw = ImageDraw.Draw(image)
        cx, cy = self.config.width // 2, int(self.config.height * 0.42)
        hud_frame(image, color=(60, 130, 140), alpha=90)
        if p < 0.45:
            draw_text(draw, (int(self.config.width * 0.06), int(self.config.height * 0.06)), "INITIALISING // NATIONAL-AVERAGE.PROC", self.tiny, (90, 150, 158))
        dashed_circle(draw, cx, cy, int(self.config.height * 0.2), (40, 90, 98), spin=p * 0.6)
        a = ease(min(1.0, p * 1.8))
        draw_text(draw, (cx, cy), TITLE.upper(), self.title_font, tuple(int(c * a) for c in INK), anchor="mm")
        if p > 0.32:
            draw_text(draw, (cx, cy + int(self.config.height * 0.16)), AUTHOR, self.h1_font, INK, anchor="mm")
        if p > 0.48:
            draw_text(draw, (cx, cy + int(self.config.height * 0.24)), WEBSITE, self.label_font, ACCENT, anchor="mm")
        return image

    # -- 1. ingest (military gaze acquisition) -----------------------------
    def phase_ingest(self, p):
        image = self.blank()
        draw = ImageDraw.Draw(image)
        subj = self.subject_at(p)
        if subj is None:
            return image
        e = _fast(p, 2.2)
        w = int(self.config.width * 0.36)
        panel = fit_image(subj.flag, (w, int(w * 0.75)))
        x = (self.config.width - panel.width) // 2
        y = (self.config.height - panel.height) // 2
        paste_rgba(image, rgba_from_rgb(panel, int(40 + 215 * e)), (x, y))
        target_designator(draw, (x - 8, y - 8, x + panel.width + 8, y + panel.height + 8), EDGE, close=e)
        # detected feature points (CV keypoints) on every region centroid
        pts = self.gaze_points(subj, x, y, panel.width, panel.height)
        for px, py in pts:
            draw.line((px - 4, py, px + 4, py), fill=(70, 140, 150), width=1)
            draw.line((px, py - 4, px, py + 4), fill=(70, 140, 150), width=1)
        # jerky machine gaze: snaps to a different feature every ~0.33s
        idx = int(self._t / 0.33) % len(pts)
        gx, gy = pts[idx]
        crosshair(draw, gx, gy, int(panel.height * 0.18), int(panel.height * 0.05), EDGE)
        dashed_circle(draw, gx, gy, int(panel.height * 0.13), EDGE, spin=int(self._t * 6) / 6.0)
        bracket_box(draw, (gx - 14, gy - 14, gx + 14, gy + 14), EDGE, length=6)
        self.histogram(draw, subj, x, y + panel.height + 12, panel.width)
        self.tag(draw, (x, y - 22), f"ACQUIRE // {subj.code.upper()}  {subj.name}   track[{idx + 1}/{len(pts)}]")
        return image

    # -- 2. segment (component detection) ----------------------------------
    def phase_segment(self, p):
        image = self.blank()
        draw = ImageDraw.Draw(image)
        subj = self.subject_at(p)
        comp = subj.component if subj else None
        if subj is None or comp is None:
            return image
        w, h = int(self.config.width * 0.46), int(self.config.width * 0.46 * 0.75)
        x, y = (self.config.width - w) // 2, (self.config.height - h) // 2
        paste_rgba(image, rgba_from_rgb(fit_image(subj.flag, (w, h)), 235), (x, y))
        if p > 0.2:
            paste_rgba(image, heat_overlay(comp.edge_image().resize((w, h)), alpha=int(120 * _fast(p))), (x, y))
        shown = int(len(comp.regions) * _fast(p, 1.8)) + 1
        for region in comp.regions[:shown]:
            bx0, by0 = x + int(region.bbox[0] * w), y + int(region.bbox[1] * h)
            bx1, by1 = x + int(region.bbox[2] * w), y + int(region.bbox[3] * h)
            bracket_box(draw, (bx0, by0, bx1, by1), EDGE, length=8)
            draw.rectangle((bx0, by0 - 11, bx0 + 58, by0 - 1), fill=(8, 12, 16))
            draw_text(draw, (bx0 + 2, by0 - 11), f"{region.hex} {region.share * 100:0.0f}%", self.tiny, (200, 214, 212))
        self.tag(draw, (x, y - 22), f"SEGMENT // {len(comp.regions)} regions  ·  {'vertical' if comp.vertical else 'horizontal'} bands")
        self.tag(draw, (x, y + h + 8), f"sym h={comp.symmetry_h:0.2f} v={comp.symmetry_v:0.2f}   edge {comp.edge_density:0.3f}", MUTED, self.small)
        return image

    # -- 3. tokenize -------------------------------------------------------
    def phase_tokenize(self, p):
        image = self.blank()
        draw = ImageDraw.Draw(image)
        subj = self.subject_at(p)
        if subj is None:
            return image
        grid = subj.saliency.grid if subj.saliency else 7
        cell = int(self.config.height * 0.07)
        gap = int(_fast(p, 2.0) * cell * 0.55)
        span = grid * cell + (grid - 1) * gap
        ox, oy = (self.config.width - span) // 2, (self.config.height - span) // 2
        src = fit_image(subj.flag, (grid * cell, grid * cell))
        for r in range(grid):
            for c in range(grid):
                patch = src.crop((c * cell, r * cell, (c + 1) * cell, (r + 1) * cell))
                px, py = ox + c * (cell + gap), oy + r * (cell + gap)
                paste_rgba(image, rgba_from_rgb(patch, 240), (px, py))
                if gap > 1:
                    draw.rectangle((px, py, px + cell, py + cell), outline=(40, 70, 76), width=1)
        self.tag(draw, (ox, oy - 22), f"TOKENIZE // {grid}x{grid} patches -> {grid * grid} tokens")
        return image

    # -- 4. attend ---------------------------------------------------------
    def phase_attend(self, p):
        image = self.blank()
        draw = ImageDraw.Draw(image)
        subj = self.subject_at(p)
        if subj is None or subj.saliency is None:
            return image
        sal = subj.saliency
        layer = min(sal.n_layers - 1, int(_fast(p, 1.3) * sal.n_layers))
        w, h = int(self.config.width * 0.46), int(self.config.width * 0.46 * 0.75)
        x, y = (self.config.width - w) // 2, (self.config.height - h) // 2
        dim = Image.eval(fit_image(subj.flag, (w, h)), lambda v: int(v * 0.36))
        paste_rgba(image, rgba_from_rgb(dim, 245), (x, y))
        paste_rgba(image, heat_overlay(sal.layer_heatmap(layer, (w, h)), alpha=245), (x, y))
        reticle(draw, x + w // 2, y + h // 2, int(h * 0.22), EDGE)
        self.tag(draw, (x, y - 22), f"ATTENTION // rollout layer {layer + 1}/{sal.n_layers}")
        return image

    # -- 5. embed ----------------------------------------------------------
    def phase_embed(self, p):
        image = self.blank()
        draw = ImageDraw.Draw(image)
        if self.assets.embedding is None:
            return image
        wireframe(image, max(40, self.config.width // 18))
        margin, top, fw, fh = self.field()
        cx, cy = self.config.width // 2, self.config.height // 2
        e = _fast(p, 1.7)
        positions = {}
        for code in self.cloud_thumbs:
            tx, ty = self.cloud_xy(code, margin, top, fw, fh)
            positions[code] = (int(cx + (tx - cx) * e), int(cy + (ty - cy) * e))
        if p > 0.45:
            for code, nbrs in self.assets.embedding.knn.items():
                if code in positions:
                    ax, ay = positions[code]
                    for nbr, _ in nbrs[:1]:
                        if nbr in positions:
                            draw.line((ax, ay, *positions[nbr]), fill=(34, 66, 72), width=1)
        for code, (x, y) in positions.items():
            paste_rgba(image, rgba_from_rgb(self.cloud_thumbs[code], int(120 + 135 * e)), (x, y))
        self.tag(draw, (margin, int(self.config.height * 0.08)), f"EMBED // {len(positions)} flags -> 512-d CLIP space")
        return image

    # -- 6. eigenbasis -----------------------------------------------------
    def phase_eigenbasis(self, p):
        image = self.blank()
        draw = ImageDraw.Draw(image)
        eig = self.assets.eigenflags
        if not eig:
            return image
        cols = 4
        cw, ch = int(self.config.width * 0.62 / cols), int(self.config.width * 0.62 / cols * 0.75)
        ox, oy = int(self.config.width * 0.07), int(self.config.height * 0.24)
        shown = int(len(eig) * _fast(p, 1.6)) + 1
        for i, comp_img in enumerate(eig[:shown]):
            r, c = divmod(i, cols)
            x, y = ox + c * (cw + 8), oy + r * (ch + 22)
            paste_rgba(image, rgba_from_rgb(fit_image(comp_img.convert("RGB"), (cw, ch)), 235), (x, y))
            draw.rectangle((x, y, x + cw, y + ch), outline=(40, 70, 76), width=1)
            draw_text(draw, (x, y + ch + 3), f"PC{i + 1}", self.tiny, MUTED)
        if self.subject is not None and self.subject.feature is not None:
            bx, by = int(self.config.width * 0.74), int(self.config.height * 0.28)
            f = self.subject.feature
            scale = float(np.abs(f).max()) or 1.0
            self.tag(draw, (bx, by - 22), "z = projection", MUTED, self.small)
            for i, v in enumerate(f[:16]):
                yy = by + i * 12
                draw.line((bx, yy, bx + int((v / scale) * self.config.width * 0.1), yy), fill=EDGE, width=3)
        self.tag(draw, (ox, int(self.config.height * 0.16)), f"BASIS // {len(eig)} eigenflags  ·  learned subspace")
        return image

    # -- 7. retrieve -------------------------------------------------------
    def phase_retrieve(self, p):
        image = self.blank()
        draw = ImageDraw.Draw(image)
        rec = self.assets.recognition.get(SUBJECT_INTENT)
        if self.assets.embedding is None or rec is None:
            return image
        margin, top, fw, fh = self.field()
        sims = rec.corpus_similarity
        lo, hi = (min(sims.values()), max(sims.values())) if sims else (0.0, 1.0)
        radius = _fast(p, 1.4)
        cx, cy = self.config.width // 2, self.config.height // 2
        for code in self.cloud_thumbs:
            x, y = self.cloud_xy(code, margin, top, fw, fh)
            s = (sims.get(code, lo) - lo) / ((hi - lo) or 1.0)
            d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 / (0.5 * self.config.width)
            lit = 1.0 if d <= radius * 1.6 else 0.2
            paste_rgba(image, rgba_from_rgb(self.cloud_thumbs[code], int(40 + 210 * s * lit)), (x, y))
        dashed_circle(draw, cx, cy, int(radius * fh * 0.5), (60, 120, 128), spin=p)
        self.tag(draw, (margin, int(self.config.height * 0.08)), "RETRIEVE // nearest neighbour to the average")
        if p > 0.55:
            nx, ny = self.cloud_xy(rec.nearest.code, margin, top, fw, fh)
            bracket_box(draw, (nx - 3, ny - 3, nx + self.cloud_tile[0] + 3, ny + self.cloud_tile[1] + 3), EDGE, length=8, width=2)
            self.tag(draw, (margin, int(self.config.height * 0.86)), f"-> {rec.nearest.name.upper()}  cos {rec.nearest.similarity:0.3f}")
        return image

    # -- 8. reconstruct ----------------------------------------------------
    def _row(self, image, draw, items, title, label_colour=EDGE):
        n = max(1, len(items))
        cw = int(self.config.width * 0.88 / n)
        ch = int(cw * 0.75)
        ox = int((self.config.width - n * cw) / 2)
        oy = (self.config.height - ch) // 2
        for i, (label, img, sub) in enumerate(items):
            x = ox + i * cw
            tile = fit_image(img.convert("RGB"), (cw - 8, ch))
            draw.rectangle((x, oy - 2, x + tile.width + 2, oy + tile.height + 2), outline=(40, 70, 76), width=1)
            paste_rgba(image, rgba_from_rgb(tile, 240), (x + 1, oy))
            draw_text(draw, (x + 1, oy + ch + 4), label, self.tiny, label_colour)
            if sub:
                draw_text(draw, (x + 1, oy + ch + 16), sub, self.tiny, INK)
        self.tag(draw, (ox, oy - 26), title)
        return oy, ch

    def phase_reconstruct(self, p):
        image = self.blank()
        draw = ImageDraw.Draw(image)
        recon = self.assets.reconstructions
        if not recon or self.subject is None:
            return image
        items = [("input", self.subject.flag, "")] + [(k, v, "") for k, v in recon.items()]
        shown = int(len(items) * _fast(p, 1.7)) + 1
        self._row(image, draw, items[:shown], "RECONSTRUCT // one flag, many representations")
        return image

    # -- 9. spaces (other averagings) --------------------------------------
    def phase_spaces(self, p):
        image = self.blank()
        draw = ImageDraw.Draw(image)
        gen = self.assets.generated.get(SUBJECT_INTENT, {})
        order = [b for b in ("pixel", "palette", "pca", "svg", "sdvae", "clip") if b in gen]
        rec = self.assets.recognition.get(SUBJECT_INTENT)
        items = []
        for b in order:
            sub = ""
            if b == "clip" and rec is not None:
                sub = f"->{rec.nearest.code.upper()}"
            items.append((b, gen[b], sub))
        shown = int(len(items) * _fast(p, 1.7)) + 1
        self._row(image, draw, items[:shown], "SPACES // one averaging, many representations")
        return image

    # -- 10. average -------------------------------------------------------
    def phase_average(self, p):
        image = self.blank()
        draw = ImageDraw.Draw(image)
        cx, cy = self.config.width // 2, self.config.height // 2
        if p < 0.45 and self.assets.embedding is not None:
            margin, top, fw, fh = self.field()
            k = _fast(p / 0.45)
            for code in self.cloud_thumbs:
                x, y = self.cloud_xy(code, margin, top, fw, fh)
                px, py = int(x + (cx - x) * k), int(y + (cy - y) * k)
                paste_rgba(image, rgba_from_rgb(self.cloud_thumbs[code], int(220 * (1 - 0.6 * k))), (px, py))
            self.tag(draw, (margin, int(self.config.height * 0.08)), "AVERAGE // z_avg = sum w_i z_i")
        else:
            k = _fast((p - 0.45) / 0.55)
            latent = self.assets.sdvae_latent
            if latent is not None and latent.ndim == 3:
                tw, th = int(self.config.width * 0.2), int(self.config.width * 0.2 * 0.75)
                gx, gy = cx - tw - 6, cy - th - 6
                for i, ch in enumerate(latent[:4]):
                    norm = (ch - ch.min()) / ((ch.max() - ch.min()) or 1.0)
                    fld = Image.fromarray(np.clip(norm * 255, 0, 255).astype(np.uint8)).convert("RGB").resize((tw, th))
                    paste_rgba(image, rgba_from_rgb(fld, int(230 * (1 - k))), (gx + (i % 2) * (tw + 12), gy + (i // 2) * (th + 12)))
            decoded = self.assets.generated.get(SUBJECT_INTENT, {}).get("sdvae")
            if decoded is not None and k > 0.05:
                w = int(self.config.width * 0.4)
                panel = fit_image(decoded, (w, int(w * 0.75)))
                paste_rgba(image, rgba_from_rgb(panel, int(245 * k)), ((self.config.width - panel.width) // 2, cy - panel.height // 2))
            self.tag(draw, (int(self.config.width * 0.1), int(self.config.height * 0.08)), "DECODE // SD-VAE latent -> image")
        return image

    # -- 11. weighting -----------------------------------------------------
    def phase_weighting(self, p):
        image = self.blank()
        draw = ImageDraw.Draw(image)
        n = len(INTENTS)
        cw, ch = int(self.config.width * 0.9 / n), int(self.config.width * 0.9 / n * 0.75)
        ox, oy = int((self.config.width - n * cw) / 2), int(self.config.height * 0.26)
        shown = int(n * _fast(p, 1.6)) + 1
        for i, intent in enumerate(INTENTS[:shown]):
            x = ox + i * cw
            img = self.preferred_image(intent)
            if img is not None:
                paste_rgba(image, rgba_from_rgb(fit_image(img, (cw - 8, ch)), 240), (x + 2, oy))
            rec, era = self.assets.recognition.get(intent), self.assets.erasure.get(intent)
            draw_text(draw, (x + 2, oy + ch + 4), intent, self.tiny, EDGE)
            if rec is not None:
                draw_text(draw, (x + 2, oy + ch + 17), f"->{rec.nearest.code.upper()} {rec.nearest.similarity:0.2f}", self.tiny, INK)
            if era is not None:
                draw_text(draw, (x + 2, oy + ch + 30), f"-{era.erased_count}", self.tiny, (220, 156, 156))
        self.tag(draw, (ox, oy - 26), "WEIGHT // the average shifts with the metric")
        return image

    # -- 12. regions (other perspectives) ----------------------------------
    def phase_regions(self, p):
        image = self.blank()
        draw = ImageDraw.Draw(image)
        regions = sorted(self.assets.region_averages.items())
        if not regions:
            return image
        n = len(regions)
        cw, ch = int(self.config.width * 0.9 / n), int(self.config.width * 0.9 / n * 0.62)
        ox, oy = int((self.config.width - n * cw) / 2), int(self.config.height * 0.3)
        shown = int(n * _fast(p, 1.6)) + 1
        for i, (region, rec) in enumerate(regions[:shown]):
            x = ox + i * cw
            thumb = self.archive_thumbs.get(rec.nearest.code)
            if thumb is not None:
                paste_rgba(image, rgba_from_rgb(fit_image(thumb, (cw - 8, ch)), 240), (x + 2, oy))
            draw_text(draw, (x + 2, oy - 16), region.upper(), self.tiny, MUTED)
            draw_text(draw, (x + 2, oy + ch + 4), f"->{rec.nearest.name[:16]}", self.tiny, INK)
            draw_text(draw, (x + 2, oy + ch + 17), f"cos {rec.nearest.similarity:0.3f}", self.tiny, EDGE)
        self.tag(draw, (ox, int(self.config.height * 0.2)), "REGIONS // the average nation, by world region")
        return image

    # -- 13. erase ---------------------------------------------------------
    def phase_erase(self, p):
        image = self.blank()
        draw = ImageDraw.Draw(image)
        era = self.assets.erasure.get(SUBJECT_INTENT)
        if era is None:
            return image
        erased = era.erased[:120]
        cols = 12
        cw, ch = int(self.config.width * 0.8 / cols), int(self.config.width * 0.8 / cols * 0.75)
        ox, oy = int(self.config.width * 0.1), int(self.config.height * 0.2)
        cutoff = _fast(p, 1.3)
        for i, contributor in enumerate(erased):
            r, c = divmod(i, cols)
            thumb = self.archive_thumbs.get(contributor.code)
            if thumb is None:
                continue
            x, y = ox + c * cw, oy + r * ch
            tile = fit_image(thumb, (cw - 5, ch - 5))
            if i < int(len(erased) * cutoff):
                tile = Image.eval(tile, lambda v: int(v * 0.07))
                draw.line((x, y, x + cw - 5, y + ch - 5), fill=(120, 40, 44), width=1)
            paste_rgba(image, rgba_from_rgb(tile, 235), (x, y))
        self.tag(draw, (ox, int(self.config.height * 0.1)), f"WEIGHT // {INTENT_LABELS.get(SUBJECT_INTENT, SUBJECT_INTENT)}")
        self.tag(draw, (ox, self.config.height - 50), f"{int(era.erased_count * cutoff)} / {era.total} nations averaged below 0.1%", (224, 150, 150))
        return image

    # -- 14. name ----------------------------------------------------------
    def phase_name(self, p):
        image = self.blank()
        draw = ImageDraw.Draw(image)
        rec = self.assets.recognition.get(SUBJECT_INTENT)
        tie = self.assets.recognition_tie
        x = int(self.config.width * 0.08)
        if rec is not None:
            self.tag(draw, (x, int(self.config.height * 0.2)), "the machine names the average:", MUTED, self.small)
            draw_text(draw, (x, int(self.config.height * 0.24)), rec.nearest.name, self.title_font, INK)
            draw_text(draw, (x, int(self.config.height * 0.24) + int(getattr(self.title_font, "size", 40) * 1.05)),
                      f"cos {rec.nearest.similarity:0.3f}   conf {rec.confidence:0.2f}   margin {rec.margin:0.3f}", self.label_font, ACCENT)
            if p > 0.4 and rec.probe_ranks:
                total = self.assets.erasure[SUBJECT_INTENT].total
                yy = int(self.config.height * 0.58)
                self.tag(draw, (x, yy - 26), "and the named, averaged away:", MUTED, self.small)
                for code, (rank, sim) in rec.probe_ranks.items():
                    draw_text(draw, (x, yy), f"{self.names.get(code, code):<22} rank {rank:>3}/{total}   cos {sim:0.3f}", self.small, (220, 156, 156))
                    yy += 22
        if tie is not None and len(tie.ranking) >= 2 and p > 0.55:
            rx, ry = int(self.config.width * 0.56), int(self.config.height * 0.58)
            a, b = tie.ranking[0], tie.ranking[1]
            self.tag(draw, (rx, ry - 26), "occupier and occupied, averaged:", MUTED, self.small)
            draw_text(draw, (rx, ry), f"{self.names.get(a.code, a.code)} = {self.names.get(b.code, b.code)}", self.h1_font, INK)
            draw_text(draw, (rx, ry + int(getattr(self.h1_font, "size", 24) * 1.2)), f"both cos {a.similarity:0.4f}  margin {tie.margin:0.4f}", self.label_font, (220, 156, 156))
        return image

    # -- 15. residual ------------------------------------------------------
    def phase_residual(self, p):
        image = self.blank()
        draw = ImageDraw.Draw(image)
        recon, energy = self.assets.pca_recon, self.assets.pca_recon_energy
        if not recon:
            return image
        idx = min(len(recon) - 1, int(_fast(p, 1.2) * len(recon)))
        steps = (1, 2, 4, 8, 16)
        w = int(self.config.width * 0.4)
        panel = fit_image(recon[idx], (w, int(w * 0.75)))
        x, y = int(self.config.width * 0.1), (self.config.height - panel.height) // 2
        paste_rgba(image, rgba_from_rgb(panel, 245), (x, y))
        self.tag(draw, (x, y - 22), f"RECONSTRUCT // {steps[min(idx, len(steps) - 1)]} eigenflags")
        if self.subject is not None and self.subject.residual is not None:
            res = self.subject.residual.heatmap().resize((w, int(w * 0.75)))
            rx = int(self.config.width * 0.54)
            paste_rgba(image, rgba_from_rgb(res.convert("RGB"), 235), (rx, y))
            self.tag(draw, (rx, y - 22), "residual // what the basis discards", MUTED, self.small)
        if idx < len(energy):
            self.tag(draw, (x, y + panel.height + 10), f"loss {energy[idx]:0.1f}/255", (220, 156, 156))
        return image

    # -- 16. coda ----------------------------------------------------------
    def phase_coda(self, p):
        image = self.blank()
        draw = ImageDraw.Draw(image)
        decoded = self.assets.generated.get(SUBJECT_INTENT, {}).get("sdvae")
        cx, cy = self.config.width // 2, self.config.height // 2
        if decoded is not None:
            w = int(self.config.width * 0.34)
            panel = fit_image(decoded, (w, int(w * 0.75)))
            paste_rgba(image, rgba_from_rgb(panel, max(0, int(210 * (1.0 - 0.7 * ease(p))))), (cx - panel.width // 2, cy - panel.height // 2 - 30))
        if p > 0.2:
            draw_text(draw, (cx, int(self.config.height * 0.72)), "the average is never neutral", self.h1_font, INK, anchor="mm")
        if p > 0.45:
            draw_text(draw, (cx, int(self.config.height * 0.81)), "every value in this film is a real computer-vision measurement", self.small, FAINT, anchor="mm")
        if p > 0.68:
            draw_text(draw, (cx, int(self.config.height * 0.9)), f"{TITLE.upper()}  ·  {AUTHOR}  ·  {WEBSITE}", self.tiny, MUTED, anchor="mm")
        return image

    # -- glitch (fast, short bursts) ---------------------------------------
    def glitch_for(self, key, local_t, dur):
        bursts, mode = self.GLITCH.get(key, ([], "block"))
        g = 0.02
        for frac, amp in bursts:
            g = max(g, amp * max(0.0, 1.0 - abs(local_t - frac * dur) / 0.10))
        return g, mode

    # -- timeline ----------------------------------------------------------
    def _phase_at(self, t):
        for i, (key, start, end) in enumerate(self.segments):
            if start <= t < end:
                return i, key, start, end
        last = self.segments[-1]
        return len(self.segments) - 1, last[0], last[1], last[2]

    def render(self, frame_index):
        t = frame_index / self.config.fps
        self._t = t
        duration = self.config.duration
        i, key, start, end = self._phase_at(t)
        p = (t - start) / max(1e-6, end - start)
        frame = self.phase_methods[key](min(1.0, p * SPEED))  # 15% faster
        if end - t < self.XFADE and i < len(self.segments) - 1:
            nxt = self.phase_methods[self.segments[i + 1][0]](0.0)
            frame = Image.blend(frame, nxt, (self.XFADE - (end - t)) / self.XFADE * 0.5)
        if key not in self.NO_HUD:
            self.hud(frame, key, t)
        amount, mode = self.glitch_for(key, t - start, end - start)
        frame = effects.treat(frame, vmask=self.vmask, glitch=amount, glitch_mode=mode,
                              seed=self.config.seed, frame=frame_index, grain_amount=5.0, aberration=1)
        fade = max(0.0, min(min(1.0, t / 1.2), min(1.0, (duration - t) / 2.5)))
        if fade < 1.0:
            frame = Image.blend(self.blank(), frame, fade)
        return frame

    def preferred_image(self, intent):
        for backend in ("sdvae", "pca", "svg", "palette", "pixel"):
            img = self.assets.generated.get(intent, {}).get(backend)
            if img is not None:
                return img
        return None
