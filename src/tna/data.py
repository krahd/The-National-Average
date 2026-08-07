"""Corpus loading and rasterisation for the representation ladder."""

from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "critical-averaging-mpl"),
)

import cairosvg
import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
FLAGS_SVG_DIR = DATA_DIR / "flags_svg"
FLAGS_PNG_DIR = DATA_DIR / "flags_png"
METADATA_PATH = DATA_DIR / "metadata.csv"
DEFAULT_CANVAS = (96, 72)  # 4:3, matching the lipis flag-icons aspect ratio

# Supranational, subnational, and placeholder lipis codes excluded so the
# learned representation spaces approximate sovereign / quasi-sovereign
# polities. Some dependent territories remain; see README.
NON_NATION_CODES = {
    "eu", "un", "arab", "asean", "cefta", "eac",      # supranational organisations
    "gb-eng", "gb-sct", "gb-wls", "gb-nir",            # UK constituent nations
    "es-ct", "es-ga", "es-pv",                         # Spanish autonomous communities
    "sh-ac", "sh-hl", "sh-ta",                         # St Helena dependencies
    "ic", "cp", "dg", "xx",                            # special / placeholder codes
}


@dataclass(frozen=True)
class MusicProfile:
    """Small symbolic music profile used by the default music backend."""

    tempo_bpm: float
    mode: str
    meter_beats: int
    root_midi: int
    motif_degrees: tuple[int, ...]
    motif_durations: tuple[float, ...]


@dataclass(frozen=True)
class Polity:
    """One row of corpus metadata plus the path to its source SVG."""

    code: str
    name: str
    population_millions: float
    area_km2: float
    region: str
    subregion: str
    un_member: bool
    recognition_year: int
    metadata_quality: str
    svg_path: Path
    music: MusicProfile
    metadata: dict[str, str]


WORKED_EXAMPLE_MUSIC = {
    # These profiles are illustrative musical descriptors, not national anthem
    # transcriptions. The goal is transparent symbolic averaging.
    "fr": MusicProfile(112, "major", 4, 60, (5, 5, 5, 1, 2, 2, 5, 3), (0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 1.0)),
    "uy": MusicProfile(96, "major", 4, 62, (1, 3, 5, 5, 6, 5, 3, 1), (1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 1.0)),
    "ps": MusicProfile(104, "minor", 4, 57, (1, 2, 3, 5, 4, 3, 2, 1), (0.75, 0.25, 0.5, 1.0, 0.5, 0.5, 1.0, 1.0)),
}

DEFAULT_MUSIC = MusicProfile(
    tempo_bpm=100,
    mode="major",
    meter_beats=4,
    root_midi=60,
    motif_degrees=(1, 2, 3, 5, 3, 2, 1, 5),
    motif_durations=(0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 1.0),
)


def parse_canvas(text: str) -> tuple[int, int]:
    width_text, _, height_text = text.lower().partition("x")
    if not width_text or not height_text:
        raise ValueError("canvas must be WIDTHxHEIGHT, e.g. 96x64")
    width, height = int(width_text), int(height_text)
    if width < 16 or height < 16:
        raise ValueError("canvas dimensions must be at least 16px")
    return width, height


def load_metadata(path: Path = METADATA_PATH) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["code"]: row for row in csv.DictReader(handle)}


def load_corpus(
    codes: list[str] | None = None,
    data_dir: Path = DATA_DIR,
) -> dict[str, Polity]:
    """Load metadata and attach SVG paths for all usable corpus entries."""

    metadata = load_metadata(data_dir / "metadata.csv")
    svg_dir = data_dir / "flags_svg"
    requested = set(codes) if codes else None
    if requested is not None:
        blocked = requested & NON_NATION_CODES
        if blocked:
            raise ValueError(
                "these codes are excluded as non-national flags: "
                + ", ".join(sorted(blocked))
            )
    corpus: dict[str, Polity] = {}
    for code, row in sorted(metadata.items()):
        if code in NON_NATION_CODES:
            continue
        if requested is not None and code not in requested:
            continue
        svg_path = svg_dir / f"{code}.svg"
        if not svg_path.exists():
            continue
        corpus[code] = Polity(
            code=code,
            name=row["name"],
            population_millions=float(row["population_millions"]),
            area_km2=float(row["area_km2"]),
            region=row["region"],
            subregion=row["subregion"],
            un_member=row["un_member"].lower() == "true",
            recognition_year=int(float(row["recognition_year"])),
            metadata_quality=row.get("metadata_quality", "unknown"),
            svg_path=svg_path,
            music=WORKED_EXAMPLE_MUSIC.get(code, DEFAULT_MUSIC),
            metadata=dict(row),
        )
    if requested:
        missing = sorted(requested - set(corpus))
        if missing:
            raise ValueError(f"unknown entity code(s): {', '.join(missing)}")
    return corpus


def rasterize_flag(polity: Polity, canvas: tuple[int, int], cache_dir: Path = FLAGS_PNG_DIR) -> Image.Image:
    width, height = canvas
    size_dir = cache_dir / f"{width}x{height}"
    cache_path = size_dir / f"{polity.code}.png"
    if cache_path.exists():
        return Image.open(cache_path).convert("RGB")
    size_dir.mkdir(parents=True, exist_ok=True)
    png = cairosvg.svg2png(url=str(polity.svg_path), output_width=width, output_height=height)
    rendered = Image.open(io.BytesIO(png)).convert("RGBA")
    # Flatten transparent padding (from aspect-ratio fit) onto white, not black.
    background = Image.new("RGBA", rendered.size, (255, 255, 255, 255))
    image = Image.alpha_composite(background, rendered).convert("RGB")
    image.save(cache_path)
    return image


def corpus_arrays(corpus: dict[str, Polity], canvas: tuple[int, int]) -> dict[str, np.ndarray]:
    """Rasterise the corpus once for backends that operate on image arrays."""

    return {
        code: np.asarray(rasterize_flag(polity, canvas), dtype=np.float64)
        for code, polity in corpus.items()
    }


def selected_items(corpus: dict[str, Polity], codes: list[str]) -> list[Polity]:
    missing = [code for code in codes if code not in corpus]
    if missing:
        raise ValueError(f"unknown entity code(s): {', '.join(missing)}")
    return [corpus[code] for code in codes]


def serialise_polity(polity: Polity) -> dict[str, object]:
    """Convert dataclass metadata to JSON-safe primitives for traces."""

    base_fields = {
        "code",
        "name",
        "population_millions",
        "area_km2",
        "region",
        "subregion",
        "un_member",
        "recognition_year",
        "metadata_quality",
    }
    return {
        "code": polity.code,
        "name": polity.name,
        "population_millions": polity.population_millions,
        "area_km2": polity.area_km2,
        "region": polity.region,
        "subregion": polity.subregion,
        "un_member": polity.un_member,
        "recognition_year": polity.recognition_year,
        "metadata_quality": polity.metadata_quality,
        "metadata": {
            key: value
            for key, value in polity.metadata.items()
            if key not in base_fields and value not in ("", None)
        },
        "music": {
            "tempo_bpm": polity.music.tempo_bpm,
            "mode": polity.music.mode,
            "meter_beats": polity.music.meter_beats,
            "root_midi": polity.music.root_midi,
            "motif_degrees": list(polity.music.motif_degrees),
            "motif_durations": list(polity.music.motif_durations),
        },
    }
