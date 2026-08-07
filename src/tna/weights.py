"""Weighting intents for critical averaging."""

from __future__ import annotations

from dataclasses import dataclass

from .data import Polity


INTENT_ALIASES = {
    # User-facing short names map to numeric metadata columns. New numeric
    # criteria can be added here once metadata.csv contains trustworthy values.
    "population": "population_millions",
    "population_millions": "population_millions",
    "area": "area_km2",
    "area_km2": "area_km2",
    "recognition_year": "recognition_year",
    "gdp": "gdp_current_usd",
    "gdp_current_usd": "gdp_current_usd",
    "co2": "co2_mt",
    "carbon": "co2_mt",
    "co2_mt": "co2_mt",
    "consumption_co2": "consumption_co2_mt",
    "cumulative_co2": "cumulative_co2_mt",
    "ghg": "total_ghg_mt",
    "energy": "primary_energy_consumption_twh",
}


SOURCE_COLUMNS = {
    "population_millions": "population_source",
    "area_km2": "area_source",
    "recognition_year": "recognition_year_source",
    "gdp_current_usd": "gdp_source",
    "co2_mt": "co2_source",
    "consumption_co2_mt": "co2_source",
    "cumulative_co2_mt": "co2_source",
    "total_ghg_mt": "co2_source",
    "primary_energy_consumption_twh": "co2_source",
}


@dataclass(frozen=True)
class WeightingRun:
    """Both raw and normalised weights for one CLI intent."""

    name: str
    source: str
    raw_weights: dict[str, float]
    weights: dict[str, float]


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def normalise_weights(raw: dict[str, float]) -> dict[str, float]:
    """Convert arbitrary non-negative weights into a probability distribution."""

    if any(value < 0 for value in raw.values()):
        raise ValueError("weights must be non-negative")
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("at least one weight must be positive")
    return {code: value / total for code, value in raw.items()}


def metadata_value(polity: Polity, criterion: str) -> float:
    """Read a numeric weighting criterion from typed or CSV metadata fields."""

    if hasattr(polity, criterion):
        return float(getattr(polity, criterion))
    value = polity.metadata.get(criterion, "")
    if value in ("", None):
        raise ValueError(f"{polity.code} has no value for metadata field {criterion!r}")
    return float(value)


def _looks_numeric(value: str | None) -> bool:
    if value in ("", None):
        return False
    try:
        float(value)
    except ValueError:
        return False
    return True


def parse_manual_weights(text: str, selected: list[Polity]) -> dict[str, float]:
    """Parse ``code=value`` input and require complete coverage of selection."""

    values: dict[str, float] = {}
    for part in parse_csv(text):
        code, separator, value = part.partition("=")
        if separator != "=":
            raise ValueError("manual weights must use code=value pairs")
        values[code.strip()] = float(value)
    selected_codes = {polity.code for polity in selected}
    unknown = sorted(set(values) - selected_codes)
    missing = sorted(selected_codes - set(values))
    if unknown:
        raise ValueError(f"manual weights include unselected code(s): {', '.join(unknown)}")
    if missing:
        raise ValueError(f"manual weights missing selected code(s): {', '.join(missing)}")
    return values


def numeric_criteria(items: list[Polity]) -> list[str]:
    """Return metadata fields that can currently drive weighting intents."""

    if not items:
        return []
    criteria = {"population_millions", "area_km2", "recognition_year"}
    for key in items[0].metadata:
        if key in SOURCE_COLUMNS or key.endswith("_source"):
            continue
        if key.endswith("_year") and key not in {"recognition_year"}:
            continue
        if key in {"code", "iso2", "iso3", "latitude", "longitude"}:
            continue
        if all(_looks_numeric(polity.metadata.get(key)) for polity in items):
            criteria.add(key)
    criteria.update(INTENT_ALIASES.values())
    return sorted(criteria)


def _require_real_metadata(selected: list[Polity], criterion: str) -> None:
    # Refusing placeholder criteria is better than producing impressive but
    # meaningless weighted outputs. Source columns are criterion-specific so a
    # row can have reliable population but no defensible recognition date.
    missing = []
    source_column = SOURCE_COLUMNS.get(criterion)
    for polity in selected:
        try:
            metadata_value(polity, criterion)
        except ValueError:
            missing.append(polity.code)
            continue
        if source_column:
            source = polity.metadata.get(source_column, "")
            if not source or source == "fallback_placeholder":
                missing.append(polity.code)
        if criterion == "recognition_year" and polity.metadata.get("recognition_year_confidence") == "none":
            missing.append(polity.code)
    if missing:
        raise ValueError(
            f"weighting by {criterion!r} requires sourced numeric metadata; "
            f"missing or fallback-only entities: {', '.join(sorted(set(missing)))}."
        )


def weights_from_intent(
    intent: str,
    selected: list[Polity],
    manual: dict[str, float] | None = None,
) -> WeightingRun:
    """Resolve a named CLI weighting intent into raw and normalised weights."""

    if intent == "manual":
        if manual is None:
            raise ValueError("--intents manual requires --manual-weights")
        raw = {polity.code: manual[polity.code] for polity in selected}
        return WeightingRun("manual", "manual", raw, normalise_weights(raw))
    if intent == "equal":
        raw = {polity.code: 1.0 for polity in selected}
        return WeightingRun("equal", "equal", raw, normalise_weights(raw))
    if intent.startswith("population_alpha:"):
        alpha = float(intent.split(":", 1)[1])
        _require_real_metadata(selected, "population_millions")
        raw = {polity.code: polity.population_millions**alpha for polity in selected}
        return WeightingRun(f"population_alpha_{alpha:g}", f"population_millions^{alpha:g}", raw, normalise_weights(raw))

    criterion = INTENT_ALIASES.get(intent, intent)
    if criterion not in numeric_criteria(selected):
        supported = ["equal", "manual", "population_alpha:<alpha>"] + sorted(INTENT_ALIASES)
        raise ValueError(f"unknown intent {intent!r}; supported intents include: {', '.join(supported)}")
    _require_real_metadata(selected, criterion)
    raw = {polity.code: metadata_value(polity, criterion) for polity in selected}
    return WeightingRun(intent, criterion, raw, normalise_weights(raw))
