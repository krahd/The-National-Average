"""Build the flag metadata CSV from audited bulk data sources.

The script intentionally reads source snapshots from disk instead of fetching
them itself. That keeps network access explicit and lets the committed CSV be
rebuilt from archived inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = Path("/tmp/tna_metadata")
DEFAULT_INPUT = ROOT / "data" / "metadata.csv"
DEFAULT_OUTPUT = ROOT / "data" / "metadata.csv"
DEFAULT_COVERAGE = ROOT / "data" / "metadata_coverage.json"

SOURCE_URLS = {
    "countries.json": "https://raw.githubusercontent.com/mledoze/countries/master/countries.json",
    "worldbank_countries.json": "https://api.worldbank.org/v2/country?format=json&per_page=400",
    "worldbank_population.json": "https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?format=json&per_page=20000&mrv=5",
    "worldbank_gdp.json": "https://api.worldbank.org/v2/country/all/indicator/NY.GDP.MKTP.CD?format=json&per_page=20000&mrv=5",
    "owid-co2-data.csv": "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv",
    "wikidata_iso2.json": "https://query.wikidata.org/sparql",
}

WORKED_EXAMPLE_RECOGNITION = {
    "fr": "843",
    "uy": "1828",
    "ps": "1988",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_existing(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def clean_join(items: Any) -> str:
    if not items:
        return ""
    if isinstance(items, dict):
        items = items.values()
    return ";".join(str(item) for item in items if item not in (None, ""))


def format_float(value: Any, digits: int = 6) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):.{digits}f}".rstrip("0").rstrip(".")


def reusable_previous_value(row: dict[str, str], value_key: str, source_key: str) -> str:
    value = row.get(value_key, "")
    if not value or value in {"1", "1.0", "1.000000"}:
        return ""
    source = row.get(source_key, "")
    if source == "fallback_placeholder":
        return ""
    return value


def year_from_wikidata(value: dict[str, str] | None) -> str:
    if not value:
        return ""
    text = value.get("value", "")
    if not text or text.startswith("-"):
        return ""
    return text[:4]


def latest_world_bank_indicator(source_dir: Path, filename: str, iso3_to_iso2: dict[str, str]) -> dict[str, tuple[float, int]]:
    records = read_json(source_dir / filename)[1]
    output: dict[str, tuple[float, int]] = {}
    for record in records:
        iso2 = iso3_to_iso2.get(record.get("countryiso3code", ""))
        value = record.get("value")
        if not iso2 or value is None:
            continue
        year = int(record["date"])
        if iso2 not in output or year > output[iso2][1]:
            output[iso2] = (float(value), year)
    return output


def latest_owid_records(source_dir: Path, iso3_to_iso2: dict[str, str], countries: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    country_iso3_to_iso2 = {
        country.get("cca3", ""): country.get("cca2", "").lower()
        for country in countries
        if country.get("cca3") and country.get("cca2")
    }
    output: dict[str, dict[str, str]] = {}
    with (source_dir / "owid-co2-data.csv").open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            iso3 = record.get("iso_code", "")
            iso2 = iso3_to_iso2.get(iso3) or country_iso3_to_iso2.get(iso3)
            if not iso2:
                continue
            try:
                year = int(record["year"])
            except (TypeError, ValueError):
                continue
            if iso2 not in output or year > int(output[iso2]["year"]):
                output[iso2] = record
    return output


def wikidata_by_iso2(source_dir: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    bindings = read_json(source_dir / "wikidata_iso2.json")["results"]["bindings"]
    for binding in bindings:
        iso2 = binding["iso2"]["value"].lower()
        entry = output.setdefault(
            iso2,
            {"labels": set(), "inception_years": set(), "un_start_years": set(), "areas": set()},
        )
        if "itemLabel" in binding:
            entry["labels"].add(binding["itemLabel"]["value"])
        if "inception" in binding:
            entry["inception_years"].add(year_from_wikidata(binding["inception"]))
        if "unStart" in binding:
            entry["un_start_years"].add(year_from_wikidata(binding["unStart"]))
        if "area" in binding:
            entry["areas"].add(binding["area"]["value"])

    for entry in output.values():
        entry["inception_year"] = min([value for value in entry["inception_years"] if value], default="")
        entry["un_membership_year"] = min([value for value in entry["un_start_years"] if value], default="")
        areas = []
        for area in entry["areas"]:
            try:
                areas.append(float(area))
            except ValueError:
                pass
        entry["area_km2"] = max(areas) if areas else None
    return output


def build_rows(existing_rows: list[dict[str, str]], source_dir: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    countries = read_json(source_dir / "countries.json")
    country_by_iso2 = {
        country.get("cca2", "").lower(): country
        for country in countries
        if country.get("cca2")
    }

    world_bank_countries = read_json(source_dir / "worldbank_countries.json")[1]
    wb_by_iso2 = {
        country["iso2Code"].lower(): country
        for country in world_bank_countries
        if country.get("iso2Code") and country.get("region", {}).get("value") != "Aggregates"
    }
    wb_iso3_to_iso2 = {
        country["id"]: country["iso2Code"].lower()
        for country in world_bank_countries
        if country.get("iso2Code") and country.get("region", {}).get("value") != "Aggregates"
    }
    wb_population = latest_world_bank_indicator(source_dir, "worldbank_population.json", wb_iso3_to_iso2)
    wb_gdp = latest_world_bank_indicator(source_dir, "worldbank_gdp.json", wb_iso3_to_iso2)
    owid = latest_owid_records(source_dir, wb_iso3_to_iso2, countries)
    wikidata = wikidata_by_iso2(source_dir)

    rows: list[dict[str, str]] = []
    for old in existing_rows:
        code = old["code"].lower()
        country = country_by_iso2.get(code)
        world_bank = wb_by_iso2.get(code)
        wd = wikidata.get(code, {})
        owid_record = owid.get(code, {})

        row: dict[str, str] = {
            "code": code,
            "name": (country and country.get("name", {}).get("common"))
            or (world_bank and world_bank.get("name"))
            or old["name"],
            "official_name": (country and country.get("name", {}).get("official")) or "",
            "iso2": (country and country.get("cca2")) or (world_bank and world_bank.get("iso2Code")) or code.upper(),
            "iso3": (country and country.get("cca3")) or (world_bank and world_bank.get("id")) or "",
        }
        row["official_name"] = row["official_name"] or row["name"]

        if code in wb_population:
            population, year = wb_population[code]
            row["population_millions"] = format_float(population / 1_000_000)
            row["population_year"] = str(year)
            row["population_source"] = "World Bank WDI SP.POP.TOTL"
        elif owid_record.get("population"):
            row["population_millions"] = format_float(float(owid_record["population"]) / 1_000_000)
            row["population_year"] = owid_record.get("year", "")
            row["population_source"] = "Our World in Data CO2 population"
        elif reusable_previous_value(old, "population_millions", "population_source"):
            row["population_millions"] = reusable_previous_value(old, "population_millions", "population_source")
            row["population_year"] = ""
            row["population_source"] = "previous_metadata"
        else:
            row["population_millions"] = "1"
            row["population_year"] = ""
            row["population_source"] = "fallback_placeholder"

        if country and country.get("area") not in (None, ""):
            row["area_km2"] = format_float(country["area"])
            row["area_source"] = "mledoze/countries area"
        elif wd.get("area_km2"):
            row["area_km2"] = format_float(wd["area_km2"])
            row["area_source"] = "Wikidata P2046 area"
        elif reusable_previous_value(old, "area_km2", "area_source"):
            row["area_km2"] = reusable_previous_value(old, "area_km2", "area_source")
            row["area_source"] = "previous_metadata"
        else:
            row["area_km2"] = "1"
            row["area_source"] = "fallback_placeholder"

        row["region"] = (country and country.get("region")) or (
            world_bank and world_bank.get("region", {}).get("value", "").strip()
        ) or old.get("region", "")
        row["subregion"] = (country and country.get("subregion")) or (
            world_bank and world_bank.get("adminregion", {}).get("value", "").strip()
        ) or old.get("subregion", "")
        row["worldbank_region"] = (world_bank and world_bank.get("region", {}).get("value", "").strip()) or ""
        row["worldbank_income_level"] = (world_bank and world_bank.get("incomeLevel", {}).get("value")) or ""
        row["worldbank_lending_type"] = (world_bank and world_bank.get("lendingType", {}).get("value")) or ""
        row["independent"] = str(bool(country.get("independent"))).lower() if country and "independent" in country else ""
        row["status"] = (country and country.get("status")) or ""
        row["un_member"] = str(bool(country.get("unMember"))).lower() if country and "unMember" in country else old.get("un_member", "")
        row["un_regional_group"] = (country and country.get("unRegionalGroup")) or ""
        row["capital"] = clean_join(country.get("capital") if country else []) or (world_bank and world_bank.get("capitalCity")) or ""

        latlng = country.get("latlng") if country else None
        if latlng and len(latlng) >= 2:
            row["latitude"] = format_float(latlng[0], 8)
            row["longitude"] = format_float(latlng[1], 8)
            row["coordinates_source"] = "mledoze/countries latlng"
        elif world_bank and world_bank.get("latitude") and world_bank.get("longitude"):
            row["latitude"] = world_bank["latitude"]
            row["longitude"] = world_bank["longitude"]
            row["coordinates_source"] = "World Bank country metadata"
        else:
            row["latitude"] = ""
            row["longitude"] = ""
            row["coordinates_source"] = ""

        row["landlocked"] = str(bool(country.get("landlocked"))).lower() if country and "landlocked" in country else ""
        row["borders"] = clean_join(country.get("borders") if country else [])
        row["currencies"] = clean_join((country.get("currencies") or {}).keys()) if country else ""
        row["languages"] = clean_join((country.get("languages") or {}).values()) if country else ""
        row["inception_year"] = wd.get("inception_year") or ""
        row["un_membership_year"] = wd.get("un_membership_year") or ""

        if code in WORKED_EXAMPLE_RECOGNITION:
            row["recognition_year"] = WORKED_EXAMPLE_RECOGNITION[code]
            row["recognition_year_source"] = "previous_worked_example"
            row["recognition_year_confidence"] = "worked_example"
        elif row["un_membership_year"]:
            row["recognition_year"] = row["un_membership_year"]
            row["recognition_year_source"] = "Wikidata UN membership start (P463/P580)"
            row["recognition_year_confidence"] = "medium"
        elif row["inception_year"]:
            row["recognition_year"] = row["inception_year"]
            row["recognition_year_source"] = "Wikidata inception (P571)"
            row["recognition_year_confidence"] = "low"
        else:
            row["recognition_year"] = old.get("recognition_year") or "1900"
            row["recognition_year_source"] = "fallback_placeholder"
            row["recognition_year_confidence"] = "none"

        if code in wb_gdp:
            gdp, year = wb_gdp[code]
            row["gdp_current_usd"] = format_float(gdp, 2)
            row["gdp_year"] = str(year)
            row["gdp_source"] = "World Bank WDI NY.GDP.MKTP.CD"
        elif owid_record.get("gdp"):
            row["gdp_current_usd"] = format_float(owid_record["gdp"], 2)
            row["gdp_year"] = owid_record.get("year", "")
            row["gdp_source"] = "Our World in Data CO2 gdp"
        else:
            row["gdp_current_usd"] = ""
            row["gdp_year"] = ""
            row["gdp_source"] = ""

        row["co2_year"] = owid_record.get("year", "")
        co2_fields = {
            "co2_mt": "co2",
            "co2_per_capita": "co2_per_capita",
            "consumption_co2_mt": "consumption_co2",
            "consumption_co2_per_capita": "consumption_co2_per_capita",
            "cumulative_co2_mt": "cumulative_co2",
            "co2_including_luc_mt": "co2_including_luc",
            "total_ghg_mt": "total_ghg",
            "total_ghg_excluding_lucf_mt": "total_ghg_excluding_lucf",
            "ghg_per_capita": "ghg_per_capita",
            "primary_energy_consumption_twh": "primary_energy_consumption",
            "energy_per_capita_kwh": "energy_per_capita",
            "share_global_co2_pct": "share_global_co2",
            "temperature_change_from_co2_c": "temperature_change_from_co2",
        }
        has_co2 = False
        for output_key, input_key in co2_fields.items():
            value = owid_record.get(input_key, "")
            row[output_key] = value
            has_co2 = has_co2 or value not in ("", None)
        row["co2_source"] = "Our World in Data CO2 dataset" if has_co2 else ""

        source_values = []
        for key in ["population_source", "area_source", "gdp_source", "co2_source", "recognition_year_source"]:
            value = row.get(key)
            if value and value not in source_values:
                source_values.append(value)
        row["metadata_sources"] = ";".join(source_values)

        if row["population_source"] != "fallback_placeholder" and row["area_source"] != "fallback_placeholder" and (country or world_bank):
            row["metadata_quality"] = "bulk_enriched"
        elif country or world_bank or has_co2:
            row["metadata_quality"] = "partial_enriched"
        else:
            row["metadata_quality"] = "placeholder"

        rows.append(row)

    coverage = {
        "row_count": len(rows),
        "column_count": len(rows[0]) if rows else 0,
        "source_urls": SOURCE_URLS,
        "counts": {
            "metadata_quality": Counter(row["metadata_quality"] for row in rows),
            "population_source": Counter(row["population_source"] for row in rows),
            "area_source": Counter(row["area_source"] for row in rows),
            "gdp_source": Counter(row["gdp_source"] for row in rows),
            "co2_source": Counter(row["co2_source"] for row in rows),
            "recognition_year_confidence": Counter(row["recognition_year_confidence"] for row in rows),
        },
        "placeholder_codes": [row["code"] for row in rows if row["metadata_quality"] == "placeholder"],
    }
    coverage["counts"] = {key: dict(value) for key, value in coverage["counts"].items()}
    return rows, coverage


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate data/metadata.csv from source snapshots.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    args = parser.parse_args()

    rows, coverage = build_rows(read_existing(args.input), args.source_dir)
    write_csv(args.output, rows)
    args.coverage.write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote {len(rows)} rows and {len(rows[0]) if rows else 0} columns to {args.output}")
    print(f"Wrote coverage summary to {args.coverage}")


if __name__ == "__main__":
    main()
