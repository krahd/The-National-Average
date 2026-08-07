# Metadata sources and provenance

The metadata is one of the materials of **The National Average**. Weighting the same set of flags by population, area, GDP, emissions, energy use, or historical variables produces different averages; the provenance and limitations of those quantities are therefore part of the research rather than incidental implementation detail.

`data/metadata.csv` contains the country- and territory-level metadata used for description and weighting. It was assembled from public sources in June 2026 and is accompanied by field-level source information and coverage metadata.

## Rebuilding the metadata

The builder reads source snapshots from `/tmp/tna_metadata`:

```bash
mkdir -p /tmp/tna_metadata
curl -sL 'https://raw.githubusercontent.com/mledoze/countries/master/countries.json' -o /tmp/tna_metadata/countries.json
curl -sL 'https://api.worldbank.org/v2/country?format=json&per_page=400' -o /tmp/tna_metadata/worldbank_countries.json
curl -sL 'https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?format=json&per_page=20000&mrv=5' -o /tmp/tna_metadata/worldbank_population.json
curl -sL 'https://api.worldbank.org/v2/country/all/indicator/NY.GDP.MKTP.CD?format=json&per_page=20000&mrv=5' -o /tmp/tna_metadata/worldbank_gdp.json
curl -sL 'https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv' -o /tmp/tna_metadata/owid-co2-data.csv
curl -sL --get 'https://query.wikidata.org/sparql' \
  --data-urlencode 'format=json' \
  --data-urlencode 'query=SELECT ?iso2 ?itemLabel ?inception ?area ?unStart WHERE { ?item wdt:P297 ?iso2 . OPTIONAL { ?item wdt:P571 ?inception . } OPTIONAL { ?item wdt:P2046 ?area . } OPTIONAL { ?item p:P463 ?st . ?st ps:P463 wd:Q1065 . OPTIONAL { ?st pq:P580 ?unStart . } } SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } }' \
  -o /tmp/tna_metadata/wikidata_iso2.json
```

Then regenerate the committed metadata:

```bash
python scripts/build_metadata.py --source-dir /tmp/tna_metadata
```

The script writes:

- `data/metadata.csv`
- `data/metadata_coverage.json`

## Sources and their roles

| Source | Columns populated |
|---|---|
| `mledoze/countries` | names, ISO codes, area, region/subregion, independent/status, UN membership, UN regional group, capital, coordinates, landlocked, borders, currencies, languages |
| World Bank WDI country metadata | World Bank region, income level, lending type, fallback capital/coordinates, ISO2/ISO3 crosswalk |
| World Bank WDI `SP.POP.TOTL` | `population_millions`, `population_year` |
| UN DESA World Population Prospects 2024 | Western Sahara population (2024 medium variant) |
| World Bank WDI `NY.GDP.MKTP.CD` | `gdp_current_usd`, `gdp_year` |
| Our World in Data CO2 dataset | current annual CO2, cumulative CO2, greenhouse-gas, energy, and fallback population/GDP fields |
| Wikidata SPARQL | `inception_year`, `un_membership_year`, fallback `area_km2` where missing |

## Recognition-year field

`recognition_year` is retained for compatibility, but it is not a uniform independence-date variable. Depending on the entry, the available historical marker may refer to UN membership, inception, or a hand-checked project value.

The builder resolves it as follows:

1. Keep the hand-checked values used by the France–Uruguay–Palestine example.
2. Use Wikidata UN membership start year (`member of` United Nations with `start time`) when present.
3. Use Wikidata `inception` year for non-UN members and territories when present.
4. Fall back to `1900` only when no sourced date exists.

`recognition_year_source` and `recognition_year_confidence` should always be read with the value itself. The CLI rejects `recognition_year` weighting for rows whose source is only `fallback_placeholder`.

## Coverage

The current generated coverage is recorded in `data/metadata_coverage.json`:

- rows: 271
- columns: 52
- `bulk_enriched`: 226 rows
- `partial_enriched`: 24 rows
- `placeholder`: 21 rows

The placeholder rows are supranational, subnational, or special flag codes already excluded from the default analytical corpus, including `eu`, `un`, `arab`, `asean`, `gb-eng`, and the Spanish autonomous-community flags.

`metadata_quality` is a row-level coverage indicator rather than a guarantee that every field in the row is sourced:

- `bulk_enriched`: population and area are sourced and the row matched a country or territory record.
- `partial_enriched`: at least one source matched, but one of the core fields is missing or placeholder.
- `placeholder`: no reliable bulk source matched the local flag code.

For field-level decisions, use the corresponding `*_source`, `*_year`, and confidence columns.

## Units

| Column | Unit |
|---|---|
| `population_millions` | millions of people |
| `area_km2` | square kilometres |
| `gdp_current_usd` | current US dollars |
| `co2_mt` | million tonnes of CO2 |
| `consumption_co2_mt` | million tonnes of CO2 |
| `cumulative_co2_mt` | million tonnes of CO2 |
| `co2_including_luc_mt` | million tonnes of CO2 including land-use change |
| `total_ghg_mt` | million tonnes of CO2-equivalent |
| `primary_energy_consumption_twh` | terawatt-hours |
| `energy_per_capita_kwh` | kilowatt-hours per person |
| `share_global_co2_pct` | percent share |
| `temperature_change_from_co2_c` | degrees Celsius contribution |

## Licensing

The project MIT licence does not relicense this database. The combined metadata incorporates `mledoze/countries`, whose database is licensed under ODbL 1.0; `data/metadata.csv` is therefore distributed subject to ODbL 1.0 to the extent that licence applies to the derived database. Incorporated fields also retain source-specific attribution and terms, including World Bank WDI dataset terms, Our World in Data and underlying-provider terms, Wikidata CC0, and the applicable UN DESA terms.

See `../LICENSE-data.md` for the distribution notice and source-licence links.
