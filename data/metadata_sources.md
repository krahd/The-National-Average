# Metadata Sources and Provenance

`data/metadata.csv` is the metadata table used by the flag corpus. It was
rebuilt from bulk public sources in June 2026 so the project can use weighting
criteria beyond the original worked example.

## Rebuild Inputs

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

## Source Roles

| Source | Columns populated |
|---|---|
| `mledoze/countries` | names, ISO codes, area, region/subregion, independent/status, UN membership, UN regional group, capital, coordinates, landlocked, borders, currencies, languages |
| World Bank WDI country metadata | World Bank region, income level, lending type, fallback capital/coordinates, ISO2/ISO3 crosswalk |
| World Bank WDI `SP.POP.TOTL` | `population_millions`, `population_year` |
| UN DESA World Population Prospects 2024 | Western Sahara population (2024 medium variant) |
| World Bank WDI `NY.GDP.MKTP.CD` | `gdp_current_usd`, `gdp_year` |
| Our World in Data CO2 dataset | current annual CO2, cumulative CO2, greenhouse-gas, energy, and fallback population/GDP fields |
| Wikidata SPARQL | `inception_year`, `un_membership_year`, fallback `area_km2` where missing |

## Licensing

The project MIT license does not relicense this database. The combined metadata
incorporates `mledoze/countries`, whose database is licensed under ODbL 1.0;
`data/metadata.csv` is therefore distributed subject to ODbL 1.0 to the extent
that license applies to the derived database. Incorporated fields also retain
source-specific attribution and terms, including World Bank WDI dataset terms,
Our World in Data and underlying-provider terms, Wikidata CC0, and the
applicable UN DESA terms.

See the repository-level `LICENSE-data.md` for the distribution notice and
source-license links.

## Recognition Year Caveat

The old CSV already had a `recognition_year` column, so it is retained for code
compatibility. It is not a uniform independence-date field.

The builder fills it as follows:

1. Keep the previous hand-vetted worked-example values for `fr`, `uy`, and `ps`.
2. Use Wikidata UN membership start year (`member of` United Nations with `start time`) when present.
3. Use Wikidata `inception` year for non-UN members and territories when present.
4. Fall back to `1900` only when no sourced date exists.

Use `recognition_year_source` and `recognition_year_confidence` before treating
this as an analytical variable. The CLI rejects `recognition_year` weighting for
rows whose source is only `fallback_placeholder`.

## Coverage

The current generated coverage is recorded in `data/metadata_coverage.json`.
At generation time:

- rows: 271
- columns: 52
- `bulk_enriched`: 226 rows
- `partial_enriched`: 24 rows
- `placeholder`: 21 rows

The placeholder rows are supranational, subnational, or special flag codes
already excluded from the default nation corpus, such as `eu`, `un`, `arab`,
`asean`, `gb-eng`, and the Spanish autonomous-community flags.

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

## Quality Flags

`metadata_quality` is a row-level coverage flag, not a guarantee that every
single column is sourced.

- `bulk_enriched`: population and area are sourced and the row matched a country
  or territory record.
- `partial_enriched`: at least one source matched, but one of the core fields is
  missing or placeholder.
- `placeholder`: no reliable bulk source matched the local flag code.

For field-level decisions, use the `*_source`, `*_year`, and
`recognition_year_confidence` columns.
