# Data licensing and attribution

The project-level `LICENSE` applies to original project source code and documentation. It does **not** relicense third-party or derivative data in `data/`.

## `data/metadata.csv`

`data/metadata.csv` is a derived/combined database assembled from several public sources. Because it incorporates material from `mledoze/countries`, whose database is licensed under the **Open Database License (ODbL) 1.0**, this metadata database is made available under the ODbL 1.0 to the extent that the ODbL applies to the database or derivative database.

ODbL 1.0: https://opendatacommons.org/licenses/odbl/1-0/

Contains information from `mledoze/countries`:
https://github.com/mledoze/countries

Source-specific terms and attribution obligations continue to apply to incorporated fields. The current build also uses:

- **World Bank World Development Indicators / country metadata** — normally CC BY 4.0 with the World Bank dataset terms and any indicator-specific exceptions: https://datacatalog.worldbank.org/public-licenses
- **Our World in Data CO2 and greenhouse-gas data** — OWID-produced material is CC BY; third-party series retain the terms of their original providers: https://github.com/owid/co2-data
- **Wikidata** — structured data is provided under CC0: https://www.wikidata.org/wiki/Wikidata:Licensing
- **UN DESA World Population Prospects 2024** — relevant UN DESA publication/documentation is identified as CC BY 3.0 IGO: https://www.un.org/development/desa/pd/world-population-prospects-2024

`data/metadata_sources.md` records which sources populate which fields, rebuild inputs, units, and quality caveats. `data/metadata_coverage.json` is generated from the combined metadata table.

Users redistributing the database or derived databases are responsible for complying with the applicable ODbL and source-specific attribution/terms. This notice is intended to preserve those terms rather than replace them.

## Flag assets

The SVG flags in `data/flags_svg/` are separately covered by the upstream `lipis/flag-icons` MIT license. See `LICENSE-flags.txt`.
