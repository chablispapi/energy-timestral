# Integrating the prepared Svk workbook

## Source preparation completed

`scripts/prepare_svk_workbook.py` produces `data/prepared/svk_2026/` without
altering the original workbook. Install its single decoding dependency with
`uv pip install --python .venv/bin/python -r requirements-data.txt`.

The output contains import-ready `load.csv`, `wind.csv`, `solar.csv`, a supporting
`components.csv` and `manifest.json`, plus frozen independent reconciliation
evidence. Canonical imports and development study configurations are now available.

- All 5,832 source hours are retained. May's 744 text timestamps are parsed along
  with the 5,088 Excel dates. Duplicate/unordered/missing source hours and unknown
  header layouts fail preparation instead of being silently repaired.
- The source clock is fixed Swedish standard time, UTC+1 throughout the year.
  This is established by independently matching wind, solar and hydro against
  eSett's explicit UTC timestamps in winter and summer. Do not use a DST-aware
  Stockholm timezone to interpret the original timestamps.
- Exact UTC coverage is **2025-12-31 23:00 inclusive to 2026-08-31 23:00 exclusive**.
  Each MWh value represents one hour and is exported as average MW. Source
  precision is retained; rounding to 0.001 MW belongs to the existing Rust importer.
- Positive registered SE2 load includes metered ordinary consumption, separately
  reported industry over 50 MW, interruptible consumption, storage charging,
  metered losses, profiled consumption and profiled losses:
  `load = -(C + G + K + O + S + AZ + BD)`.
  The metered subtotal and profiled subtotal independently reconcile against
  eSett. This is registered withdrawal including network losses, not a correction
  for hidden behind-the-meter self-generation or a claim of exact equivalence
  with a different provider's load definition.
- Wind is `AE + AI`; solar is `AR`. Offshore wind's observed zeros stay zero.
  Missing components remain `NA` and make their aggregate missing. Storage
  generation is retained separately and is never subtracted from consumption.
- Source/output hashes, source column mappings, units, signs, clock rules and
  individual reconciliation differences are in the manifest. Production checks
  use a 0.000001 MW tolerance. Consumption snapshots differ by at most 0.003468 MW
  in the sampled July hours; the explicitly recorded 0.01 MW cross-snapshot
  tolerance does not modify any workbook value.

Verified: seven focused Python tests, 98 independent comparisons, unchanged
source SHA256, and temporary imports of all three CSVs through the existing Rust
CLI. Each temporary canonical file contained 5,832 observations and 96 missing
future-calendar rows; conversion error was at most 0.0005 MW. These preparation checks preceded the development GPU studies described below.
The July–September holdout remains unscored.

Independent sources:

- [eSett Open Data API specification](https://api.opendata.esett.com/openapi?format=json),
  `EXP15/Aggregate` and `EXP16/Aggregate`; exact queries and response hashes are
  retained in `tests/fixtures/svk/sources.json` and the prepared evidence directory.
- [Svk statistics methodology](https://www.svk.se/om-kraftsystemet/kraftsystemdata/elstatistik/beskrivning-av-statistiken/).
  Settlement statistics do not include all hidden production or consumption.
- [Svk publication schedule](https://www.svk.se/om-kraftsystemet/kraftsystemdata/elstatistik/).
  The monthly historical export is not a live operational observation feed.

## Implemented integration

`ingest --source-manifest PATH` requires `--csv`. It checks the selected CSV's
SHA256 and kind, SE2 MW units, hourly intervals, complete timestamp grid and
exact declared bounds before writing canonical output. It uses the existing
UTC parser, hourly normalization and fixed-point conversion. The canonical
sidecar retains the full preparation manifest, its hash and the selected CSV
hash under `source_provenance`, plus a canonical TSV hash, observed bounds and
the separate future-calendar row count. Existing imports without a manifest
remain supported; their coverage is bounded by actual nonmissing observations.

The imported files are `data/svk_2026/{load,wind,solar}.tsv`. Each has 5,832
observations and 96 missing future rows on an identical UTC grid. Neither the
original workbook nor any prepared source value was changed. Missing components
stay missing; observed zeros stay zero.

The study runner verifies canonical hashes, records source definitions,
retrospective publication-vintage limitations and internal missing counts in
`run.json`, `metrics.json` and HTML reports. Ancillary load/wind/solar inputs also
carry their provenance into runs. Before any weather preparation, GPU request
or holdout lock, every scheduled scoring job must fit the observed bounds.
For legacy files without coverage metadata, first/last nonmissing observations
set the bounds. Price checks use local-delivery-job UTC intervals. Genuine
internal gaps are retained and counted. Forecast-only runs can extend beyond
actual observations.

## Development execution

`config/svk-load-smoke.toml` uses the March 8 origin in its own output directory.
The four `config/svk-load-{history,calendar,weather,calendar-weather}.toml` files
use daily UTC origins from March 8 through June 30 (115 origins), 672 hours of
context and separate output directories. January and February remain available
for context and past-only baseline calibration. Scoring clips at July 1.
The checkpoint defaults to the embedded `reference/PIN.json` Timestral-5M pin;
the worker endpoint remains the existing SSH tunnel at `127.0.0.1:7419`.
No GPU or Keel compiler changes were made.

```sh
python3 scripts/run_svk_development.py
python3 scripts/verify_svk_development.py
python3 scripts/verify_svk_future.py
.venv/bin/python -m unittest discover -s tests -p 'test_svk_*.py'
scripts/check.sh
python3 scripts/integration.py
```

The runner preserves results in the configured output directories and diagnostics
in `output/validation/svk-*-development.log`. Existing run directories cannot be
reused; choose new output paths for a repeat study. The acceptance audit writes
`output/validation/svk-acceptance.json` and the comparison in
[svk-development-results.md](svk-development-results.md). Detailed run reports
contain MAE, MASE, pinball loss, 80%/90% coverage and widths, horizon errors,
quantile calibration and paired weekly-block baseline comparisons.

Validation covers preparation/reconciliation, tampered hashes, wrong series
kinds, incompatible units and intervals, UTC-offset conversion, missing values
versus observed zeros, aligned canonical grids and rounding tolerance. Real
worker checks verify the pinned Intel checkpoint, sorted quantiles and Keel
`live=0`. `scripts/verify_svk_future.py` proves that removing post-cutoff real
observations leaves inputs and forecasts unchanged, and that a September
forecast can run without September actuals. The existing integration suite
checks future ancillary load/generation isolation and fixed-point metric units.
`tests/test_svk_study.py` confirms the full July–September scoring request fails
coverage before weather, inference, frozen-selection access or output creation.

Weather archive resilience: malformed HTTP-200 response bodies are retained and
retried with a fresh snapshot (at most three attempts). If the preferred run is
unavailable, both weather variants may use the preceding daily cycle, provided
one complete run covers the entire target horizon and is initialized at least
eight hours before cutoff. The runner records preferred/selected initializations
and failures under `source_provenance.weather`; it never stitches model runs.

## Reserved evaluation and price work

Do not run final scoring until September data and the missing last UTC hour of
August have been obtained and overlapping hours reconciled. Keep the full
July–September holdout reserved; do not shorten it. Freeze the development
selection only after comparing the completed development variants.

Price experiments wait for aligned SE2 prices. Then reuse the imported files as
`load_data`, `wind_data` and `solar_data` through the existing pre-auction
forecast-derived residual-load path. Future actual electricity observations
must never become known-future covariates. No ENTSO-E token is needed for this
workbook import.
