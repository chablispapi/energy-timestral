# Validation status

Implemented and deployed. Real SE2 evaluation is pending an ENTSO-E token or imported electricity data. No real-market forecast advantage or GPU-over-CPU speedup has been established.

## Proven checks

- 12 Rust unit tests pass, including UTC/DST handling, interval aggregation, variable-block XML, framing, categorical encoding and holdout boundaries.
- Keel deterministic fixtures match evaluator and native output exactly; native memory statistics report `live=0`.
- `cargo fmt --check` and `cargo clippy --locked --all-targets -- -D warnings` pass.
- Seven original-PyTorch versus Intel-GPU cases pass: seasonal load, missing/padded context, constants, negative values, multivariate targets, covariates/categories, and all-missing context.
- Final quantile tolerance: `rtol=1e-3`, `atol=1e-3`. Layer tolerances use RMS-normalized tensors, `rtol=1e-3`, `atol=1e-4`. Maximum absolute final differences range from zero to 0.000836 across the fixtures.
- End-to-end synthetic load, archived-weather conditioning, residual-load conditioning and a 25-hour DST price day complete successfully; Keel reports `live=0`.
- Integration tests verify fragmented transport, wrong-checkpoint rejection, malformed covariates, future-electricity leakage prevention, independently recomputed MAE/pinball units, and retained failure diagnostics.

## Running deployment

- Host: `chablis@192.168.0.167`.
- Directory: `~/Projects/energy-timestral`; original `~/Projects/timestral` is unchanged.
- Service: transient `systemd --user` unit `energy-timestral.service`, bound to `127.0.0.1:7419`.
- Adapter: Intel UHD Graphics 600, Vulkan, PCI `8086:3185`; software fallback is forbidden.
- Checkpoint: Timestral-5M at Hub revision `bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f`.
- Config + weights fingerprint: `5c49704c490ce2a42d78dd87e4512754c24bd7972add8847fe30beabd4b29995`.
- Five measured warm requests: median worker 264 ms; median through SSH 328.2 ms (672-step context, 48-step horizon, symmetric averaging).

## Artifacts

- `output/weather-smoke/report.html`: synthetic target with actual archived weather covariates.
- `output/price-dst-smoke/report.html`: synthetic 25-hour price delivery day.
- `output/derived-smoke/report.html`: synthetic load/wind/solar forecast-derived covariates.
- `output/validation/*-reference.json` and `*-intel.json`: raw numerical parity evidence.
- `output/validation/benchmark.json`: per-request timings and hardware identity.
- `output/validation/integration-53ef75f2`: latest leakage/failure integration artifacts.

## Remaining empirical work

Provide `ENTSOE_TOKEN` or canonical CSV data, then run the development ablations and `scripts/compare_models.py`. The 28M/123M/209M comparison runner and pinned checkpoint downloader are implemented; those sizes have not been validated on this GPU. Select a development result before freezing and using July–September 2026. Historical revised-data limitations and conservative weather-run eligibility remain explicit in reports.
