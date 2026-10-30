# Svk SE2 development results

Daily UTC origins: March 8–June 30, 2026 (115 origins), 672 hours of context.
Each variant scores 5,496 forecast/actual pairs, including overlapping 48-hour
forecasts; the final origin is clipped at July 1. No July–September scoring was
performed. Load is registered SE2 withdrawal including network losses and
storage charging. These are retrospective settlement observations, with no
proof of publication-time availability. January and February provide context
and trailing baseline calibration. Weather variants use the same archived
values and initialization choices. When the preferred archive run is unavailable,
the preceding daily cycle is used only if it covers the complete horizon; all
choices are recorded under source_provenance.weather in run metadata.

MAE, pinball and widths are in MW; MASE is dimensionless. Baseline intervals use
past-only residuals. The same baselines apply to all four variants.

| Forecast | MAE | MASE | Pinball | 80% coverage | 90% coverage | 80% width | 90% width |
|---|---:|---:|---:|---:|---:|---:|---:|
| history | 70.023 | 0.655 | 24.126 | 80.68% | 90.48% | 233.966 | 307.962 |
| calendar | 68.782 | 0.646 | 23.559 | 80.93% | 90.61% | 226.145 | 301.638 |
| weather | 73.035 | 0.685 | 25.270 | 85.13% | 93.16% | 272.810 | 364.079 |
| calendar-weather | 70.610 | 0.663 | 24.277 | 83.77% | 92.72% | 250.173 | 336.343 |
| yesterday | 101.405 | 0.947 | 34.841 | 80.19% | 88.03% | 334.354 | 412.952 |
| last_week | 99.542 | 0.930 | 34.798 | 82.24% | 89.85% | 367.387 | 460.929 |

All variants have zero missing target observations. Detailed horizon errors,
quantile calibration, source provenance and paired weekly-block bootstrap
comparisons are retained in each run’s metrics.json and report.html.

Acceptance: 5,832 aligned source observations plus 96 missing calendar rows per
series, unchanged source workbook and CSVs, pinned Intel GPU checkpoint, sorted
21-quantile forecasts, Keel live=0, all development targets before July 1.
See output/validation/svk-future.json and integration.json for real-data cutoff
isolation and ancillary-load/generation leakage checks; test_svk_study.py
verifies that the full holdout fails coverage preflight.
