# Energy Load and Price Forecasting with Timestral

A language-agnostic plan. Goal: forecast electricity demand for the next hours, then test whether that forecast improves price prediction.

## 1. Goal and Scope

- **Primary goal:** forecast electricity consumption for the next 1 to 48 hours using a zero-shot time series foundation model (Timestral).
- **Secondary goal:** test whether load and renewable forecasts improve electricity price forecasts.
- **Non-goal (at first):** building a trading system. A better forecast is not the same as a trading edge, and day-ahead prices already include public forecasts.
- **Success criterion:** beat simple baselines on a proper out-of-sample backtest.

## 2. Model

- **Timestral** (Mistral): open-weights family of 5M, 28M, 123M and 209M parameters.
- Zero-shot forecasting, so no training is needed to start.
- Inputs: univariate or multivariate series, past-only and known-future covariates, NaN for missing values.
- Outputs: 21 quantiles (0.01 to 0.99), which gives uncertainty bands, not just point forecasts.
- Context up to 32,768 steps. Forecast horizon up to 1,024 steps per pass.
- Inference is in PyTorch (CPU is enough). Start with the 5M model, then compare against larger sizes.

## 3. Data Sources

| Data | Role | Possible source |
|---|---|---|
| Historical load | Target | Grid operator downloads, ENTSO-E Transparency Platform |
| Day-ahead prices | Second target | ENTSO-E or the local power exchange |
| Weather (temperature, wind, solar radiation) | Covariate | Open-Meteo or similar |
| Calendar features | Known-future covariate | Computed locally |
| Wind and solar generation | Covariate | ENTSO-E |

Check each source's current documentation for access rules, rate limits and licence terms.

## 4. Pipeline Overview

1. **Ingest:** download history, cache it on disk, never re-download what you already have.
2. **Clean:** convert to one time zone (UTC internally), resample to a fixed interval, mark true gaps as missing instead of filling them.
3. **Feature build:** create covariates (see section 5), aligned to the same time grid.
4. **Forecast:** feed context window plus covariates to the model, read out quantiles.
5. **Evaluate:** walk-forward backtest against baselines (section 6).
6. **Report:** error metrics, plots of forecast vs actual, and calibration of the quantile bands.

## 5. Covariates

**Known in advance (known-future covariates):**
- Hour of day, day of week, month
- Public holidays and school holidays
- Weather forecast (not observed weather, otherwise you leak the future into the backtest)

**Past-only covariates:**
- Observed weather
- Past wind and solar output

**Derived (for the price stage):**
- **Residual load** = demand minus wind minus solar. It is usually the strongest single driver of price.
- Fuel prices (gas, carbon) if available, since they set marginal cost.

## 6. Evaluation Design

- **Walk-forward backtest:** repeatedly forecast from a cutoff, then move the cutoff forward. Never train or tune on data after the cutoff.
- **Baselines to beat:**
  - Same hour yesterday
  - Same hour last week (hard to beat for load)
  - For prices: the same two, plus a simple seasonal model
- **Metrics:** MASE or MAPE for point accuracy, pinball loss or CRPS for quantiles, and coverage of the 80% and 90% bands.
- **Ablations:** compare history only, history plus calendar, history plus weather, and history plus residual load.
- **Beware leakage:** use weather forecasts as they were issued at the time, not the weather that actually happened.

## 7. Phases

1. **Load only, history only.** Smallest working version. Check that the pipeline and baselines work.
2. **Add calendar and weather covariates.** Measure the gain per covariate.
3. **Compare model sizes.** Is the 5M model good enough, or does 209M help meaningfully?
4. **Add prices.** First price history only, then price plus load forecast, then price plus residual load.
5. **Optional:** fine-tuning, probabilistic scheduling (for example shifting flexible consumption to cheap hours), a live dashboard.

## 8. Risks and Caveats

- **Market efficiency:** public forecasts are already priced into day-ahead markets, so expect small or no gains for price prediction.
- **Forecast vs trading:** an accurate forecast has more value for scheduling your own consumption than for speculation.
- **Distribution shift:** structural changes (new solar capacity, price spikes, policy changes) can break patterns learned from older data.
- **Data quality:** revisions, missing values, daylight-saving transitions and time zone mix-ups cause most bugs.
- **Backtest overfitting:** limit how many variants you try, and keep a final untouched test period.

## 9. Possible Extensions

- Quantile-based decisions: act only when the forecast interval is narrow enough.
- Multiple bidding zones modeled jointly.
- Compare Timestral against other foundation models (for example TimesFM or Chronos) on the same backtest.
- Port the inference code to another language once the Python version is validated, checking outputs layer by layer against the PyTorch reference.
