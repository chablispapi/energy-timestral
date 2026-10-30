use crate::{data, model::Request, protocol};
use anyhow::{ensure, Context, Result};
use chrono::{TimeZone, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Study {
    pub data: PathBuf,
    pub output: PathBuf,
    pub from: String,
    pub to: String,
    #[serde(default = "load")]
    pub target: String,
    #[serde(default = "history")]
    pub features: String,
    #[serde(default = "context")]
    pub context: usize,
    #[serde(default = "worker")]
    pub worker: String,
    #[serde(default)]
    pub observation_lag_hours: usize,
    #[serde(default)]
    pub load_data: Option<PathBuf>,
    #[serde(default)]
    pub wind_data: Option<PathBuf>,
    #[serde(default)]
    pub solar_data: Option<PathBuf>,
    #[serde(default)]
    pub school_holidays: Option<PathBuf>,
    #[serde(default = "development")]
    pub phase: String,
    #[serde(default)]
    pub frozen_selection: Option<PathBuf>,
    #[serde(default = "checkpoint")]
    pub checkpoint: String,
    #[serde(default)]
    pub synthetic: bool,
}
fn checkpoint() -> String {
    let pin: serde_json::Value = serde_json::from_str(include_str!("../reference/PIN.json"))
        .expect("embedded checkpoint pin");
    pin["fingerprint"]
        .as_str()
        .expect("checkpoint fingerprint")
        .into()
}
fn load() -> String {
    "load".into()
}
fn history() -> String {
    "history".into()
}
fn context() -> usize {
    672
}
fn worker() -> String {
    "127.0.0.1:7419".into()
}
fn development() -> String {
    "development".into()
}
pub fn read_config(path: &Path) -> Result<Study> {
    let c: Study = toml::from_str(&std::fs::read_to_string(path)?)?;
    ensure!(
        ["load", "price"].contains(&c.target.as_str()),
        "target must be load or price"
    );
    ensure!(
        [
            "history",
            "calendar",
            "weather",
            "calendar-weather",
            "load",
            "residual"
        ]
        .contains(&c.features.as_str()),
        "invalid feature ablation"
    );
    ensure!(c.context >= 32 && c.context <= 32768, "invalid context");
    ensure!(
        c.worker == "127.0.0.1:7419",
        "Keel worker endpoint is fixed to the SSH tunnel at 127.0.0.1:7419"
    );
    ensure!(
        ["development", "final"].contains(&c.phase.as_str()),
        "invalid phase"
    );
    Ok(c)
}
pub fn selection(c: &Study) -> String {
    format!(
        "{:x}",
        Sha256::digest(format!(
            "{}|{}|{}|{}|{}|{:?}|{:?}|{:?}|{:?}",
            c.checkpoint,
            c.target,
            c.features,
            c.context,
            c.observation_lag_hours,
            c.load_data,
            c.wind_data,
            c.solar_data,
            c.school_holidays
        ))
    )
}
pub type Table = BTreeMap<i64, Vec<Option<i64>>>;
type ErrorTable = BTreeMap<String, BTreeMap<(i64, i64), (f64, f64)>>;
pub fn table(path: &Path) -> Result<Table> {
    let mut t = BTreeMap::new();
    let mut last = None;
    for line in std::fs::read_to_string(path)?.lines() {
        let cells: Vec<_> = line.split('\t').collect();
        ensure!(cells.len() >= 2, "invalid canonical row");
        let ts: i64 = cells[0].parse()?;
        ensure!(
            ts % 3600 == 0 && last.is_none_or(|p| ts == p + 3600),
            "non-hourly or unsorted canonical grid"
        );
        last = Some(ts);
        let values = cells[1..]
            .iter()
            .map(|v| {
                if *v == "NA" {
                    Ok(None)
                } else {
                    Ok(Some(v.parse::<i64>()?))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        t.insert(ts, values);
    }
    ensure!(!t.is_empty(), "empty canonical data");
    Ok(t)
}
fn at(t: &Table, ts: i64) -> Option<f32> {
    t.get(&ts).and_then(|r| r[0]).map(|v| v as f32 / 1000.)
}
fn call(endpoint: &str, r: &Request) -> Result<Vec<Vec<f32>>> {
    let mut s = std::net::TcpStream::connect_timeout(
        &endpoint.parse()?,
        std::time::Duration::from_secs(10),
    )?;
    s.set_read_timeout(Some(std::time::Duration::from_secs(120)))?;
    s.set_write_timeout(Some(std::time::Duration::from_secs(120)))?;
    protocol::write_frame(&mut s, &protocol::request(r))?;
    let body = protocol::read_frame(&mut s)?;
    let mut lines = body.lines();
    let header: Vec<_> = lines.next().context("worker header")?.split('\t').collect();
    ensure!(
        header.len() >= 3 && header[2] == "OK",
        "worker error: {body}"
    );
    let mut out = vec![];
    for line in lines {
        let c: Vec<_> = line.split('\t').collect();
        if c[0] == "Q" && c[1] == "0" {
            out.push(
                c[3..]
                    .iter()
                    .map(|v| Ok(v.parse::<i64>()? as f32 / 1000.))
                    .collect::<Result<Vec<_>>>()?,
            );
        }
    }
    ensure!(out.len() == r.horizon, "worker horizon mismatch");
    Ok(out)
}
fn derived(c: &Study, path: &Path, origin: i64, until: i64) -> Result<BTreeMap<i64, i64>> {
    let t = table(path)?;
    let end = origin - c.observation_lag_hours as i64 * 3600;
    let h = ((until - end) / 3600) as usize;
    ensure!(
        h > 0 && h <= 1024,
        "derived forecast horizon exceeds supported range"
    );
    let r = Request {
        id: format!("derived-{origin}"),
        context: vec![(0..c.context)
            .map(|i| at(&t, end - (c.context - i) as i64 * 3600))
            .collect()],
        past: vec![],
        known: vec![],
        categorical_past: vec![],
        categorical_known: vec![],
        horizon: h,
        max_context: c.context,
        symmetric: true,
        clip_nonnegative: true,
        trace: false,
        expected_checkpoint: c.checkpoint.clone(),
    };
    let predictions = call(&c.worker, &r)?;
    let mut out = BTreeMap::new();
    for (ts, row) in &t {
        if *ts < end {
            if let Some(v) = row[0] {
                out.insert(*ts, v);
            }
        }
    }
    for (i, qs) in predictions.iter().enumerate() {
        out.insert(end + i as i64 * 3600, protocol::fixed(qs[10] as f64)?);
    }
    Ok(out)
}
// Initialization is not publication. Eight hours is conservative eligibility;
// archive retrieval times remain recorded, including for an older daily cycle.
fn weather_runs(origin: i64) -> [i64; 2] {
    let preferred = (origin - 8 * 3600).div_euclid(6 * 3600) * (6 * 3600);
    [preferred, preferred - 86400]
}
fn weather(origin: i64, until: i64, cache: &Path) -> Result<Table> {
    let candidates = weather_runs(origin);
    let mut failures = vec![];
    for run in candidates {
        match weather_run(run, cache).and_then(|t| {
            ensure!(
                (origin..until).step_by(3600).all(|ts| t.contains_key(&ts)),
                "archived run does not cover the complete forecast horizon"
            );
            Ok(t)
        }) {
            Ok(t) => {
                let selection = serde_json::json!({"origin":origin,"preferred_initialization":candidates[0],"selected_initialization":run,"minimum_initialization_lag_hours":8,"fallback_attempts":failures,"selected_at":Utc::now()});
                std::fs::write(
                    cache.join(format!("selection-{origin}.json")),
                    serde_json::to_vec_pretty(&selection)?,
                )?;
                return Ok(t);
            }
            Err(error) => failures
                .push(serde_json::json!({"initialization":run,"error":format!("{error:#}")})),
        }
    }
    anyhow::bail!(
        "no eligible archived weather run covers this origin: {}",
        serde_json::to_string(&failures)?
    )
}
fn weather_run(run: i64, cache: &Path) -> Result<Table> {
    let runstr = Utc
        .timestamp_opt(run, 0)
        .unwrap()
        .format("%Y-%m-%dT%H:%M")
        .to_string();
    let mut out: Table = BTreeMap::new();
    for (lat, lon, offset) in [(62.3908, 17.3069, 0), (63.1792, 14.6357, 3)] {
        let params = vec![
            ("latitude".into(), lat.to_string()),
            ("longitude".into(), lon.to_string()),
            ("models".into(), "ecmwf_ifs".into()),
            ("run".into(), runstr.clone()),
            (
                "hourly".into(),
                "temperature_2m,wind_speed_100m,shortwave_radiation".into(),
            ),
            ("wind_speed_unit".into(), "ms".into()),
            ("timeformat".into(), "unixtime".into()),
            ("forecast_days".into(), "4".into()),
        ];
        let mut parsed = None;
        for attempt in 0..3 {
            let bytes = data::cached(
                "https://single-runs-api.open-meteo.com/v1/forecast",
                &params,
                cache,
                attempt > 0,
            )?;
            // The archive can stream a timeout message with HTTP 200. Keep that
            // raw snapshot as evidence, but refresh its cache index on retry.
            if let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                parsed = Some(value);
                break;
            }
        }
        let v = parsed.context("weather archive returned invalid JSON after three attempts; raw responses retained in cache")?;
        let times = v["hourly"]["time"]
            .as_array()
            .context("weather archive returned no hourly times")?;
        for (i, t) in times.iter().enumerate() {
            let row = out
                .entry(t.as_i64().context("weather timestamp")?)
                .or_insert(vec![None; 6]);
            for (j, key) in ["temperature_2m", "wind_speed_100m", "shortwave_radiation"]
                .iter()
                .enumerate()
            {
                row[offset + j] = v["hourly"][key][i]
                    .as_f64()
                    .map(protocol::fixed)
                    .transpose()?;
            }
        }
    }
    Ok(out)
}
fn covariates(c: &Study, t: &Table, origin: i64, end: i64, h: usize, path: &Path) -> Result<usize> {
    let calendar = c.features == "calendar" || c.features == "calendar-weather";
    let weather_on = c.features == "weather" || c.features == "calendar-weather";
    let derived_on = c.features == "load" || c.features == "residual";
    let until = end + h as i64 * 3600;
    let wt = if weather_on {
        weather(origin, until, &c.output.join("weather-cache"))?
    } else {
        BTreeMap::new()
    };
    let lt = if derived_on {
        derived(
            c,
            c.load_data.as_deref().context("load_data required")?,
            origin,
            until,
        )?
    } else {
        BTreeMap::new()
    };
    let wind = if c.features == "residual" {
        derived(
            c,
            c.wind_data
                .as_deref()
                .context("wind_data required (combined wind output)")?,
            origin,
            until,
        )?
    } else {
        BTreeMap::new()
    };
    let solar = if c.features == "residual" {
        derived(
            c,
            c.solar_data.as_deref().context("solar_data required")?,
            origin,
            until,
        )?
    } else {
        BTreeMap::new()
    };
    let school = if let Some(p) = &c.school_holidays {
        std::fs::read_to_string(p)?
            .lines()
            .filter(|s| !s.starts_with('#'))
            .map(str::to_owned)
            .collect::<Vec<_>>()
    } else {
        vec![]
    };
    let mut out = String::new();
    let mut count = 0;
    for ts in (end - c.context as i64 * 3600..until).step_by(3600) {
        let mut row = vec![Some(0)];
        if calendar {
            let source = t.get(&ts).context(
                "canonical data must include calendar columns through the forecast horizon",
            )?;
            ensure!(source.len() >= 5, "missing canonical calendar columns");
            row.extend_from_slice(&source[1..5]);
            if c.school_holidays.is_some() {
                let d = Utc
                    .timestamp_opt(ts, 0)
                    .unwrap()
                    .with_timezone(&chrono_tz::Europe::Stockholm)
                    .format("%Y-%m-%d")
                    .to_string();
                row.push(Some(if school.contains(&d) { 1000 } else { 0 }));
            }
        }
        if weather_on {
            row.extend(wt.get(&ts).cloned().unwrap_or(vec![None; 6]));
        }
        if derived_on {
            let v = lt.get(&ts).copied();
            row.push(if c.features == "residual" {
                v.zip(wind.get(&ts))
                    .zip(solar.get(&ts))
                    .map(|((l, w), s)| l - w - s)
            } else {
                v
            });
        }
        count = row.len() - 1;
        out.push_str(&ts.to_string());
        for v in row {
            out.push('\t');
            out.push_str(&v.map(|n| n.to_string()).unwrap_or("NA".into()));
        }
        out.push('\n');
    }
    std::fs::write(path, out)?;
    Ok(count)
}
// Price date ranges refer to local delivery dates, not prior auction dates.
// Backtest load targets are clipped at the exclusive scoring boundary.
fn job_times(
    target: &str,
    day: i64,
    to: i64,
    lag: usize,
    live: bool,
) -> Result<(i64, i64, i64, usize)> {
    if target == "price" {
        let delivery_day = Utc.timestamp_opt(day, 0).unwrap().date_naive();
        let auction_day = delivery_day - chrono::Duration::days(1);
        let (cut, start, stop) = data::delivery(auction_day)?;
        Ok((cut, start, start, ((stop - start) / 3600) as usize))
    } else {
        let h = if live {
            48
        } else {
            48.min(((to - day) / 3600) as usize)
        };
        Ok((day, day - lag as i64 * 3600, day, h))
    }
}

// Calendar-only padding never extends observed coverage. Internal gaps are retained.
fn source_info(path: &Path, t: &Table) -> Result<serde_json::Value> {
    let sidecar = path.with_extension("json");
    let metadata: serde_json::Value = if sidecar.exists() {
        serde_json::from_slice(&std::fs::read(sidecar)?)?
    } else {
        serde_json::Value::Null
    };
    let hash = data::sha256(path)?;
    if let Some(expected) = metadata["canonical_sha256"].as_str() {
        ensure!(
            expected == hash,
            "canonical SHA256 differs from sidecar: {}",
            path.display()
        );
    }
    let first = t
        .iter()
        .find(|(_, r)| r[0].is_some())
        .context("no observed data")?
        .0;
    let last = t.iter().rev().find(|(_, r)| r[0].is_some()).unwrap().0;
    let from = metadata["observed_from"].as_i64().unwrap_or(*first);
    let to = metadata["observed_to_exclusive"]
        .as_i64()
        .unwrap_or(*last + 3600);
    ensure!(
        from < to && t.contains_key(&from) && t.contains_key(&(to - 3600)),
        "invalid observed coverage"
    );
    let gaps = (from..to)
        .step_by(3600)
        .filter(|ts| t.get(ts).is_none_or(|r| r[0].is_none()))
        .count();
    Ok(
        serde_json::json!({"path":path,"sha256":hash,"observed_from":from,"observed_to_exclusive":to,"internal_missing_hours":gaps,"metadata":metadata}),
    )
}
fn check_targets(info: &serde_json::Value, start: i64, h: usize) -> Result<()> {
    ensure!(start >= info["observed_from"].as_i64().unwrap() && start + h as i64 * 3600 <= info["observed_to_exclusive"].as_i64().unwrap(),
        "scheduled target interval exceeds observed source coverage (calendar padding is not observations)");
    Ok(())
}

pub fn run(c: &Study, forecast_only: bool) -> Result<()> {
    let from = data::date(&c.from)?;
    let to = data::date(&c.to)?;
    ensure!(from < to, "empty study dates");
    ensure!(
        from % 86400 == 0 && to % 86400 == 0,
        "study dates must be whole calendar dates"
    );
    let t = table(&c.data)?;
    let target_source = source_info(&c.data, &t)?;
    let mut sources = serde_json::json!({"target":target_source});
    for (role, path) in [
        ("load", &c.load_data),
        ("wind", &c.wind_data),
        ("solar", &c.solar_data),
    ] {
        if let Some(path) = path {
            sources[role] = source_info(path, &table(path)?)?;
        }
    }
    // Complete preflight for every scheduled job before weather, inference or holdout locks.
    for origin in (from..to).step_by(86400) {
        let (_, end, start, h) = job_times(
            &c.target,
            origin,
            to,
            c.observation_lag_hours,
            forecast_only,
        )?;
        ensure!(
            end - c.context as i64 * 3600 >= *t.first_key_value().unwrap().0,
            "insufficient context history"
        );
        if !forecast_only {
            check_targets(&sources["target"], start, h)?;
        }
        if forecast_only {
            break;
        }
    }
    if c.phase == "final" {
        ensure!(
            from >= data::date("2026-07-01")? && to <= data::date("2026-10-01")?,
            "final dates must be July–September 2026"
        );
        let frozen = std::fs::read_to_string(
            c.frozen_selection
                .as_deref()
                .context("final runs require frozen_selection")?,
        )?;
        ensure!(
            frozen.trim() == selection(c),
            "selection differs from frozen development configuration"
        );
    } else {
        ensure!(
            to <= data::date("2026-07-01")? || forecast_only,
            "development backtests cannot inspect the final test period"
        );
    }
    std::fs::create_dir_all(&c.output)?;
    let lock = c.output.join("run.json");
    ensure!(
        !lock.exists(),
        "output already contains a run; choose a new directory to preserve results"
    );
    let mut jobtext = String::new();
    let mut origin = from;
    let mut n = 0;
    while origin < to {
        let (cutoff, end, start, h) = job_times(
            &c.target,
            origin,
            to,
            c.observation_lag_hours,
            forecast_only,
        )?;
        ensure!(
            end - c.context as i64 * 3600 >= *t.first_key_value().unwrap().0,
            "insufficient context history"
        );
        let gap = ((start - end) / 3600) as usize;
        let covpath = c.output.join(format!("covariates-{cutoff}.tsv"));
        let features = if c.features == "history" {
            0
        } else {
            covariates(c, &t, cutoff, end, h + gap, &covpath)?
        };
        jobtext.push_str(&format!(
            "{cutoff}\t{end}\t{start}\t{h}\t{}\t{features}\n",
            covpath.display()
        ));
        n += 1;
        origin += 86400;
        if forecast_only {
            break;
        }
    }
    if c.features == "weather" || c.features == "calendar-weather" {
        let selections = (from..if forecast_only { from + 86400 } else { to })
            .step_by(86400)
            .map(|origin| {
                let (cutoff, _, _, _) = job_times(
                    &c.target,
                    origin,
                    to,
                    c.observation_lag_hours,
                    forecast_only,
                )?;
                Ok(serde_json::from_slice::<serde_json::Value>(
                    &std::fs::read(
                        c.output
                            .join(format!("weather-cache/selection-{cutoff}.json")),
                    )?,
                )?)
            })
            .collect::<Result<Vec<_>>>()?;
        sources["weather"] = serde_json::json!(selections);
    }
    let jobs = c.output.join("jobs.tsv");
    std::fs::write(&jobs, jobtext)?;
    let out = c.output.join("results.tsv");
    let input = format!(
        "{}\n{}\n{}\n{}\n{}\n{}\n",
        c.data.display(),
        out.display(),
        c.context,
        if c.target == "price" { 0 } else { 1 },
        jobs.display(),
        c.checkpoint
    );
    // Build before starting the run; a failed build must not reserve the final holdout.
    std::fs::create_dir_all("build")?;
    ensure!(
        Command::new("keelc")
            .args(["build", "-o", "build/energy"])
            .status()?
            .success(),
        "Keel build failed"
    );
    let provenance = serde_json::json!({"config":c,"source_provenance":sources,"selection":selection(c),"jobs":n,"data_sha256":format!("{:x}",Sha256::digest(std::fs::read(&c.data)?)),"data_status":"retrospective revised-data backtest; publication-time electricity vintages not proven","weather_policy":"single archived run, initialization at least eight hours before cutoff; unavailable preferred run falls back one daily cycle with full horizon coverage; earlier context weather missing","started_at":Utc::now(),"status":"running"});
    if c.phase == "final" {
        std::fs::create_dir_all("data/final-runs")?;
        let guard = Path::new("data/final-runs").join(format!("{}.lock", selection(c)));
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(guard)
            .context("this frozen selection has already used the final holdout")?;
    }
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock)?;
    f.write_all(&serde_json::to_vec_pretty(&provenance)?)?;
    let mut child = Command::new("build/energy")
        .env("KEEL_STATS", "1")
        .stdin(Stdio::piped())
        .spawn()?;
    child.stdin.take().unwrap().write_all(input.as_bytes())?;
    let success = child.wait()?.success();
    if !success {
        let mut failed = provenance.clone();
        failed["status"] = "failed".into();
        std::fs::write(&lock, serde_json::to_vec_pretty(&failed)?)?;
        anyhow::bail!(
            "Keel run failed; partial results and worker error headers retained in results.tsv"
        );
    }
    report(&c.output)?;
    let mut complete = provenance;
    complete["status"] = "complete".into();
    complete["completed_at"] = serde_json::to_value(Utc::now())?;
    std::fs::write(&lock, serde_json::to_vec_pretty(&complete)?)?;
    Ok(())
}

#[derive(Default, Clone)]
struct Totals {
    n: f64,
    missing: f64,
    ae: f64,
    pb: f64,
    c80: f64,
    c90: f64,
    w80: f64,
    w90: f64,
    mase: f64,
    mase_n: f64,
}
pub fn report(dir: &Path) -> Result<()> {
    let run: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("run.json"))?)?;
    let synthetic = std::fs::read(dir.join("run.json"))
        .ok()
        .and_then(|s| serde_json::from_slice::<serde_json::Value>(&s).ok())
        .is_some_and(|v| v["config"]["synthetic"] == true);
    let label = if synthetic {
        "SYNTHETIC TEST DATA — verifies the software, not SE2 forecasting skill"
    } else {
        "Retrospective revised-data backtest; historical electricity publication vintages are not proven"
    };
    let text = std::fs::read_to_string(dir.join("results.tsv"))?;
    let mut totals: BTreeMap<String, Totals> = BTreeMap::new();
    let mut unavailable: BTreeMap<String, u64> = BTreeMap::new();
    let mut missing_targets: BTreeMap<String, u64> = BTreeMap::new();
    let mut points = vec![];
    let mut allpoints = vec![];
    for line in text.lines() {
        let c: Vec<_> = line.split('\t').collect();
        if c[0] == "M" {
            ensure!(c.len() == 13, "invalid metric record");
            let nums = c[2..]
                .iter()
                .map(|v| v.parse::<f64>())
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let t = totals.entry(c[1].into()).or_default();
            t.n += nums[1];
            t.missing += nums[2];
            t.ae += nums[3];
            t.pb += nums[4];
            t.c80 += nums[5];
            t.c90 += nums[6];
            t.w80 += nums[7];
            t.w90 += nums[8];
            if nums[9] > 0. && nums[10] > 0. {
                t.mase += nums[3] / (nums[9] / nums[10]);
                t.mase_n += nums[1];
            }
        } else if c[0] == "P" {
            ensure!(c.len() == 26, "invalid forecast record");
            if c[4] == "NA" {
                *missing_targets.entry(c[3].into()).or_default() += 1;
            }
            let p = c.iter().map(|s| s.to_string()).collect::<Vec<_>>();
            if c[3] == "timestral" {
                points.push(p.clone());
            }
            allpoints.push(p);
        } else if c[0] == "U" {
            ensure!(c.len() == 4, "invalid unavailable-prediction record");
            *unavailable.entry(c[1].into()).or_default() += 1;
        }
    }
    let mut metrics = BTreeMap::new();
    let mut rows = String::new();
    for (name, t) in &totals {
        let n = t.n.max(1.);
        let m = serde_json::json!({"n":t.n,"missing":t.missing,"missing_targets":missing_targets.get(name).copied().unwrap_or(0),"unavailable_predictions":unavailable.get(name).copied().unwrap_or(0),"mae":if t.n>0.{Some(t.ae/n/1000.)}else{None},"mase":if t.mase_n>0.{Some(t.mase/t.mase_n)}else{None},"pinball":if t.n>0.{Some(t.pb/n/21./1e7)}else{None},"coverage80":t.c80/n,"coverage90":t.c90/n,"width80":t.w80/n/1000.,"width90":t.w90/n/1000.});
        rows.push_str(&format!("<tr><th>{name}</th><td>{}</td><td>{:.4}</td><td>{:.4}</td><td>{:.1}%</td><td>{:.1}%</td></tr>",t.n,t.ae/n/1000.,t.pb/n/21./1e7,100.*t.c80/n,100.*t.c90/n));
        metrics.insert(name.clone(), m);
    }
    let mut errors: ErrorTable = BTreeMap::new();
    let mut reliability = [0.; 21];
    let mut reliability_n = 0f64;
    let mut firsts: BTreeMap<i64, i64> = BTreeMap::new();
    for p in &allpoints {
        let o = p[1].parse::<i64>()?;
        let t = p[2].parse::<i64>()?;
        firsts
            .entry(o)
            .and_modify(|v| *v = (*v).min(t))
            .or_insert(t);
    }
    let mut horizons: BTreeMap<i64, (f64, f64)> = BTreeMap::new();
    for p in &allpoints {
        if let Ok(y) = p[4].parse::<f64>() {
            let o = p[1].parse::<i64>()?;
            let t = p[2].parse::<i64>()?;
            let qs = p[5..]
                .iter()
                .map(|v| v.parse::<f64>())
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let ae = (y - qs[10]).abs() / 1000.;
            let mut pb = 0.;
            for (q, pred) in qs.iter().enumerate() {
                let level = [
                    0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                    0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
                ][q];
                let e = y - pred;
                pb += if e < 0. {
                    (1. - level) * (-e)
                } else {
                    level * e
                };
                if p[3] == "timestral" && y <= *pred {
                    reliability[q] += 1.;
                }
            }
            errors
                .entry(p[3].clone())
                .or_default()
                .insert((o, t), (ae, pb / 21. / 1000.));
            if p[3] == "timestral" {
                reliability_n += 1.;
                let h = (t - firsts[&o]) / 3600 + 1;
                let v = horizons.entry(h).or_default();
                v.0 += ae;
                v.1 += 1.;
            }
        }
    }
    let mut comparisons = BTreeMap::new();
    if let Some(model) = errors.get("timestral") {
        for (name, base) in &errors {
            if name == "timestral" {
                continue;
            }
            let mut mae = vec![];
            let mut pinball = vec![];
            for ((o, t), (a, p)) in model {
                if let Some((b, q)) = base.get(&(*o, *t)) {
                    mae.push((*o, a - b));
                    pinball.push((*o, p - q));
                }
            }
            if !mae.is_empty() {
                comparisons.insert(
                    name.clone(),
                    serde_json::json!({"mae":bootstrap(&mae),"pinball":bootstrap(&pinball)}),
                );
            }
        }
    }
    let horizon_metrics = horizons
        .iter()
        .map(|(h, (sum, n))| serde_json::json!({"horizon":h,"mae":sum/n,"n":n}))
        .collect::<Vec<_>>();
    let calibration = reliability
        .iter()
        .map(|n| {
            if reliability_n > 0. {
                Some(n / reliability_n)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let payload = serde_json::json!({"label":label,"source_provenance":run["source_provenance"],"metrics":metrics,"paired_week_block_difference_model_minus_baseline":comparisons,"horizon_metrics":horizon_metrics,"calibration":calibration});
    std::fs::write(
        dir.join("metrics.json"),
        serde_json::to_vec_pretty(&payload)?,
    )?;
    let mut csv=String::from("origin,timestamp,actual,q01,q05,q10,q15,q20,q25,q30,q35,q40,q45,q50,q55,q60,q65,q70,q75,q80,q85,q90,q95,q99\n");
    for p in &points {
        csv.push_str(&format!(
            "{},{},{},{}\n",
            p[1],
            p[2],
            p[4],
            p[5..].join(",")
        ));
    }
    std::fs::write(dir.join("forecasts.csv"), csv)?;
    let plot = plot(&points);
    let horizon_svg = chart(
        &horizons
            .iter()
            .map(|(h, (sum, n))| (*h as f64, sum / n))
            .collect::<Vec<_>>(),
        "MAE by forecast horizon",
        false,
    );
    let calibration_svg = chart(
        &reliability
            .iter()
            .enumerate()
            .map(|(i, n)| {
                (
                    [
                        0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
                        0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
                    ][i],
                    *n / reliability_n.max(1.),
                )
            })
            .collect::<Vec<_>>(),
        "Observed cumulative quantile coverage",
        true,
    );
    let definition = run["source_provenance"]["target"]["metadata"]["source_provenance"]
        ["manifest"]["load_definition"]
        .as_str()
        .unwrap_or("")
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");
    let source_text = serde_json::to_string_pretty(&run["source_provenance"])?
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");
    let html=format!("<!doctype html><meta charset='utf-8'><title>SE2 forecasting</title><style>body{{font:16px system-ui;max-width:1100px;margin:3rem auto;padding:1rem;color:#172738}}table{{border-collapse:collapse}}th,td{{padding:.6rem;border-bottom:1px solid #ccd}}svg{{width:100%;height:auto}}.note{{background:#fff4ce;padding:1rem}}</style><h1>SE2 forecasting</h1><p class='note'>{label}. Forecast accuracy is not a trading result.</p><table><tr><th>Model</th><th>Scored hours</th><th>MAE</th><th>Pinball</th><th>80% coverage</th><th>90% coverage</th></tr>{rows}</table><h2>Sources</h2><p>{definition}</p><details><summary>Definitions, hashes, observed coverage and publication vintage</summary><pre>{source_text}</pre></details><h2>First forecast: median, 80% band and actual</h2>{plot}<h2>Error by horizon</h2>{horizon_svg}<h2>Quantile calibration</h2>{calibration_svg}<p>Values in the CSV are fixed-point integers (divide by 1,000). Metrics use original MW or EUR/MWh units. MASE uses past-only weekly differences; zero denominators are excluded. Detailed metrics and paired weekly-block comparisons are in metrics.json. Intervals for baselines use trailing past residuals.</p>");
    std::fs::write(dir.join("report.html"), html)?;
    Ok(())
}
fn bootstrap(pairs: &[(i64, f64)]) -> serde_json::Value {
    let mut weeks: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
    for (t, v) in pairs {
        weeks.entry(t.div_euclid(604800)).or_default().push(*v);
    }
    let blocks: Vec<_> = weeks.values().collect();
    let mut state = 42u64;
    let mut boot = vec![];
    for _ in 0..1000 {
        let mut sum = 0.;
        let mut n = 0;
        for _ in 0..blocks.len() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let b = blocks[(state >> 32) as usize % blocks.len()];
            sum += b.iter().sum::<f64>();
            n += b.len();
        }
        boot.push(sum / n as f64);
    }
    boot.sort_by(f64::total_cmp);
    serde_json::json!({"mean":pairs.iter().map(|p|p.1).sum::<f64>()/pairs.len()as f64,"ci95":[boot[25],boot[974]],"weeks":blocks.len(),"paired_predictions":pairs.len(),"paired_origins":pairs.iter().map(|p|p.0).collect::<std::collections::BTreeSet<_>>().len(),"interpretation":if blocks.len()<8{"too few weeks for a stable uncertainty estimate"}else{"negative favors Timestral"}})
}
fn plot(points: &[Vec<String>]) -> String {
    let Some(first) = points.first() else {
        return "<p>No forecasts.</p>".into();
    };
    let p: Vec<_> = points.iter().take_while(|x| x[1] == first[1]).collect();
    let mut values = vec![];
    for r in &p {
        for i in [4, 7, 15, 23] {
            if let Ok(n) = r[i].parse::<f64>() {
                values.push(n);
            }
        }
    }
    if values.is_empty() {
        return String::new();
    }
    let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let xy = |i: usize, v: f64| {
        format!(
            "{:.2},{:.2}",
            30. + i as f64 * 940. / p.len().max(2) as f64,
            270. - (v - lo) / (hi - lo).max(1.) * 240.
        )
    };
    let line = |col: usize| {
        p.iter()
            .enumerate()
            .filter_map(|(i, r)| r[col].parse::<f64>().ok().map(|v| xy(i, v)))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let lower = line(7);
    let upper = p
        .iter()
        .enumerate()
        .rev()
        .map(|(i, r)| xy(i, r[23].parse().unwrap()))
        .collect::<Vec<_>>()
        .join(" ");
    format!("<svg viewBox='0 0 1000 300' role='img' aria-label='Forecast median and uncertainty with actual load or price'><polygon fill='#d8e7ff' points='{lower} {upper}'/><polyline fill='none' stroke='#2563eb' stroke-width='2' points='{}'/><polyline fill='none' stroke='#111827' stroke-width='2' points='{}'/></svg>",line(15),line(4))
}

fn chart(values: &[(f64, f64)], label: &str, ideal: bool) -> String {
    let ymax = if ideal {
        1.
    } else {
        values.iter().map(|p| p.1).fold(0f64, f64::max).max(1e-9)
    };
    let xmax = values.last().map(|p| p.0).unwrap_or(1.).max(1.);
    let points = values
        .iter()
        .map(|(x, y)| format!("{:.2},{:.2}", 40. + x / xmax * 920., 250. - y / ymax * 210.))
        .collect::<Vec<_>>()
        .join(" ");
    let diagonal = if ideal {
        "<path d='M40 250L960 40' stroke='#94a3b8' stroke-dasharray='5 5' fill='none'/>"
    } else {
        ""
    };
    format!("<svg viewBox='0 0 1000 290' role='img' aria-label='{label}'><path d='M40 30V250H970' fill='none' stroke='#ccd'/><text x='2' y='35' font-size='12'>{ymax:.3}</text><text x='40' y='275' font-size='12'>0</text><text x='940' y='275' font-size='12'>{xmax:.2}</text>{diagonal}<polyline points='{points}' fill='none' stroke='#2563eb' stroke-width='2'/></svg>")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn weather_fallback_remains_eligible_and_bounds_age() {
        for origin in [0, 8 * 3600, 9 * 3600] {
            let [preferred, fallback] = weather_runs(origin);
            assert!(preferred <= origin - 8 * 3600);
            assert_eq!(preferred - fallback, 86400);
            // A four-day run still covers the ordinary 48-hour load horizon.
            assert!(fallback + 96 * 3600 >= origin + 48 * 3600);
        }
    }
    #[test]
    fn coverage_excludes_padding_but_allows_internal_gaps() {
        let info = serde_json::json!({"observed_from":0,"observed_to_exclusive":10800,"internal_missing_hours":1});
        assert!(check_targets(&info, 0, 3).is_ok());
        assert!(check_targets(&info, 0, 4).is_err());
        assert!(check_targets(&info, -3600, 1).is_err());
    }
    #[test]
    fn legacy_coverage_ends_at_last_observation() {
        let mut t = Table::new();
        for (ts, v) in [(0, Some(0)), (3600, None), (7200, Some(2)), (10800, None)] {
            t.insert(ts, vec![v]);
        }
        let path = std::env::temp_dir().join(format!("svk-coverage-{}.tsv", std::process::id()));
        std::fs::write(&path, "coverage test").unwrap();
        let info = source_info(&path, &t).unwrap();
        assert_eq!(info["observed_to_exclusive"], 10800);
        assert_eq!(info["internal_missing_hours"], 1);
        assert!(check_targets(&info, 10800, 1).is_err());
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn development_load_targets_do_not_enter_july() {
        let day = data::date("2026-06-30").unwrap();
        let stop = data::date("2026-07-01").unwrap();
        let (_, _, start, h) = job_times("load", day, stop, 0, false).unwrap();
        assert_eq!(h, 24);
        assert_eq!(start + h as i64 * 3600, stop);
    }
    #[test]
    fn price_boundary_uses_delivery_day() {
        let day = data::date("2026-06-30").unwrap();
        let (cut, _, start, h) =
            job_times("price", day, data::date("2026-07-01").unwrap(), 0, false).unwrap();
        assert!(cut < day);
        assert_eq!(
            Utc.timestamp_opt(start, 0)
                .unwrap()
                .with_timezone(&chrono_tz::Europe::Stockholm)
                .date_naive()
                .to_string(),
            "2026-06-30"
        );
        assert_eq!(h, 24);
    }
    #[test]
    fn price_dst_day_has_all_delivery_intervals() {
        let day = data::date("2026-10-25").unwrap();
        let (_, _, _, h) = job_times("price", day, day + 86400, 0, false).unwrap();
        assert_eq!(h, 25);
    }
}
