use anyhow::{ensure, Context, Result};
use chrono::{DateTime, Datelike, Duration, NaiveDate, TimeZone, Timelike, Utc};
use chrono_tz::Europe::Stockholm;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path};
pub const SE2: &str = "10Y1001A1001A45N";
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Interval {
    pub start: i64,
    pub seconds: i64,
    pub value: Option<f64>,
}
#[derive(Deserialize, Default)]
struct Document {
    #[serde(rename = "TimeSeries", default)]
    series: Vec<TimeSeries>,
}
#[derive(Deserialize, Default)]
struct TimeSeries {
    #[serde(rename = "Period", default)]
    periods: Vec<Period>,
    #[serde(rename = "curveType")]
    curve: Option<String>,
    #[serde(rename = "currency_Unit.name")]
    currency: Option<String>,
    #[serde(rename = "quantity_Measure_Unit.name")]
    unit: Option<String>,
}
#[derive(Deserialize)]
struct Period {
    #[serde(rename = "timeInterval")]
    interval: TimeInterval,
    resolution: String,
    #[serde(rename = "Point", default)]
    points: Vec<Point>,
}
#[derive(Deserialize)]
struct TimeInterval {
    start: String,
    end: String,
}
#[derive(Deserialize)]
struct Point {
    position: usize,
    quantity: Option<f64>,
    #[serde(rename = "price.amount")]
    price: Option<f64>,
}
pub fn date(s: &str) -> Result<i64> {
    let owned;
    if s.ends_with("Z") && s.len() == 17 {
        owned = s.replace("Z", ":00Z");
        return date(&owned);
    }
    if let Ok(d) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        Ok(d.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp())
    } else {
        Ok(DateTime::parse_from_rfc3339(s)?.timestamp())
    }
}
pub fn xml(s: &str) -> Result<Vec<Interval>> {
    let d: Document = quick_xml::de::from_str(s).context("invalid ENTSO-E XML")?;
    ensure!(!d.series.is_empty(), "ENTSO-E returned no time series");
    let mut out = BTreeMap::new();
    for series in d.series {
        ensure!(
            series.currency.as_deref().is_none_or(|v| v == "EUR"),
            "price currency must be EUR"
        );
        ensure!(
            series
                .unit
                .as_deref()
                .is_none_or(|v| v == "MAW" || v == "MW"),
            "quantity must be average MW, not interval energy"
        );
        let curve = series.curve.as_deref().unwrap_or("A01");
        ensure!(
            ["A01", "A03"].contains(&curve),
            "unsupported ENTSO-E curve type"
        );
        for p in series.periods {
            let start = date(&p.interval.start)?;
            let end = date(&p.interval.end)?;
            let seconds = match p.resolution.as_str() {
                "PT15M" => 900,
                "PT30M" => 1800,
                "PT60M" => 3600,
                _ => anyhow::bail!("unsupported source resolution {}", p.resolution),
            };
            let mut points = p.points;
            points.sort_by_key(|p| p.position);
            for (index, point) in points.iter().enumerate() {
                ensure!(point.position > 0, "invalid ENTSO-E position");
                let ts = start + (point.position as i64 - 1) * seconds;
                ensure!(ts + seconds <= end, "point outside period");
                let value = point.quantity.or(point.price);
                let stop = if curve == "A03" {
                    points
                        .get(index + 1)
                        .map(|p| start + (p.position as i64 - 1) * seconds)
                        .unwrap_or(end)
                } else {
                    ts + seconds
                };
                ensure!(stop > ts && stop <= end, "invalid variable-block position");
                for pos in (ts..stop).step_by(seconds as usize) {
                    let key = (pos, seconds);
                    if let Some(old) = out.insert(key, value) {
                        ensure!(
                            old == value,
                            "conflicting time series; resolve revisions before aggregation"
                        );
                    }
                }
            }
        }
    }
    Ok(out
        .into_iter()
        .map(|((start, seconds), value)| Interval {
            start,
            seconds,
            value,
        })
        .collect())
}
pub fn hourly(records: &[Interval], from: i64, to: i64) -> Result<Vec<Interval>> {
    ensure!(
        from % 3600 == 0 && to % 3600 == 0 && from < to,
        "range must be aligned UTC hours"
    );
    let mut slots: BTreeMap<i64, Vec<&Interval>> = BTreeMap::new();
    for r in records {
        ensure!(
            r.seconds > 0 && r.seconds <= 3600 && 3600 % r.seconds == 0 && r.start % r.seconds == 0,
            "invalid native interval"
        );
        if r.start >= from && r.start < to {
            slots
                .entry(r.start.div_euclid(3600) * 3600)
                .or_default()
                .push(r);
        }
    }
    let mut out = vec![];
    for hour in (from..to).step_by(3600) {
        let mut parts = slots.remove(&hour).unwrap_or_default();
        parts.sort_by_key(|r| r.start);
        let mut cursor = hour;
        let mut sum = 0.;
        let mut good = true;
        for r in parts {
            ensure!(r.start >= cursor, "overlapping source intervals");
            if r.start != cursor {
                good = false;
            }
            cursor = r.start + r.seconds;
            match r.value {
                Some(x) if x.is_finite() => sum += x * r.seconds as f64,
                _ => good = false,
            };
        }
        out.push(Interval {
            start: hour,
            seconds: 3600,
            value: if good && cursor == hour + 3600 {
                Some(sum / 3600.)
            } else {
                None
            },
        });
    }
    Ok(out)
}
pub fn csv(path: &Path) -> Result<Vec<Interval>> {
    let s = std::fs::read_to_string(path)?;
    let mut records = vec![];
    for (i, line) in s.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') || line.starts_with("timestamp") {
            continue;
        }
        let v: Vec<_> = line.split(',').collect();
        ensure!(
            v.len() == 3,
            "CSV line {}: expected timestamp,duration_seconds,value",
            i + 1
        );
        records.push(Interval {
            start: if let Ok(n) = v[0].parse() {
                n
            } else {
                date(v[0])?
            },
            seconds: v[1].parse()?,
            value: if v[2] == "NA" || v[2].is_empty() {
                None
            } else {
                Some(v[2].parse()?)
            },
        });
    }
    Ok(records)
}
/// Validate the prepared source before normalization or any output is written.
pub fn sha256(path: &Path) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(std::fs::read(path)?)))
}
pub fn source_provenance(
    manifest: &Path,
    csv: &Path,
    kind: &str,
    records: &[Interval],
    from: i64,
    to: i64,
) -> Result<serde_json::Value> {
    let m: serde_json::Value = serde_json::from_slice(&std::fs::read(manifest)?)?;
    ensure!(
        ["load", "wind", "solar"].contains(&kind),
        "unsupported manifest series kind"
    );
    let filename = csv
        .file_name()
        .and_then(|s| s.to_str())
        .context("CSV filename")?;
    ensure!(
        filename == format!("{kind}.csv"),
        "CSV series kind differs from --kind"
    );
    let hash = sha256(csv)?;
    ensure!(
        m["outputs"][filename].as_str() == Some(hash.as_str()),
        "CSV SHA256 differs from source manifest"
    );
    ensure!(
        m["zone"] == "SE2" && m["output_unit"] == "average MW",
        "source must be SE2 average MW"
    );
    ensure!(
        m["interval_seconds"] == 3600,
        "source must have hourly intervals"
    );
    let start = date(m["first_interval_utc"].as_str().context("source start")?)?;
    let stop = date(m["end_exclusive_utc"].as_str().context("source end")?)?;
    ensure!(
        from == start && to == stop,
        "ingestion range differs from declared source coverage"
    );
    ensure!(
        m["rows"].as_u64() == Some(records.len() as u64)
            && records.len() as i64 * 3600 == stop - start,
        "source row count differs from coverage"
    );
    for (i, r) in records.iter().enumerate() {
        ensure!(
            r.start == start + i as i64 * 3600 && r.seconds == 3600,
            "source grid differs from declared hourly coverage"
        );
        ensure!(r.value.is_none_or(f64::is_finite), "nonfinite source value");
    }
    Ok(
        serde_json::json!({"manifest":m,"manifest_sha256":sha256(manifest)?,"csv_sha256":hash,"series_kind":kind}),
    )
}
pub fn client() -> Result<reqwest::blocking::Client> {
    Ok(reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(90))
        .user_agent("energy-timestral/0.1 research")
        .build()?)
}
/// Cache keys and errors deliberately exclude secret query parameters.
pub fn cached(
    url: &str,
    params: &[(String, String)],
    cache: &Path,
    refresh: bool,
) -> Result<Vec<u8>> {
    std::fs::create_dir_all(cache)?;
    let public: Vec<_> = params
        .iter()
        .filter(|(k, _)| k != "securityToken" && k != "apikey")
        .collect();
    let key = format!("{:x}", Sha256::digest(format!("{url}{public:?}")));
    let index = cache.join(format!("{key}.json"));
    if index.exists() && !refresh {
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(&index)?)?;
        return Ok(std::fs::read(
            cache.join(meta["sha256"].as_str().context("cache metadata")?),
        )?);
    }
    let client = client()?;
    let mut bytes = None;
    for attempt in 0..5 {
        let response = client
            .get(url)
            .query(params)
            .send()
            .map_err(|_| anyhow::anyhow!("source request failed (URL redacted)"))?;
        let status = response.status();
        if status.as_u16() == 429 || status.is_server_error() {
            std::thread::sleep(std::time::Duration::from_secs(1 << attempt));
            continue;
        }
        ensure!(
            status.is_success(),
            "source returned HTTP {} (URL redacted)",
            status.as_u16()
        );
        bytes = Some(response.bytes()?.to_vec());
        break;
    }
    let bytes = bytes.context("source rate limit/retry budget exhausted")?;
    let hash = format!("{:x}", Sha256::digest(&bytes));
    if !cache.join(&hash).exists() {
        std::fs::write(cache.join(&hash), &bytes)?;
    }
    let meta =
        serde_json::json!({"url":url,"parameters":public,"retrieved_at":Utc::now(),"sha256":hash});
    let snapshot = cache.join(format!("{key}-{}.json", Utc::now().timestamp_millis()));
    std::fs::write(snapshot, serde_json::to_vec_pretty(&meta)?)?;
    std::fs::write(&index, serde_json::to_vec_pretty(&meta)?)?;
    Ok(bytes)
}
pub fn ingest(
    kind: &str,
    from: i64,
    to: i64,
    cache: &Path,
    refresh: bool,
) -> Result<Vec<Interval>> {
    let token = std::env::var("ENTSOE_TOKEN").context("set ENTSOE_TOKEN or use ingest --csv")?;
    let mut result = vec![];
    ensure!(from < to, "empty ingestion range");
    let mut cursor = Utc
        .timestamp_opt(from, 0)
        .single()
        .context("invalid start date")?
        .date_naive()
        .with_day(1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap()
        .and_utc()
        .timestamp();
    while cursor < to {
        let month = Utc.timestamp_opt(cursor, 0).unwrap().date_naive();
        let end = (month + chrono::Months::new(1))
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp();
        let format = |ts| {
            Utc.timestamp_opt(ts, 0)
                .unwrap()
                .format("%Y%m%d%H%M")
                .to_string()
        };
        let mut params = vec![
            ("securityToken".into(), token.clone()),
            ("periodStart".into(), format(cursor)),
            ("periodEnd".into(), format(end)),
        ];
        if kind == "price" {
            params.extend([
                ("documentType".into(), "A44".into()),
                ("in_Domain".into(), SE2.into()),
                ("out_Domain".into(), SE2.into()),
            ]);
        } else if kind == "load" {
            params.extend([
                ("documentType".into(), "A65".into()),
                ("processType".into(), "A16".into()),
                ("outBiddingZone_Domain".into(), SE2.into()),
            ]);
        } else {
            let psr = match kind {
                "wind" => "B19",
                "wind-offshore" => "B18",
                "solar" => "B16",
                _ => anyhow::bail!("unknown data kind"),
            };
            params.extend([
                ("documentType".into(), "A75".into()),
                ("processType".into(), "A16".into()),
                ("in_Domain".into(), SE2.into()),
                ("psrType".into(), psr.into()),
            ]);
        }
        let body = cached("https://web-api.tp.entsoe.eu/api", &params, cache, refresh)?;
        result.extend(xml(std::str::from_utf8(&body)?)?);
        cursor = end;
    }
    let mut dedup = BTreeMap::new();
    for r in result {
        if let Some(old) = dedup.insert((r.start, r.seconds), r.value) {
            ensure!(old == r.value, "conflicting chunk revisions");
        }
    }
    Ok(dedup
        .into_iter()
        .map(|((start, seconds), value)| Interval {
            start,
            seconds,
            value,
        })
        .collect())
}
pub fn easter(year: i32) -> NaiveDate {
    let a = year % 19;
    let b = year / 100;
    let c = year % 100;
    let d = b / 4;
    let e = b % 4;
    let f = (b + 8) / 25;
    let g = (b - f + 1) / 3;
    let h = (19 * a + b - d - g + 15) % 30;
    let i = c / 4;
    let k = c % 4;
    let l = (32 + 2 * e + 2 * i - h - k) % 7;
    let m = (a + 11 * h + 22 * l) / 451;
    let n = h + l - 7 * m + 114;
    NaiveDate::from_ymd_opt(year, (n / 31) as u32, (n % 31 + 1) as u32).unwrap()
}
pub fn holiday(d: NaiveDate) -> bool {
    let md = (d.month(), d.day());
    let e = easter(d.year());
    [(1, 1), (1, 6), (5, 1), (6, 6), (12, 25), (12, 26)].contains(&md)
        || [-2, 0, 1, 39, 49]
            .iter()
            .any(|n| d == e + Duration::days(*n))
        || (d.weekday() == chrono::Weekday::Sat
            && ((d.month() == 6 && (20..=26).contains(&d.day()))
                || (d.month() == 10 && d.day() == 31)
                || (d.month() == 11 && d.day() <= 6)))
}
pub fn write_canonical(path: &Path, rows: &[Interval], future_hours: i64) -> Result<()> {
    if let Some(p) = path.parent() {
        std::fs::create_dir_all(p)?;
    }
    let mut out = String::new();
    let last = rows.last().context("empty dataset")?.start;
    for r in rows
        .iter()
        .cloned()
        .chain((1..=future_hours).map(|h| Interval {
            start: last + h * 3600,
            seconds: 3600,
            value: None,
        }))
    {
        let local = Utc
            .timestamp_opt(r.start, 0)
            .unwrap()
            .with_timezone(&Stockholm);
        let v = match r.value {
            None => "NA".into(),
            Some(v) => crate::protocol::fixed(v)?.to_string(),
        };
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\n",
            r.start,
            v,
            local.hour() * 1000,
            local.weekday().num_days_from_monday() * 1000,
            local.month() * 1000,
            if holiday(local.date_naive()) { 1000 } else { 0 }
        ));
    }
    std::fs::write(path, out)?;
    Ok(())
}
pub fn delivery(day: NaiveDate) -> Result<(i64, i64, i64)> {
    let cutoff = Stockholm
        .from_local_datetime(&day.and_hms_opt(10, 0, 0).unwrap())
        .single()
        .context("ambiguous cutoff")?
        .timestamp();
    let next = day + Duration::days(1);
    let end = day + Duration::days(2);
    let start = Stockholm
        .from_local_datetime(&next.and_hms_opt(0, 0, 0).unwrap())
        .single()
        .unwrap()
        .timestamp();
    let end = Stockholm
        .from_local_datetime(&end.and_hms_opt(0, 0, 0).unwrap())
        .single()
        .unwrap()
        .timestamp();
    Ok((cutoff, start, end))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partial_hour_missing_and_weighted_price() {
        let rs = (0..4)
            .map(|i| Interval {
                start: i * 900,
                seconds: 900,
                value: Some(i as f64),
            })
            .collect::<Vec<_>>();
        assert_eq!(hourly(&rs, 0, 3600).unwrap()[0].value, Some(1.5));
        assert!(hourly(&rs[..3], 0, 3600).unwrap()[0].value.is_none());
    }
    #[test]
    fn overlap_rejected() {
        let r = Interval {
            start: 0,
            seconds: 3600,
            value: Some(1.),
        };
        assert!(hourly(&[r.clone(), r], 0, 3600).is_err());
    }
    #[test]
    fn dst_delivery_hours() {
        for (d, n) in [("2026-03-28", 23), ("2026-10-24", 25)] {
            let (_, s, e) = delivery(NaiveDate::parse_from_str(d, "%Y-%m-%d").unwrap()).unwrap();
            assert_eq!((e - s) / 3600, n);
        }
    }
    #[test]
    fn xml_price_and_gaps() {
        let s="<Publication_MarketDocument><TimeSeries><Period><timeInterval><start>2026-01-01T00:00Z</start><end>2026-01-01T01:00Z</end></timeInterval><resolution>PT15M</resolution><Point><position>1</position><price.amount>-5.5</price.amount></Point></Period></TimeSeries></Publication_MarketDocument>";
        let r = xml(s).unwrap();
        assert_eq!(r[0].value, Some(-5.5));
    }
    #[test]
    fn variable_blocks_expand_without_inventing_gaps() {
        let s="<d><TimeSeries><curveType>A03</curveType><currency_Unit.name>EUR</currency_Unit.name><Period><timeInterval><start>2026-01-01T00:00Z</start><end>2026-01-01T01:00Z</end></timeInterval><resolution>PT15M</resolution><Point><position>1</position><price.amount>10</price.amount></Point><Point><position>3</position><price.amount>20</price.amount></Point></Period></TimeSeries></d>";
        let r = xml(s).unwrap();
        assert_eq!(r.len(), 4);
        assert_eq!(
            hourly(
                &r,
                date("2026-01-01").unwrap(),
                date("2026-01-01T01:00:00Z").unwrap()
            )
            .unwrap()[0]
                .value,
            Some(15.)
        );
        assert!(xml(&s.replace("EUR", "SEK")).is_err());
    }
}
