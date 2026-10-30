use crate::model::{Request, Response};
use anyhow::{bail, ensure, Result};
use std::io::{Read, Write};
pub const MAX_FRAME: usize = 16 * 1024 * 1024;
pub fn read_frame(r: &mut impl Read) -> Result<String> {
    let mut n = [0u8; 4];
    r.read_exact(&mut n)?;
    let len = u32::from_be_bytes(n) as usize;
    ensure!(len > 0 && len <= MAX_FRAME, "invalid frame size");
    let mut bytes = vec![0; len];
    r.read_exact(&mut bytes)?;
    Ok(String::from_utf8(bytes)?)
}
pub fn write_frame(w: &mut impl Write, s: &str) -> Result<()> {
    ensure!(s.len() <= MAX_FRAME, "frame too large");
    w.write_all(&(s.len() as u32).to_be_bytes())?;
    w.write_all(s.as_bytes())?;
    Ok(())
}
pub fn parse(s: &str) -> Result<Request> {
    let mut lines = s.lines();
    let h: Vec<_> = lines.next().unwrap_or("").split('\t').collect();
    ensure!(
        (h.len() == 7 || h.len() == 8) && h[0] == "ENERGY1",
        "invalid protocol header"
    );
    ensure!(!h[1].is_empty() && h[1].len() <= 128, "invalid request ID");
    let mut r = Request {
        id: h[1].into(),
        horizon: h[2].parse()?,
        max_context: h[3].parse()?,
        symmetric: flag(h[4])?,
        clip_nonnegative: flag(h[5])?,
        trace: flag(h[6])?,
        expected_checkpoint: h.get(7).unwrap_or(&"").to_string(),
        context: vec![],
        past: vec![],
        known: vec![],
        categorical_past: vec![],
        categorical_known: vec![],
    };
    let mut end = false;
    for line in lines {
        ensure!(!end, "trailing protocol data");
        if line == "END" {
            end = true;
            continue;
        }
        let mut cells = line.split('\t');
        let role = cells.next().unwrap_or("");
        let row = cells
            .map(|v| {
                if v == "NA" {
                    Ok(None)
                } else {
                    let n = v.parse::<i64>()?;
                    Ok(Some(n as f32 / 1000.))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        match role {
            "TARGET" => r.context.push(row),
            "PAST" => r.past.push(row),
            "KNOWN" => r.known.push(row),
            "CATPAST" => r
                .categorical_past
                .push(row.into_iter().map(|v| v.map(|n| n as f64)).collect()),
            "CATKNOWN" => r
                .categorical_known
                .push(row.into_iter().map(|v| v.map(|n| n as f64)).collect()),
            _ => bail!("invalid row role"),
        }
    }
    ensure!(end, "missing END");
    Ok(r)
}
fn flag(v: &str) -> Result<bool> {
    match v {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => bail!("invalid boolean"),
    }
}
pub fn request(r: &Request) -> String {
    let mut s = format!(
        "ENERGY1\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
        r.id,
        r.horizon,
        r.max_context,
        r.symmetric as u8,
        r.clip_nonnegative as u8,
        r.trace as u8,
        r.expected_checkpoint
    );
    for (role, rows) in [
        ("TARGET", &r.context),
        ("PAST", &r.past),
        ("KNOWN", &r.known),
    ] {
        for row in rows {
            s.push_str(role);
            for v in row {
                s.push('\t');
                s.push_str(
                    &v.map(|x| fixed(x as f64).unwrap().to_string())
                        .unwrap_or("NA".into()),
                );
            }
            s.push('\n');
        }
    }
    s.push_str("END\n");
    s
}
pub fn fixed(x: f64) -> Result<i64> {
    ensure!(x.is_finite() && x.abs() < 9e12, "fixed-point overflow");
    Ok((x * 1000.).round() as i64)
}
pub fn response(r: &Response) -> Result<String> {
    let mut s = format!(
        "ENERGY1\t{}\tOK\t{}\t{}\t{}\n",
        r.id, r.elapsed_ms, r.device, r.checkpoint_sha256
    );
    s.push_str("LEVELS");
    for q in &r.levels {
        s.push_str(&format!("\t{}", (*q * 10000.).round() as i64));
    }
    s.push('\n');
    for (v, steps) in r.quantiles.iter().enumerate() {
        for (h, qs) in steps.iter().enumerate() {
            s.push_str(&format!("Q\t{v}\t{h}"));
            for x in qs {
                s.push_str(&format!("\t{}", fixed(*x as f64)?));
            }
            s.push('\n');
        }
    }
    s.push_str("END\n");
    Ok(s)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fragmented_frame_and_invalid_length() {
        let mut buf = vec![];
        write_frame(&mut buf, "hello").unwrap();
        assert_eq!(read_frame(&mut &buf[..]).unwrap(), "hello");
        assert!(read_frame(&mut &[255u8; 4][..]).is_err());
    }
    #[test]
    fn strict_request() {
        let r = parse("ENERGY1\tid\t48\t672\t1\t0\t0\nTARGET\t-1234\tNA\nEND\n").unwrap();
        assert_eq!(r.context[0], vec![Some(-1.234), None]);
        assert!(parse("ENERGY1\tid\t48\t672\t2\t0\t0\nEND\n").is_err());
    }
}
