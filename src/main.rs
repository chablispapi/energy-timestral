mod checkpoint;
mod data;
mod gpu;
mod model;
mod protocol;
mod research;
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::{io::Write, path::PathBuf};
#[derive(Parser)]
#[command(
    version,
    about = "SE2 forecasting: Keel application and Timestral GPU worker"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    GpuCheck {
        #[arg(long)]
        intel: bool,
    },
    Infer {
        #[arg(long, default_value = "models/5m")]
        model: PathBuf,
        #[arg(long)]
        request: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        intel: bool,
    },
    Serve {
        #[arg(long, default_value = "models/5m")]
        model: PathBuf,
        #[arg(long, default_value = "127.0.0.1:7419")]
        listen: String,
        #[arg(long)]
        intel: bool,
    },
    Ingest {
        #[arg(long, default_value = "load")]
        kind: String,
        #[arg(long)]
        from: String,
        #[arg(long)]
        to: String,
        #[arg(long)]
        csv: Option<PathBuf>,
        #[arg(long, requires = "csv")]
        source_manifest: Option<PathBuf>,
        #[arg(long, default_value = "data/raw")]
        cache: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        refresh: bool,
    },
    Forecast {
        #[arg(long)]
        config: PathBuf,
    },
    Backtest {
        #[arg(long)]
        config: PathBuf,
    },
    Report {
        directory: PathBuf,
    },
    Freeze {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Checkpoint {
        #[arg(long, default_value = "5m")]
        size: String,
        #[arg(long)]
        revision: Option<String>,
        #[arg(long)]
        output: PathBuf,
    },
}
fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}
fn run() -> Result<()> {
    match Cli::parse().command {
        Command::GpuCheck { intel } => {
            let g = gpu::Gpu::new(intel)?;
            let x = g.upload(&[1., -2., 3.]);
            let z = g.binary(&x, &x, 2)?;
            anyhow::ensure!(g.read(&z)? == vec![2., -4., 6.], "GPU compute failed");
            println!("{}: compute PASS", g.identity);
        }
        Command::Checkpoint {
            size,
            revision,
            output,
        } => checkpoint::download(&size, revision.as_deref(), &output)?,
        Command::Ingest {
            kind,
            from,
            to,
            csv,
            source_manifest,
            cache,
            output,
            refresh,
        } => {
            let start = data::date(&from)?;
            let stop = data::date(&to)?;
            let source = if let Some(p) = &csv {
                data::csv(&p)?
            } else {
                data::ingest(&kind, start, stop, &cache, refresh)?
            };
            let provenance = source_manifest
                .as_deref()
                .map(|p| {
                    data::source_provenance(p, csv.as_deref().unwrap(), &kind, &source, start, stop)
                })
                .transpose()?;
            let hourly = data::hourly(&source, start, stop)?;
            data::write_canonical(&output, &hourly, 96)?;
            let mut manifest = serde_json::json!({"zone":"SE2","eic":data::SE2,"kind":kind,"unit":if kind=="price"{"EUR/MWh"}else{"MW"},"scale":1000,"from":from,"to":to,"retrieved_at":chrono::Utc::now(),"missing_hours":hourly.iter().filter(|r|r.value.is_none()).count(),"vintage_status":"retrospective revised-data; publication-time snapshots unavailable","raw_cache":cache});
            if provenance.is_some() {
                manifest["observed_from"] = serde_json::json!(start);
                manifest["observed_to_exclusive"] = serde_json::json!(stop);
            } else if let (Some(first), Some(last)) = (
                hourly.iter().find(|r| r.value.is_some()),
                hourly.iter().rev().find(|r| r.value.is_some()),
            ) {
                manifest["observed_from"] = serde_json::json!(first.start);
                manifest["observed_to_exclusive"] = serde_json::json!(last.start + 3600);
            }
            manifest["future_calendar_rows"] = serde_json::json!(96);
            manifest["canonical_sha256"] = serde_json::json!(data::sha256(&output)?);
            if let Some(p) = provenance {
                manifest["source_provenance"] = p;
            }
            std::fs::write(
                output.with_extension("json"),
                serde_json::to_vec_pretty(&manifest)?,
            )?;
        }
        Command::Forecast { config } => research::run(&research::read_config(&config)?, true)?,
        Command::Backtest { config } => research::run(&research::read_config(&config)?, false)?,
        Command::Report { directory } => research::report(&directory)?,
        Command::Freeze { config, output } => {
            let c = research::read_config(&config)?;
            anyhow::ensure!(
                c.phase == "development",
                "freeze the chosen development configuration"
            );
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(output)?;
            writeln!(file, "{}", research::selection(&c))?;
        }
        Command::Infer {
            model,
            request,
            output,
            intel,
        } => {
            let m = model::Model::load(&model, intel)?;
            let r = serde_json::from_slice(&std::fs::read(request)?)?;
            std::fs::write(output, serde_json::to_vec(&m.forecast(&r)?)?)?;
        }
        Command::Serve {
            model,
            listen,
            intel,
        } => {
            let addr: std::net::SocketAddr = listen.parse()?;
            anyhow::ensure!(addr.ip().is_loopback(), "worker must bind to loopback");
            let m = model::Model::load(&model, intel)?;
            let socket = std::net::TcpListener::bind(addr)?;
            eprintln!(
                "ready {} {} {}",
                socket.local_addr()?,
                m.gpu.identity,
                m.sha256
            );
            for stream in socket.incoming() {
                let mut stream = stream?;
                stream.set_read_timeout(Some(std::time::Duration::from_secs(120)))?;
                stream.set_write_timeout(Some(std::time::Duration::from_secs(120)))?;
                loop {
                    let body = match protocol::read_frame(&mut stream) {
                        Ok(s) => s,
                        Err(_) => break,
                    };
                    let id = body
                        .lines()
                        .next()
                        .and_then(|s| s.split('\t').nth(1))
                        .unwrap_or("unknown")
                        .to_string();
                    let result = protocol::parse(&body)
                        .and_then(|r| m.forecast(&r))
                        .and_then(|r| protocol::response(&r));
                    let reply = match result {
                        Ok(s) => s,
                        Err(e) => format!(
                            "ENERGY1\t{id}\tERROR\t{}\nEND\n",
                            e.to_string().replace(['\n', '\t'], " ")
                        ),
                    };
                    if protocol::write_frame(&mut stream, &reply).is_err() {
                        break;
                    }
                    stream.flush().context("flushing worker response")?;
                }
            }
        }
    }
    Ok(())
}
