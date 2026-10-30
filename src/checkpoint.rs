use anyhow::{ensure, Context, Result};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::Path;

pub fn download(size: &str, revision: Option<&str>, out: &Path) -> Result<()> {
    let size = match size.to_ascii_lowercase().as_str() {
        "5m" => "5M",
        "28m" => "28M",
        "123m" => "123M",
        "209m" => "209M",
        _ => anyhow::bail!("model size must be 5m, 28m, 123m or 209m"),
    };
    let repo = format!("mistralai/Timestral-{size}");
    let client = crate::data::client()?;
    let rev = if let Some(r) = revision {
        ensure!(
            r.len() == 40 && r.chars().all(|c| c.is_ascii_hexdigit()),
            "revision must be an immutable 40-character commit"
        );
        r.to_string()
    } else {
        let v: serde_json::Value = client
            .get(format!("https://huggingface.co/api/models/{repo}"))
            .send()?
            .error_for_status()?
            .json()?;
        v["sha"].as_str().context("Hub revision")?.into()
    };
    if out.join("checkpoint.json").exists() {
        let old: serde_json::Value =
            serde_json::from_slice(&std::fs::read(out.join("checkpoint.json"))?)?;
        ensure!(
            old["revision"] == rev && old["repository"] == repo,
            "checkpoint directory already pins a different model"
        );
        for name in ["config.json", "model.safetensors"] {
            let bytes = std::fs::read(out.join(name))?;
            ensure!(
                old["files"][name] == format!("{:x}", Sha256::digest(&bytes)),
                "cached checkpoint hash mismatch"
            );
        }
        return Ok(());
    }
    std::fs::create_dir_all(out)?;
    for name in ["config.json", "model.safetensors", "LICENSE"] {
        let url = format!("https://huggingface.co/{repo}/resolve/{rev}/{name}");
        let mut response = client.get(url).send()?.error_for_status()?;
        let tmp = out.join(format!("{name}.partial"));
        let mut file = std::fs::File::create(&tmp)?;
        std::io::copy(&mut response, &mut file)?;
        file.sync_all()?;
        std::fs::rename(tmp, out.join(name))?;
    }
    let config = std::fs::read(out.join("config.json"))?;
    let weights = std::fs::read(out.join("model.safetensors"))?;
    let mut h = Sha256::new();
    h.update(&config);
    h.update(&weights);
    let meta = json!({"repository":repo,"revision":rev,"fingerprint":format!("{:x}",h.finalize()),"files":{"config.json":format!("{:x}",Sha256::digest(&config)),"model.safetensors":format!("{:x}",Sha256::digest(&weights))}});
    std::fs::write(
        out.join("checkpoint.json"),
        serde_json::to_vec_pretty(&meta)?,
    )?;
    println!("{}", meta["fingerprint"].as_str().unwrap());
    Ok(())
}
