//! Faithful FP32 Timestral inference. Tensor contractions and transformer operations
//! execute on wgpu; statistics and protocol conversion execute on the host.
use crate::gpu::{Gpu, Params, Tensor};
use anyhow::{ensure, Context, Result};
use safetensors::{Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::Path,
};

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub patch_size: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub mlp_hidden_dim: usize,
    pub block_schedule: Vec<String>,
    pub quantile_knots: Vec<f32>,
    pub block_size: usize,
    pub context_length: usize,
    pub rope_theta: f32,
    pub rope_partial_factor: f32,
    pub norm_eps: f32,
    pub attn_bias: bool,
    pub mlp_bias: bool,
    pub time_attn_gate: String,
    pub variate_attn_gate: String,
    pub minimum_scale: f32,
    pub use_asinh: bool,
    pub stabilize_with_global: bool,
    pub covariate_enabled: bool,
    pub split_context_horizon_attn: bool,
    pub attend_missing_patches: bool,
    pub quantile_width_factor: f32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Request {
    pub id: String,
    pub context: Vec<Vec<Option<f32>>>,
    #[serde(default)]
    pub past: Vec<Vec<Option<f32>>>,
    #[serde(default)]
    pub known: Vec<Vec<Option<f32>>>,
    #[serde(default)]
    pub categorical_past: Vec<Vec<Option<f64>>>,
    #[serde(default)]
    pub categorical_known: Vec<Vec<Option<f64>>>,
    pub horizon: usize,
    #[serde(default = "yes")]
    pub symmetric: bool,
    #[serde(default = "yes")]
    pub clip_nonnegative: bool,
    #[serde(default = "default_context")]
    pub max_context: usize,
    #[serde(default)]
    pub trace: bool,
    #[serde(default)]
    pub expected_checkpoint: String,
}
fn yes() -> bool {
    true
}
fn default_context() -> usize {
    672
}

#[derive(Serialize, Deserialize)]
pub struct Response {
    pub id: String,
    pub levels: Vec<f32>,
    pub quantiles: Vec<Vec<Vec<f32>>>,
    pub device: String,
    pub elapsed_ms: u128,
    pub checkpoint_sha256: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub trace: BTreeMap<String, Vec<f32>>,
}
pub struct Model {
    pub gpu: Gpu,
    pub config: Config,
    weights: HashMap<String, Tensor>,
    pub sha256: String,
}

impl Model {
    pub fn load(path: &Path, intel: bool) -> Result<Self> {
        use sha2::{Digest, Sha256};
        let config_bytes = std::fs::read(path.join("config.json"))?;
        let c: Config = serde_json::from_slice(&config_bytes)?;
        ensure!(
            !c.stabilize_with_global,
            "unsupported stabilize_with_global checkpoint"
        );
        ensure!(
            c.patch_size > 0
                && c.num_heads > 0
                && c.d_model.is_multiple_of(c.num_heads)
                && c.block_size > 0,
            "invalid model dimensions"
        );
        ensure!(
            c.quantile_knots.contains(&0.5) && c.quantile_knots.windows(2).all(|w| w[0] < w[1]),
            "invalid quantile levels"
        );
        for kind in &c.block_schedule {
            ensure!(
                ["time", "variate", "time_variate", "variate_time"].contains(&kind.as_str()),
                "invalid block schedule"
            );
        }
        for gate in [&c.time_attn_gate, &c.variate_attn_gate] {
            ensure!(
                ["none", "sdpa_sigmoid"].contains(&gate.as_str()),
                "invalid attention gate"
            );
        }
        let gpu = Gpu::new(intel)?;
        let bytes = std::fs::read(path.join("model.safetensors"))?;
        let mut hash = Sha256::new();
        hash.update(&config_bytes);
        hash.update(&bytes);
        let sha256 = format!("{:x}", hash.finalize());
        let safe = SafeTensors::deserialize(&bytes)?;
        let mut expected: HashMap<String, Vec<usize>> = HashMap::new();
        fn linear(e: &mut HashMap<String, Vec<usize>>, s: &str, i: usize, o: usize, b: bool) {
            e.insert(format!("{s}.weight"), vec![o, i]);
            if b {
                e.insert(format!("{s}.bias"), vec![o]);
            }
        }
        fn residual(
            e: &mut HashMap<String, Vec<usize>>,
            s: &str,
            i: usize,
            h: usize,
            o: usize,
            b: bool,
        ) {
            linear(e, &format!("{s}.hidden_layer"), i, h, b);
            linear(e, &format!("{s}.output_layer"), h, o, b);
            linear(e, &format!("{s}.residual_layer"), i, o, b);
        }
        residual(
            &mut expected,
            "representation.patch_proj",
            2 * c.patch_size,
            4 * c.d_model,
            c.d_model,
            true,
        );
        expected.insert("representation.forecast_query".into(), vec![c.d_model]);
        for (l, kind) in c.block_schedule.iter().enumerate() {
            for (a, axis) in axes(kind).iter().enumerate() {
                let prefix = format!("transformer.layers.{l}");
                expected.insert(format!("{prefix}.attn_norms.{a}.scale"), vec![c.d_model]);
                let base = format!("{prefix}.attns.{a}");
                linear(
                    &mut expected,
                    &format!("{base}.wQKV"),
                    c.d_model,
                    3 * c.d_model,
                    c.attn_bias,
                );
                linear(
                    &mut expected,
                    &format!("{base}.wO"),
                    c.d_model,
                    c.d_model,
                    c.attn_bias,
                );
                if (if *axis {
                    &c.time_attn_gate
                } else {
                    &c.variate_attn_gate
                }) == "sdpa_sigmoid"
                {
                    linear(
                        &mut expected,
                        &format!("{base}._gate"),
                        c.d_model,
                        c.d_model,
                        c.attn_bias,
                    );
                }
            }
            let p = format!("transformer.layers.{l}");
            expected.insert(format!("{p}.mlp_norm.scale"), vec![c.d_model]);
            linear(
                &mut expected,
                &format!("{p}.mlp.w12"),
                c.d_model,
                2 * c.mlp_hidden_dim,
                c.mlp_bias,
            );
            linear(
                &mut expected,
                &format!("{p}.mlp.w3"),
                c.mlp_hidden_dim,
                c.d_model,
                c.mlp_bias,
            );
        }
        expected.insert("transformer.out_norm.scale".into(), vec![c.d_model]);
        residual(
            &mut expected,
            "head.projection",
            c.d_model,
            4 * c.d_model,
            c.patch_size * c.quantile_knots.len(),
            false,
        );
        let actual: HashSet<_> = safe.names().into_iter().cloned().collect();
        ensure!(
            actual == expected.keys().cloned().collect(),
            "checkpoint tensor names differ from architecture"
        );
        let mut weights = HashMap::new();
        for (name, shape) in expected {
            let t = safe.tensor(&name)?;
            ensure!(
                t.dtype() == Dtype::F32 && t.shape() == shape,
                "checkpoint shape/dtype mismatch: {name}"
            );
            ensure!(
                t.data().len() as u64 <= gpu.max_buffer,
                "weight exceeds device buffer limit"
            );
            let values: Vec<f32> = t
                .data()
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            weights.insert(name, gpu.upload(&values));
        }
        Ok(Self {
            gpu,
            config: c,
            weights,
            sha256,
        })
    }
    fn w(&self, n: &str) -> Result<&Tensor> {
        self.weights
            .get(n)
            .with_context(|| format!("missing weight {n}"))
    }
    fn linear(&self, x: &Tensor, n: &str, i: usize, o: usize) -> Result<Tensor> {
        let zeros = self.gpu.upload(&vec![0.; o]);
        let b = self.weights.get(&format!("{n}.bias")).unwrap_or(&zeros);
        self.gpu.linear(x, self.w(&format!("{n}.weight"))?, b, i, o)
    }
    fn residual(&self, x: &Tensor, n: &str, i: usize, h: usize, o: usize) -> Result<Tensor> {
        let hidden = self.linear(x, &format!("{n}.hidden_layer"), i, h)?;
        let act = self.gpu.unary(&hidden, 3, h)?;
        let out = self.linear(&act, &format!("{n}.output_layer"), h, o)?;
        let skip = self.linear(x, &format!("{n}.residual_layer"), i, o)?;
        self.gpu.binary(&out, &skip, 2)
    }
    pub fn forecast(&self, r: &Request) -> Result<Response> {
        let start = std::time::Instant::now();
        let c = &self.config;
        ensure!(
            r.expected_checkpoint.is_empty() || r.expected_checkpoint == self.sha256,
            "checkpoint fingerprint mismatch"
        );
        ensure!(
            !r.context.is_empty() && !r.context[0].is_empty(),
            "empty context"
        );
        let t = r.context[0].len();
        ensure!(r.context.iter().all(|x| x.len() == t), "ragged context");
        ensure!((1..=1024).contains(&r.horizon), "horizon must be 1..1024");
        ensure!(
            (c.patch_size..=c.context_length).contains(&r.max_context),
            "invalid max_context"
        );
        for x in &r.past {
            ensure!(x.len() == t, "past covariate length");
        }
        for x in &r.known {
            ensure!(x.len() == t + r.horizon, "known covariate length");
        }
        for x in &r.categorical_past {
            ensure!(x.len() == t, "categorical past length");
        }
        for x in &r.categorical_known {
            ensure!(x.len() == t + r.horizon, "categorical known length");
        }
        ensure!(
            r.context
                .iter()
                .chain(&r.past)
                .chain(&r.known)
                .flatten()
                .all(|x| x.is_none_or(|v| v.is_finite())),
            "nonfinite value; use null for missing"
        );
        let mut req = r.clone();
        let encoded_target = if r.context.len() == 1 {
            Some(encoding_target(&r.context[0], r.max_context))
        } else {
            None
        };
        for codes in &r.categorical_past {
            req.past.push(encode(codes, encoded_target.as_deref(), t));
        }
        for codes in &r.categorical_known {
            req.known.push(encode(codes, encoded_target.as_deref(), t));
        }
        let cov = !req.past.is_empty() || !req.known.is_empty();
        ensure!(
            !cov || c.covariate_enabled,
            "checkpoint does not support covariates"
        );
        if !req.known.is_empty() {
            let ops: Vec<_> = c.block_schedule.iter().flat_map(|s| axes(s)).collect();
            ensure!(
                ops.iter().enumerate().any(|(i, x)| *x
                    && ops[i + 1..].contains(&true)
                    && ops[i + 1..].contains(&false)),
                "checkpoint cannot route known future covariates"
            );
        }
        let mut trace = BTreeMap::new();
        let mut out = self.raw(&req, &mut trace)?;
        if r.symmetric {
            let mut mirrored = req.clone();
            mirrored.trace = false;
            for row in mirrored
                .context
                .iter_mut()
                .chain(&mut mirrored.past)
                .chain(&mut mirrored.known)
            {
                for v in row.iter_mut().flatten() {
                    *v = -*v;
                }
            }
            let other = self.raw(&mirrored, &mut BTreeMap::new())?;
            let nq = c.quantile_knots.len();
            ensure!(
                c.quantile_knots
                    .iter()
                    .zip(c.quantile_knots.iter().rev())
                    .all(|(a, b)| (a + b - 1.).abs() < 1e-6),
                "asymmetric quantile levels"
            );
            for v in 0..out.len() {
                for h in 0..r.horizon {
                    for q in 0..nq {
                        out[v][h][q] = 0.5 * (out[v][h][q] - other[v][h][nq - 1 - q]);
                    }
                }
            }
        }
        for (v, row) in r.context.iter().enumerate() {
            let positive =
                row.iter().flatten().next().is_some() && row.iter().flatten().all(|x| *x >= 0.);
            for qs in &mut out[v] {
                for x in qs {
                    if r.clip_nonnegative && positive {
                        *x = x.max(0.);
                    }
                    ensure!(x.is_finite(), "nonfinite forecast");
                }
            }
        }
        Ok(Response {
            id: r.id.clone(),
            levels: c.quantile_knots.clone(),
            quantiles: out,
            device: self.gpu.identity.clone(),
            elapsed_ms: start.elapsed().as_millis(),
            checkpoint_sha256: self.sha256.clone(),
            trace,
        })
    }
    fn raw(
        &self,
        r: &Request,
        trace: &mut BTreeMap<String, Vec<f32>>,
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        let c = &self.config;
        let ps = c.patch_size;
        let nt = r.context.len();
        let nq = c.quantile_knots.len();
        let mid = nq / 2;
        let mut rows: Vec<Vec<f32>> = r.context.iter().map(|x| ffill(x)).collect();
        rows.extend(
            r.past
                .iter()
                .map(|x| x.iter().map(|v| v.unwrap_or(0.)).collect::<Vec<_>>()),
        );
        rows.extend(r.known.iter().map(|x| {
            x[..r.context[0].len()]
                .iter()
                .map(|v| v.unwrap_or(0.))
                .collect::<Vec<_>>()
        }));
        let mut obs: Vec<Vec<bool>> = r
            .context
            .iter()
            .chain(&r.past)
            .map(|x| x.iter().map(Option::is_some).collect())
            .collect();
        obs.extend(r.known.iter().map(|x| {
            x[..r.context[0].len()]
                .iter()
                .map(Option::is_some)
                .collect::<Vec<_>>()
        }));
        let roles: Vec<usize> = (0..rows.len())
            .map(|i| {
                if i < nt {
                    0
                } else if i < nt + r.past.len() {
                    1
                } else {
                    2
                }
            })
            .collect();
        for (x, m) in rows.iter_mut().zip(&mut obs) {
            let drop = x.len().saturating_sub(r.max_context);
            x.drain(..drop);
            m.drain(..drop);
            let pad = (ps - x.len() % ps) % ps;
            x.splice(0..0, vec![0.; pad]);
            m.splice(0..0, vec![false; pad]);
        }
        let original = rows.clone();
        let original_obs = obs.clone();
        let mut out = vec![Vec::new(); nt];
        let mut offset = 0;
        while offset < r.horizon {
            let hpatch = (r.horizon - offset).div_ceil(ps).min(c.block_size);
            let h = hpatch * ps;
            let mut future = vec![vec![0.; h]; rows.len()];
            let mut fm = vec![vec![false; h]; rows.len()];
            for (j, row) in r.known.iter().enumerate() {
                let rr = nt + r.past.len() + j;
                for k in 0..h.min(r.horizon - offset) {
                    let x = row[r.context[0].len() + offset + k];
                    future[rr][k] = x.unwrap_or(0.);
                    fm[rr][k] = x.is_some();
                }
            }
            let mut pred = self.block(
                &rows,
                &obs,
                &roles,
                &future,
                &fm,
                hpatch,
                r.trace && offset == 0,
                trace,
            )?;
            // Move later forecast patches into the advancing refinement frame.
            for rr in 0..nt {
                let (loc0, scale0) = stats(&rows[rr], &obs[rr], c.minimum_scale);
                if hpatch > 1 {
                    let mut state = Moments::from(&rows[rr], &obs[rr]);
                    let offset_mean = state.mean;
                    for patch in 0..hpatch {
                        let (loc, scale) = if patch == 0 {
                            (loc0, scale0)
                        } else {
                            (state.mean as f32, state.scale(c.minimum_scale) as f32)
                        };
                        let mut folded = Vec::new();
                        for qs in &mut pred[rr][patch * ps..(patch + 1) * ps] {
                            let median_norm = (qs[mid] as f64 - loc0 as f64) / scale0 as f64;
                            folded.push(
                                ((median_norm * state.scale(c.minimum_scale) + state.mean
                                    - offset_mean)
                                    .clamp(-1.8446743e16, 1.8446743e16))
                                    + offset_mean,
                            );
                            for x in qs.iter_mut() {
                                *x = (*x - loc0) / scale0 * scale + loc;
                            }
                            clamp_nonfinite(qs);
                        }
                        state.combine(&folded);
                    }
                }
                out[rr].extend_from_slice(&pred[rr][..h.min(r.horizon - offset)]);
            }
            offset += h;
            if offset < r.horizon {
                for rr in 0..rows.len() {
                    for k in 0..h {
                        let (v, m) = match roles[rr] {
                            0 => {
                                let x = pred[rr][k][mid];
                                (
                                    if x.is_finite() && x.abs() < 1e30 {
                                        x
                                    } else {
                                        0.
                                    },
                                    true,
                                )
                            }
                            1 => (0., false),
                            _ => (future[rr][k], fm[rr][k]),
                        };
                        rows[rr].push(v);
                        obs[rr].push(m);
                    }
                    if rows[rr].len() > r.max_context {
                        let keep = (r.max_context / ps * ps).max(ps);
                        let n = rows[rr].len() - keep;
                        rows[rr].drain(..n);
                        obs[rr].drain(..n);
                    }
                }
            }
        }
        for rr in 0..nt {
            let (_, scale) = stats(&original[rr], &original_obs[rr], c.minimum_scale);
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for (v, m) in original[rr].iter().zip(&original_obs[rr]) {
                if *m {
                    lo = lo.min(*v);
                    hi = hi.max(*v);
                }
            }
            if !lo.is_finite() {
                lo = 0.;
                hi = 0.;
            }
            for qs in &mut out[rr] {
                for q in qs {
                    *q = q.clamp(lo - scale * 1e4, hi + scale * 1e4);
                }
            }
        }
        Ok(out)
    }
    // This boundary mirrors the upstream predict_block tensor inputs.
    #[allow(clippy::too_many_arguments)]
    fn block(
        &self,
        rows: &[Vec<f32>],
        obs: &[Vec<bool>],
        roles: &[usize],
        future: &[Vec<f32>],
        fm: &[Vec<bool>],
        hp: usize,
        trace_on: bool,
        trace: &mut BTreeMap<String, Vec<f32>>,
    ) -> Result<Vec<Vec<Vec<f32>>>> {
        let c = &self.config;
        let ps = c.patch_size;
        let d = c.d_model;
        let nr = rows.len();
        let ll = rows[0].len() / ps;
        let cov = roles.iter().any(|x| *x != 0);
        let ne = ll + if cov { hp } else { 0 };
        let nd = if cov { ne } else { hp };
        let slots = ne + nd;
        // Bound masks, activations and weights before allocating on a 4 GB UMA host.
        let estimate =
            (nr * slots * slots * c.num_heads + nr * slots * (12 * d + 4 * c.mlp_hidden_dim)) * 4;
        ensure!(
            estimate < 512 * 1024 * 1024 && nr <= 32 && slots <= 4096,
            "request exceeds worker memory budget"
        );
        ensure!(
            (nr * slots * slots * 4) as u64 <= self.gpu.max_buffer
                && (slots * nr * nr * 4) as u64 <= self.gpu.max_buffer,
            "attention mask exceeds GPU buffer limit"
        );
        let mut embedded = Vec::new();
        let mut valid = vec![vec![false; ne]; nr];
        let mut anchors = Vec::new();
        let mut first = ne;
        for rr in 0..nr {
            let mut values = rows[rr].clone();
            let mut mask = obs[rr].clone();
            if cov {
                values.extend(&future[rr]);
                mask.extend(&fm[rr]);
            }
            let mut locs = vec![0.; values.len()];
            let mut scales = vec![c.minimum_scale; values.len()];
            for p in 0..ne {
                let end = (p + 1) * ps;
                let (loc, s) = stats(&values[..end], &mask[..end], c.minimum_scale);
                for k in p * ps..end {
                    locs[k] = loc;
                    scales[k] = s;
                }
            }
            anchors.push(stats(&rows[rr], &obs[rr], c.minimum_scale));
            let mut binary = false;
            if roles[rr] == 2 {
                binary = values
                    .iter()
                    .zip(&mask)
                    .filter(|(_, m)| **m)
                    .all(|(x, _)| *x == 0. || *x == 1.)
                    && mask.iter().any(|x| *x);
                let count = mask.iter().filter(|x| **x).count().max(1) as f32;
                let mean = (values
                    .iter()
                    .zip(&mask)
                    .filter(|(_, m)| **m)
                    .map(|(v, _)| *v as f64)
                    .sum::<f64>()
                    / count as f64) as f32;
                let variance = (values
                    .iter()
                    .zip(&mask)
                    .filter(|(_, m)| **m)
                    .map(|(v, _)| (*v - mean).powi(2) as f64)
                    .sum::<f64>()
                    / count as f64) as f32;
                locs.fill(if binary { 0. } else { mean });
                scales.fill(if binary {
                    1.
                } else {
                    variance.sqrt().max(c.minimum_scale)
                });
            }
            let mut started = false;
            for p in 0..ne {
                let observed = mask[p * ps..(p + 1) * ps].iter().any(|x| *x);
                if observed {
                    first = first.min(p);
                    started = true;
                }
                valid[rr][p] = observed || (c.attend_missing_patches && started);
                for k in p * ps..(p + 1) * ps {
                    let v = if mask[k] {
                        (values[k] - locs[k]) / scales[k].max(f32::EPSILON)
                    } else {
                        0.
                    };
                    embedded.push(if c.use_asinh && !binary { v.asinh() } else { v });
                }
                for observed in &mask[p * ps..(p + 1) * ps] {
                    embedded.push(if *observed { 0. } else { 1. });
                }
            }
        }
        if first == ne {
            first = 0;
        }
        let x = self.residual(
            &self.gpu.upload(&embedded),
            "representation.patch_proj",
            2 * ps,
            4 * d,
            d,
        )?;
        let enc = self.gpu.read(&x)?; // packing encoder and query tokens is host layout work only
        let query = self.gpu.read(self.w("representation.forecast_query")?)?;
        let mut packed = Vec::new();
        let mut positions = Vec::new();
        for rr in 0..nr {
            packed.extend_from_slice(&enc[rr * ne * d..(rr + 1) * ne * d]);
            for _ in 0..nd {
                packed.extend(&query);
            }
            for i in 0..slots {
                let p = if cov { i % ne } else { i };
                positions.push(p as f32 - first as f32);
            }
        }
        let mut time = vec![0.; nr * slots * slots];
        let mut variate = vec![0.; slots * nr * nr];
        for rr in 0..nr {
            for q in 0..slots {
                for k in 0..slots {
                    let allowed = if q < ne {
                        k < ne && ((k <= q || roles[rr] == 2) && valid[rr][k] || k == q)
                    } else if !cov {
                        if k < ne {
                            valid[rr][k]
                        } else {
                            k <= q
                        }
                    } else {
                        let qp = q - ne;
                        let active = qp >= ll;
                        let anchor = if active {
                            ll + (qp - ll) / c.block_size * c.block_size
                        } else {
                            0
                        };
                        if k < ne {
                            active && k < anchor && valid[rr][k]
                        } else {
                            let kp = k - ne;
                            let ka = if kp >= ll {
                                ll + (kp - ll) / c.block_size * c.block_size
                            } else {
                                0
                            };
                            q == k || (active && kp >= ll && kp <= qp && anchor == ka)
                        }
                    };
                    time[(rr * slots + q) * slots + k] = if allowed { 1. } else { 0. };
                }
            }
        }
        for s in 0..slots {
            for q in 0..nr {
                for k in 0..nr {
                    let firewall = roles[q] != 0 && roles[k] == 0 || roles[q] == 2 && roles[k] != 2;
                    let allow = q == k || (!firewall && (s >= ne || valid[k][s]));
                    variate[(s * nr + q) * nr + k] = if allow { 1. } else { 0. };
                }
            }
        }
        let tm = self.gpu.upload(&time);
        let vm = self.gpu.upload(&variate);
        let pos = self.gpu.upload(&positions);
        let mut x = self.gpu.upload(&packed);
        if trace_on {
            trace.insert("embedding".into(), packed);
        }
        for (l, kind) in c.block_schedule.iter().enumerate() {
            let prefix = format!("transformer.layers.{l}");
            for (a, is_time) in axes(kind).into_iter().enumerate() {
                let norm = self.gpu.norm(
                    &x,
                    self.w(&format!("{prefix}.attn_norms.{a}.scale"))?,
                    c.norm_eps,
                )?;
                let base = format!("{prefix}.attns.{a}");
                let mut qkv = self.linear(&norm, &format!("{base}.wQKV"), d, 3 * d)?;
                if is_time {
                    qkv = self.gpu.op(
                        &qkv,
                        &pos,
                        &pos,
                        Params {
                            op: 6,
                            m: (nr * slots) as u32,
                            n: (3 * d) as u32,
                            heads: c.num_heads as u32,
                            theta: c.rope_theta,
                            rotary: ((d / c.num_heads) as f32 * c.rope_partial_factor) as u32,
                            ..Default::default()
                        },
                        qkv.len,
                    )?;
                }
                let (batch, queries, keys, mask) = if is_time {
                    (nr, slots, slots, &tm)
                } else {
                    (slots, nr, nr, &vm)
                };
                let scores = self.gpu.op(
                    &qkv,
                    mask,
                    mask,
                    Params {
                        op: 7,
                        m: batch as u32,
                        n: queries as u32,
                        k: keys as u32,
                        rows: d as u32,
                        slots: slots as u32,
                        heads: c.num_heads as u32,
                        axis: (!is_time) as u32,
                        ..Default::default()
                    },
                    batch * queries * c.num_heads * keys,
                )?;
                let mut att = self.gpu.op(
                    &qkv,
                    &scores,
                    &scores,
                    Params {
                        op: 8,
                        m: (nr * slots) as u32,
                        n: d as u32,
                        k: keys as u32,
                        rows: queries as u32,
                        slots: slots as u32,
                        heads: c.num_heads as u32,
                        axis: (!is_time) as u32,
                        ..Default::default()
                    },
                    nr * slots * d,
                )?;
                if (if is_time {
                    &c.time_attn_gate
                } else {
                    &c.variate_attn_gate
                }) == "sdpa_sigmoid"
                {
                    let gate = self.linear(&norm, &format!("{base}._gate"), d, d)?;
                    att = self.gpu.binary(&att, &gate, 5)?;
                }
                let projected = self.linear(&att, &format!("{base}.wO"), d, d)?;
                x = self.gpu.binary(&x, &projected, 2)?;
            }
            let norm =
                self.gpu
                    .norm(&x, self.w(&format!("{prefix}.mlp_norm.scale"))?, c.norm_eps)?;
            let fused =
                self.linear(&norm, &format!("{prefix}.mlp.w12"), d, 2 * c.mlp_hidden_dim)?;
            let activated = self.gpu.unary(&fused, 4, c.mlp_hidden_dim)?;
            let out = self.linear(&activated, &format!("{prefix}.mlp.w3"), c.mlp_hidden_dim, d)?;
            x = self.gpu.binary(&x, &out, 2)?;
            if trace_on {
                trace.insert(format!("layer_{l}"), self.gpu.read(&x)?);
            }
        }
        let normalized = self
            .gpu
            .norm(&x, self.w("transformer.out_norm.scale")?, c.norm_eps)?;
        let host = self.gpu.read(&normalized)?;
        let mut dec = Vec::new();
        for rr in 0..nr {
            let from = rr * slots + ne + if cov { ll } else { 0 };
            dec.extend_from_slice(&host[from * d..(from + hp) * d]);
        }
        let head = self.residual(
            &self.gpu.upload(&dec),
            "head.projection",
            d,
            4 * d,
            ps * c.quantile_knots.len(),
        )?;
        let raw = self.gpu.read(&head)?;
        if trace_on {
            trace.insert("head_raw".into(), raw.clone());
        }
        let nq = c.quantile_knots.len();
        let mid = nq / 2;
        let mut out = vec![vec![vec![0.; nq]; hp * ps]; nr];
        for (rr, row) in out.iter_mut().enumerate() {
            for (k, qs) in row.iter_mut().enumerate() {
                let offset = (rr * hp * ps + k) * nq;
                let values = &raw[offset..offset + nq];
                qs[mid] = values[mid];
                for q in mid + 1..nq {
                    qs[q] = qs[q - 1]
                        + c.quantile_width_factor
                            * (c.quantile_knots[q] - c.quantile_knots[q - 1])
                            * softplus(values[q]);
                }
                for q in (0..mid).rev() {
                    qs[q] = qs[q + 1]
                        - c.quantile_width_factor
                            * (c.quantile_knots[q + 1] - c.quantile_knots[q])
                            * softplus(values[q]);
                }
                for x in qs.iter_mut() {
                    *x = if c.use_asinh { x.sinh() } else { *x };
                    *x = *x * anchors[rr].1 + anchors[rr].0;
                }
                clamp_nonfinite(qs);
            }
        }
        Ok(out)
    }
}
fn axes(s: &str) -> Vec<bool> {
    match s {
        "time" => vec![true],
        "variate" => vec![false],
        "time_variate" => vec![true, false],
        _ => vec![false, true],
    }
}
fn softplus(x: f32) -> f32 {
    if x > 20. {
        x
    } else {
        x.exp().ln_1p()
    }
}
fn clamp_nonfinite(x: &mut [f32]) {
    let lo = x
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::INFINITY, f32::min);
    let hi = x
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    for v in x {
        if *v == f32::INFINITY {
            *v = if hi.is_finite() { hi } else { 0. };
        } else if *v == f32::NEG_INFINITY {
            *v = if lo.is_finite() { lo } else { 0. };
        }
    }
}
#[derive(Default)]
struct Moments {
    count: f64,
    mean: f64,
    m2: f64,
}
impl Moments {
    fn from(x: &[f32], m: &[bool]) -> Self {
        let mut s = Self::default();
        for (v, ok) in x.iter().zip(m) {
            if *ok {
                s.add(*v as f64);
            }
        }
        s
    }
    fn add(&mut self, v: f64) {
        self.count += 1.;
        let d = v - self.mean;
        self.mean += d / self.count;
        self.m2 += d * (v - self.mean);
    }
    fn scale(&self, min: f32) -> f64 {
        (self.m2 / (self.count - 1.).max(1.))
            .max(0.)
            .sqrt()
            .max(min as f64)
    }
    fn combine(&mut self, values: &[f64]) {
        for x in values {
            self.add(*x);
        }
    }
}
fn stats(x: &[f32], m: &[bool], min: f32) -> (f32, f32) {
    let s = Moments::from(x, m);
    (s.mean as f32, s.scale(min) as f32)
}
fn ffill(x: &[Option<f32>]) -> Vec<f32> {
    let mut prev = 0.;
    x.iter()
        .map(|v| {
            if let Some(y) = v {
                prev = *y;
            }
            prev
        })
        .collect()
}
fn encoding_target(x: &[Option<f32>], max: usize) -> Vec<Option<f64>> {
    let mut prev = None;
    x.iter()
        .enumerate()
        .map(|(i, v)| {
            if let Some(y) = v {
                prev = Some(*y as f64);
            }
            if i + max < x.len() {
                None
            } else {
                prev
            }
        })
        .collect()
}
fn encode(codes: &[Option<f64>], target: Option<&[Option<f64>]>, t: usize) -> Vec<Option<f32>> {
    let Some(target) = target else {
        return codes.iter().map(|c| c.map(|x| x as f32)).collect();
    };
    let mut sum = 0.;
    let mut count = 0.;
    let mut buckets: HashMap<u64, (f64, f64)> = HashMap::new();
    let key = |c: Option<f64>| {
        c.map(|v| if v == 0. { 0 } else { v.to_bits() })
            .unwrap_or(u64::MAX)
    };
    for i in 0..t {
        if let Some(v) = target[i] {
            sum += v;
            count += 1.;
            let b = buckets.entry(key(codes[i])).or_default();
            b.0 += v;
            b.1 += 1.;
        }
    }
    let mean = if count > 0. { sum / count } else { 0. };
    codes
        .iter()
        .enumerate()
        .map(|(i, c)| {
            if i < t && target[i].is_none() && c.is_none() {
                None
            } else {
                let (s, n) = buckets.get(&key(*c)).copied().unwrap_or_default();
                Some(((mean + s) / (1. + n)) as f32)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn causal_stats_ignore_missing() {
        assert_eq!(
            stats(&[1., 999., 3.], &[true, false, true], 1e-6),
            (2., 2f32.sqrt())
        );
    }
    #[test]
    fn categorical_unseen_and_padding() {
        let t = vec![None, Some(2.), Some(4.)];
        let c = vec![None, Some(1.), Some(1.), Some(2.)];
        assert_eq!(
            encode(&c, Some(&t), 3),
            vec![None, Some(3.), Some(3.), Some(3.)]
        );
    }
}
