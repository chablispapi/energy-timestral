use anyhow::{bail, Context, Result};
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub op: u32,
    pub rows: u32,
    pub slots: u32,
    pub heads: u32,
    pub axis: u32,
    pub eps: f32,
    pub theta: f32,
    pub rotary: u32,
    pub unused: u32,
}

#[derive(Clone)]
pub struct Tensor {
    pub buffer: Arc<wgpu::Buffer>,
    pub len: usize,
}

pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    pub identity: String,
    pub max_buffer: u64,
}

impl Gpu {
    pub fn new(require_intel: bool) -> Result<Self> {
        pollster::block_on(Self::init(require_intel))
    }
    async fn init(require_intel: bool) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: if require_intel {
                wgpu::Backends::VULKAN
            } else {
                wgpu::Backends::all()
            },
            ..Default::default()
        });
        let adapter = instance
            .enumerate_adapters(if require_intel {
                wgpu::Backends::VULKAN
            } else {
                wgpu::Backends::all()
            })
            .into_iter()
            .find(|a| {
                let i = a.get_info();
                i.device_type != wgpu::DeviceType::Cpu
                    && (!require_intel || (i.vendor == 0x8086 && i.device == 0x3185))
            })
            .context("required hardware GPU unavailable (software rendering is forbidden)")?;
        let info = adapter.get_info();
        let limits = adapter.limits();
        let max_buffer = limits.max_storage_buffer_binding_size as u64;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Timestral FP32"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                },
                None,
            )
            .await?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Timestral kernels"),
            source: wgpu::ShaderSource::Wgsl(include_str!("kernels.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Timestral"),
            layout: None,
            module: &shader,
            entry_point: "main",
        });
        Ok(Self {
            device,
            queue,
            pipeline,
            identity: format!(
                "{} {:?} vendor={:04x} device={:04x}",
                info.name, info.backend, info.vendor, info.device
            ),
            max_buffer,
        })
    }
    pub fn upload(&self, x: &[f32]) -> Tensor {
        let zero = [0.0f32];
        let values = if x.is_empty() { &zero[..] } else { x };
        Tensor {
            buffer: Arc::new(
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("tensor"),
                        contents: bytemuck::cast_slice(values),
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    }),
            ),
            len: x.len(),
        }
    }
    pub fn read(&self, t: &Tensor) -> Result<Vec<f32>> {
        let size = (t.len * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&t.buffer, 0, &staging, 0, size);
        self.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()??;
        let values = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
        staging.unmap();
        Ok(values)
    }
    pub fn op(
        &self,
        a: &Tensor,
        b: &Tensor,
        c: &Tensor,
        mut p: Params,
        count: usize,
    ) -> Result<Tensor> {
        anyhow::ensure!(
            count.div_ceil(64)
                <= self.device.limits().max_compute_workgroups_per_dimension as usize,
            "request exceeds GPU dispatch limit; reduce context or covariate count"
        );
        if count == 0 || count as u64 * 4 > self.max_buffer {
            bail!("tensor allocation exceeds GPU buffer limit");
        }
        p.unused = count.try_into()?;
        let out = Tensor {
            buffer: Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("result"),
                size: count as u64 * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })),
            len: count,
        };
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dimensions"),
                contents: bytemuck::bytes_of(&p),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let layout = self.pipeline.get_bind_group_layout(0);
        let bindings = [
            a.buffer.as_entire_binding(),
            b.buffer.as_entire_binding(),
            c.buffer.as_entire_binding(),
            out.buffer.as_entire_binding(),
            uniform.as_entire_binding(),
        ];
        let entries: Vec<_> = bindings
            .into_iter()
            .enumerate()
            .map(|(i, resource)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource,
            })
            .collect();
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Timestral"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
        Ok(out)
    }
    pub fn unary(&self, x: &Tensor, op: u32, width: usize) -> Result<Tensor> {
        let count = if op == 4 { x.len / 2 } else { x.len };
        self.op(
            x,
            x,
            x,
            Params {
                op,
                m: (count / width) as u32,
                n: width as u32,
                ..Default::default()
            },
            count,
        )
    }
    pub fn binary(&self, x: &Tensor, y: &Tensor, op: u32) -> Result<Tensor> {
        anyhow::ensure!(x.len == y.len, "binary shape mismatch");
        self.op(
            x,
            y,
            x,
            Params {
                op,
                m: 1,
                n: x.len as u32,
                ..Default::default()
            },
            x.len,
        )
    }
    pub fn linear(
        &self,
        x: &Tensor,
        w: &Tensor,
        bias: &Tensor,
        input: usize,
        output: usize,
    ) -> Result<Tensor> {
        anyhow::ensure!(
            x.len.is_multiple_of(input) && w.len == input * output && bias.len == output,
            "linear shape mismatch"
        );
        self.op(
            x,
            w,
            bias,
            Params {
                m: (x.len / input) as u32,
                n: output as u32,
                k: input as u32,
                ..Default::default()
            },
            x.len / input * output,
        )
    }
    pub fn norm(&self, x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
        self.op(
            x,
            w,
            x,
            Params {
                op: 1,
                m: (x.len / w.len) as u32,
                n: w.len as u32,
                eps,
                ..Default::default()
            },
            x.len,
        )
    }
}
