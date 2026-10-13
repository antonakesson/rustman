//! wgpu setup: one compute pass traces into an offscreen texture, one render pass
//! scales it onto the window and draws the egui overlay on top.
//! trace.wgsl is hot-reloaded when the file changes.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;

use winit::window::Window;

use crate::sim::{CHUNK, HIST_CAP};

const TRACE_FALLBACK: &str = include_str!("../shaders/trace.wgsl");
const BLIT_SRC: &str = include_str!("../shaders/blit.wgsl");
const OUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Uniform block, laid out as in trace.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Params {
    pub cam_pos: [f32; 4],
    pub cam_right: [f32; 4],
    pub cam_up: [f32; 4],
    pub cam_fwd: [f32; 4],
    pub misc: [f32; 4],
    pub counts: [u32; 4],
    pub fx: [f32; 4],
    pub obs: [f32; 4],
}

/// One frame of egui output, ready to draw.
pub struct GuiFrame {
    pub primitives: Vec<egui::ClippedPrimitive>,
    pub textures: egui::TexturesDelta,
    pub pixels_per_point: f32,
}

fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: only used with padding-free plain-old-data (f32/u32 arrays).
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}

pub struct Gpu {
    pub window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    trace_layout: wgpu::BindGroupLayout,
    trace_pipeline_layout: wgpu::PipelineLayout,
    trace_pipeline: wgpu::ComputePipeline,
    trace_bind: wgpu::BindGroup,
    blit_layout: wgpu::BindGroupLayout,
    blit_pipeline: wgpu::RenderPipeline,
    blit_bind: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    egui: egui_wgpu::Renderer,

    params_buf: wgpu::Buffer,
    objs_buf: wgpu::Buffer,
    hist_buf: wgpu::Buffer,
    chunks_buf: wgpu::Buffer,
    out_size: (u32, u32),
    pub render_scale: f32,

    shader_path: PathBuf,
    shader_mtime: Option<SystemTime>,
}

impl Gpu {
    /// `objects` = bodies + the light.
    pub fn new(window: Arc<Window>, objects: usize) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance.create_surface(window.clone()).expect("create surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .expect("no suitable GPU adapter");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).expect("request device");

        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("surface not supported by adapter");
        // Trace writes linear color; let an sRGB swapchain do the encoding.
        let caps = surface.get_capabilities(&adapter);
        if let Some(f) = caps.formats.iter().find(|f| f.is_srgb()) {
            config.format = *f;
        }
        config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(&device, &config);

        let storage = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let trace_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("trace"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1),
                storage(2),
                storage(3),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: OUT_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let trace_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("trace"),
            bind_group_layouts: &[Some(&trace_layout)],
            ..Default::default()
        });

        let shader_path = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/trace.wgsl"));
        let shader_mtime = std::fs::metadata(&shader_path).and_then(|m| m.modified()).ok();
        let src = std::fs::read_to_string(&shader_path).unwrap_or_else(|_| TRACE_FALLBACK.to_string());
        let trace_pipeline =
            create_trace_pipeline(&device, &trace_pipeline_layout, &src).expect("trace.wgsl failed to compile");

        let blit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let blit_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SRC.into()),
        });
        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit"),
            bind_group_layouts: &[Some(&blit_layout)],
            ..Default::default()
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_module,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_module,
                entry_point: Some("fs"),
                targets: &[Some(config.format.into())],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Picks its sRGB or linear output path from the swapchain format.
        let egui = egui_wgpu::Renderer::new(&device, config.format, Default::default());

        let buffer = |label, size: usize, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let params_buf = buffer("params", size_of::<Params>(), wgpu::BufferUsages::UNIFORM);
        let objs_buf = buffer("objs", objects * 32, wgpu::BufferUsages::STORAGE);
        let hist_buf = buffer("hist", objects * HIST_CAP * 32, wgpu::BufferUsages::STORAGE);
        let chunks_buf = buffer("chunks", objects * HIST_CAP / CHUNK * 16, wgpu::BufferUsages::STORAGE);

        // Bind groups are built by resize(); start with placeholders sized 1x1.
        let (trace_bind, blit_bind) = Self::make_targets(
            &device,
            &trace_layout,
            &blit_layout,
            &sampler,
            [&params_buf, &objs_buf, &hist_buf, &chunks_buf],
            (1, 1),
        );

        let mut gpu = Gpu {
            window,
            surface,
            device,
            queue,
            config,
            trace_layout,
            trace_pipeline_layout,
            trace_pipeline,
            trace_bind,
            blit_layout,
            blit_pipeline,
            blit_bind,
            sampler,
            egui,
            params_buf,
            objs_buf,
            hist_buf,
            chunks_buf,
            out_size: (1, 1),
            render_scale: 1.0,
            shader_path,
            shader_mtime,
        };
        gpu.resize(size.width, size.height);
        gpu
    }

    fn make_targets(
        device: &wgpu::Device,
        trace_layout: &wgpu::BindGroupLayout,
        blit_layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        bufs: [&wgpu::Buffer; 4],
        (w, h): (u32, u32),
    ) -> (wgpu::BindGroup, wgpu::BindGroup) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("traced"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: OUT_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        let mut entries: Vec<wgpu::BindGroupEntry> = bufs
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
            .collect();
        entries.push(wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&view) });
        let trace = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trace"),
            layout: trace_layout,
            entries: &entries,
        });
        let blit = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit"),
            layout: blit_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
            ],
        });
        (trace, blit)
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        let out = (
            ((width as f32 * self.render_scale) as u32).max(1),
            ((height as f32 * self.render_scale) as u32).max(1),
        );
        let (trace, blit) = Self::make_targets(
            &self.device,
            &self.trace_layout,
            &self.blit_layout,
            &self.sampler,
            [&self.params_buf, &self.objs_buf, &self.hist_buf, &self.chunks_buf],
            out,
        );
        self.trace_bind = trace;
        self.blit_bind = blit;
        self.out_size = out;
    }

    pub fn aspect(&self) -> f32 {
        self.config.width as f32 / self.config.height as f32
    }

    pub fn set_render_scale(&mut self, scale: f32) {
        self.render_scale = scale;
        self.resize(self.config.width, self.config.height);
    }

    /// Recompile trace.wgsl if it changed on disk. Errors are printed and the old pipeline kept.
    pub fn hot_reload(&mut self) {
        let mtime = std::fs::metadata(&self.shader_path).and_then(|m| m.modified()).ok();
        if mtime.is_none() || mtime == self.shader_mtime {
            return;
        }
        self.shader_mtime = mtime;
        let Ok(src) = std::fs::read_to_string(&self.shader_path) else { return };
        match create_trace_pipeline(&self.device, &self.trace_pipeline_layout, &src) {
            Ok(p) => {
                self.trace_pipeline = p;
                println!("trace.wgsl reloaded");
            }
            Err(e) => eprintln!("trace.wgsl reload failed:\n{e}"),
        }
    }

    pub fn upload(&self, params: &Params, objs: &[[f32; 4]], hist: &[[f32; 4]], chunks: &[[f32; 4]]) {
        self.queue.write_buffer(&self.params_buf, 0, as_bytes(std::slice::from_ref(params)));
        self.queue.write_buffer(&self.objs_buf, 0, as_bytes(objs));
        self.queue.write_buffer(&self.hist_buf, 0, as_bytes(hist));
        self.queue.write_buffer(&self.chunks_buf, 0, as_bytes(chunks));
    }

    pub fn render(&mut self, mut gui: GuiFrame) {
        // Upload egui's font/texture changes even if this frame is skipped.
        for (id, deltas) in &gui.textures.set {
            for delta in deltas {
                self.egui.update_texture(&self.device, &self.queue, *id, delta);
            }
        }
        self.present(&gui);
        for id in &gui.textures.free {
            self.egui.free_texture(id);
        }
        gui.textures.clear(); // marks the deltas applied
    }

    fn present(&mut self, gui: &GuiFrame) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            _ => return,
        };
        let view = frame.texture.create_view(&Default::default());
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.trace_pipeline);
            pass.set_bind_group(0, &self.trace_bind, &[]);
            pass.dispatch_workgroups(self.out_size.0.div_ceil(8), self.out_size.1.div_ceil(8), 1);
        }
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: gui.pixels_per_point,
        };
        let gui_cmds = self.egui.update_buffers(&self.device, &self.queue, &mut enc, &gui.primitives, &screen);
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    depth_slice: None,
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.blit_pipeline);
            pass.set_bind_group(0, &self.blit_bind, &[]);
            pass.draw(0..3, 0..1);
            self.egui.render(&mut pass.forget_lifetime(), &gui.primitives, &screen);
        }
        self.queue.submit(gui_cmds.into_iter().chain([enc.finish()]));
        self.window.pre_present_notify();
        self.queue.present(frame);
    }
}

fn create_trace_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    src: &str,
) -> Result<wgpu::ComputePipeline, String> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("trace"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("trace"),
        layout: Some(layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    match pollster::block_on(scope.pop()) {
        Some(e) => Err(e.to_string()),
        None => Ok(pipeline),
    }
}
