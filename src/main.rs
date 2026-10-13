//! rustman: a ray-traced room where light travels at a slow, adjustable speed.
//!
//! Controls:
//!   WASD / Q E     move, down / up        right mouse   look
//!   F              light on/off           L             flashlight / ceiling lamp
//!   M              lamp circles / stays
//!   Up / Down      c x1.25 / /1.25        I             infinite c (compare)
//!   B              ambient on/off         R             half / full resolution
//!   T              emission-time debug view
//!   V              Doppler color on/off   [ / ]         Doppler strength /1.5 / x1.5
//!   O              relativity on/off (aberration, beaming, length contraction)
//!   Space          pause                  H             hide / show the control panel
//!   Esc            quit

mod gpu;
mod sim;
mod ui;

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use glam::Vec3;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use gpu::{Gpu, GuiFrame, Params};
use sim::{HIST_CAP, LIGHT_RADIUS, Material, SIM_DT, Shape, World};

const INFINITE_C: f32 = 1.0e6;
const LAMP_INTENSITY: f32 = 34.0;
const FLASHLIGHT_INTENSITY: f32 = 80.0;
/// Doppler color strength: 1 is physical (most of the shift lands outside visible light at c = 8).
const DOPPLER_DEFAULT: f32 = 0.25;
/// Flashlight held a little right of and below the eye, so its shadows are visible.
const HAND_OFFSET: (f32, f32) = (0.25, -0.2);
const AMBIENT: f32 = 0.04;
const FOV_Y: f32 = 70.0;
const MOVE_SPEED: f32 = 3.0;
/// How fast the camera eases toward its target velocity, 1/s. A sudden jump in velocity
/// would snap the aberration.
const CAM_ACCEL: f32 = 8.0;
/// With relativity on, the camera can't reach c.
const MAX_CAM_BETA: f32 = 0.9;
const LOOK_SPEED: f32 = 0.003;

struct Camera {
    pos: Vec3,
    vel: Vec3,
    yaw: f32,
    pitch: f32,
}

impl Camera {
    fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(sy * cp, sp, -cy * cp)
    }
}

struct App {
    gpu: Option<Gpu>,
    world: World,
    cam: Camera,
    c: f32,
    infinite_c: bool,
    ambient: bool,
    /// Color by how long ago the light left each surface.
    debug_age: bool,
    /// Shift colors by the Doppler factor along each light path.
    doppler: bool,
    doppler_strength: f32,
    /// Special relativity: the camera's own aberration, Doppler and beaming, length
    /// contraction, and gamma terms in the Doppler factors. Off = classical light delay.
    relativity: bool,
    paused: bool,
    show_panel: bool,
    egui_ctx: egui::Context,
    /// Created with the window.
    egui_state: Option<egui_winit::State>,
    keys: HashSet<KeyCode>,
    looking: bool,
    last_frame: Instant,
    accumulator: f32,
    title_timer: f32,
    hist_scratch: Vec<[f32; 4]>,
    chunk_scratch: Vec<[f32; 4]>,
}

impl App {
    fn new() -> Self {
        App {
            gpu: None,
            world: World::new(),
            cam: Camera { pos: Vec3::new(0.0, 1.7, 8.25), vel: Vec3::ZERO, yaw: 0.0, pitch: -0.1 },
            c: 8.0,
            infinite_c: false,
            ambient: true,
            debug_age: false,
            doppler: true,
            doppler_strength: DOPPLER_DEFAULT,
            relativity: true,
            paused: false,
            show_panel: true,
            egui_ctx: egui::Context::default(),
            egui_state: None,
            keys: HashSet::new(),
            looking: false,
            last_frame: Instant::now(),
            accumulator: 0.0,
            title_timer: 0.0,
            hist_scratch: Vec::new(),
            chunk_scratch: Vec::new(),
        }
    }

    fn effective_c(&self) -> f32 {
        if self.infinite_c { INFINITE_C } else { self.c }
    }

    fn on_key(&mut self, key: KeyCode, event_loop: &ActiveEventLoop) {
        match key {
            KeyCode::Escape => event_loop.exit(),
            KeyCode::KeyF => self.world.light.on = !self.world.light.on,
            KeyCode::KeyM => self.world.light.moving = !self.world.light.moving,
            KeyCode::KeyL => self.toggle_flashlight(),
            KeyCode::ArrowUp => self.c = (self.c * 1.25).min(1000.0),
            KeyCode::ArrowDown => self.c = (self.c / 1.25).max(1.0),
            KeyCode::KeyI => self.infinite_c = !self.infinite_c,
            KeyCode::KeyB => self.ambient = !self.ambient,
            KeyCode::KeyT => self.debug_age = !self.debug_age,
            KeyCode::KeyV => self.doppler = !self.doppler,
            KeyCode::KeyO => self.relativity = !self.relativity,
            KeyCode::BracketLeft => self.doppler_strength = (self.doppler_strength / 1.5).max(0.02),
            KeyCode::BracketRight => self.doppler_strength = (self.doppler_strength * 1.5).min(4.0),
            KeyCode::Space => self.paused = !self.paused,
            KeyCode::KeyH => self.show_panel = !self.show_panel,
            KeyCode::KeyR => {
                if let Some(gpu) = &mut self.gpu {
                    let scale = if gpu.render_scale < 1.0 { 1.0 } else { 0.5 };
                    gpu.set_render_scale(scale);
                }
            }
            _ => {}
        }
        self.title_timer = f32::MAX;
    }

    fn toggle_flashlight(&mut self) {
        // Switch with the light off: the jump between positions is faster than c.
        self.world.light.flashlight = !self.world.light.flashlight;
        self.world.light.on = false;
    }

    /// Run the control panel for this frame and tessellate it for drawing.
    fn run_gui(&mut self) -> Option<GuiFrame> {
        let window = self.gpu.as_ref()?.window.clone();
        let raw = self.egui_state.as_mut()?.take_egui_input(&window);
        let ctx = self.egui_ctx.clone();
        let out = ctx.run_ui(raw, |ui| self.panel(ui.ctx()));
        self.egui_state.as_mut()?.handle_platform_output(&window, out.platform_output);
        Some(GuiFrame {
            primitives: ctx.tessellate(out.shapes, out.pixels_per_point),
            textures: out.textures_delta,
            pixels_per_point: out.pixels_per_point,
        })
    }

    fn update(&mut self, dt: f32) {
        let fwd = self.cam.forward();
        let flat = Vec3::new(fwd.x, 0.0, fwd.z).normalize_or_zero();
        let right = flat.cross(Vec3::Y);
        let mut mv = Vec3::ZERO;
        for (key, dir) in [
            (KeyCode::KeyW, flat),
            (KeyCode::KeyS, -flat),
            (KeyCode::KeyD, right),
            (KeyCode::KeyA, -right),
            (KeyCode::KeyE, Vec3::Y),
            (KeyCode::KeyQ, -Vec3::Y),
        ] {
            if self.keys.contains(&key) {
                mv += dir;
            }
        }
        let mut target = mv.normalize_or_zero() * MOVE_SPEED;
        if self.relativity {
            target = target.clamp_length_max(MAX_CAM_BETA * self.effective_c());
        }
        self.cam.vel += (target - self.cam.vel) * (1.0 - (-CAM_ACCEL * dt).exp());
        let margin = Vec3::splat(0.2);
        let next = self.cam.pos + self.cam.vel * dt;
        self.cam.pos = next.clamp(sim::ROOM_MIN + margin, sim::ROOM_MAX - margin);
        // Stop dead against a wall, or the view would keep its aberration while standing still.
        for i in 0..3 {
            if self.cam.pos[i] != next[i] {
                self.cam.vel[i] = 0.0;
            }
        }

        let cam_right = fwd.cross(Vec3::Y).normalize();
        let hand = self.cam.pos + cam_right * HAND_OFFSET.0 + Vec3::Y * HAND_OFFSET.1;

        if !self.paused {
            // Fixed-step sim: history samples must be evenly spaced.
            self.accumulator = (self.accumulator + dt).min(0.25);
            while self.accumulator >= SIM_DT {
                // Where the hand was when this tick ends, not where it is now. Otherwise the
                // flashlight moves in per-frame jerks, and its Doppler shows them as stripes.
                let behind = self.accumulator - SIM_DT;
                self.world.light.hand = (hand - self.cam.vel * behind, fwd);
                self.world.step();
                self.accumulator -= SIM_DT;
            }
        }
    }

    fn params(&self, aspect: f32) -> Params {
        let fwd = self.cam.forward();
        let right = fwd.cross(Vec3::Y).normalize();
        let up = right.cross(fwd);
        let tan_half = (FOV_Y.to_radians() * 0.5).tan();
        let r = right * tan_half * aspect;
        let u = up * tan_half;
        let p = self.cam.pos;
        let v = self.cam.vel;
        Params {
            cam_pos: [p.x, p.y, p.z, self.effective_c()],
            cam_right: [r.x, r.y, r.z, 0.0],
            cam_up: [u.x, u.y, u.z, 0.0],
            cam_fwd: [fwd.x, fwd.y, fwd.z, 0.0],
            misc: [SIM_DT, if self.world.light.flashlight { FLASHLIGHT_INTENSITY } else { LAMP_INTENSITY }, if self.ambient { AMBIENT } else { 0.0 }, if self.debug_age { 1.0 } else { 0.0 }],
            counts: [self.world.bodies.len() as u32, self.world.history_len() as u32, HIST_CAP as u32, 0],
            fx: [if self.doppler { self.doppler_strength } else { 0.0 }, 0.0, 0.0, 0.0],
            obs: [v.x, v.y, v.z, if self.relativity { 1.0 } else { 0.0 }],
        }
    }

    fn objects(&self) -> Vec<[f32; 4]> {
        let mut objs = Vec::new();
        for b in &self.world.bodies {
            let (col, kind) = match b.material {
                Material::Diffuse(c) => (c, 0.0),
                Material::Mirror(c) => (c, 1.0),
            };
            objs.push([col.x, col.y, col.z, kind]);
            objs.push(match b.shape {
                Shape::Sphere(r) => [r, 0.0, 0.0, 0.0],
                Shape::Box(h) => [h.x, h.y, h.z, 1.0],
            });
        }
        objs.push([1.0, 1.0, 1.0, 2.0]);
        // The flashlight is in your hand; don't draw its bulb in your face.
        let bulb = if self.world.light.flashlight { 0.0 } else { LIGHT_RADIUS };
        objs.push([bulb, 0.0, 0.0, 0.0]);
        objs
    }

    fn update_title(&mut self) {
        let Some(gpu) = &self.gpu else { return };
        let c = if self.infinite_c { "inf".to_string() } else { format!("{:.1}", self.c) };
        let warn = if !self.infinite_c && self.world.max_speed > self.c { "  [objects faster than c!]" } else { "" };
        gpu.window.set_title(&format!(
            "rustman  |  c = {c} m/s  |  {} {}{}  |  doppler {}  |  {}  |  t = {:.1}s{}{}{}",
            if self.world.light.flashlight { "flashlight" } else { "lamp" },
            if self.world.light.on { "ON" } else { "off" },
            if self.world.light.moving && !self.world.light.flashlight { " (moving)" } else { "" },
            if self.doppler { format!("x{:.2}", self.doppler_strength) } else { "off".to_string() },
            if self.relativity { "relativistic" } else { "classical" },
            self.world.time,
            if self.paused { "  PAUSED" } else { "" },
            if self.debug_age { "  [emission-time view]" } else { "" },
            warn,
        ));
    }

    fn frame(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;
        self.update(dt);

        self.title_timer += dt;
        if self.title_timer > 0.25 {
            self.title_timer = 0.0;
            self.update_title();
        }

        let Some(gui) = self.run_gui() else { return };
        let objs = self.objects();
        self.world.history_gpu(&mut self.hist_scratch);
        self.world.chunks_gpu(&mut self.chunk_scratch);
        let Some(aspect) = self.gpu.as_ref().map(Gpu::aspect) else { return };
        let params = self.params(aspect);
        let Some(gpu) = &mut self.gpu else { return };
        gpu.hot_reload();
        gpu.upload(&params, &objs, &self.hist_scratch, &self.chunk_scratch);
        gpu.render(gui);
        gpu.window.request_redraw();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("rustman")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        self.egui_state = Some(egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            window.theme(),
            None,
        ));
        self.gpu = Some(Gpu::new(window, self.world.bodies.len() + 1));
        self.last_frame = Instant::now();
        self.update_title();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // The panel gets first look; clicks and typing it uses don't reach the camera or hotkeys.
        let mut consumed = false;
        if let (Some(gpu), Some(state)) = (&self.gpu, &mut self.egui_state) {
            consumed = state.on_window_event(&gpu.window, &event).consumed;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.resize(size.width, size.height);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let PhysicalKey::Code(key) = event.physical_key else { return };
                match event.state {
                    ElementState::Pressed if consumed => {}
                    ElementState::Pressed => {
                        if !event.repeat {
                            self.on_key(key, event_loop);
                        }
                        self.keys.insert(key);
                    }
                    ElementState::Released => {
                        self.keys.remove(&key);
                    }
                }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Right, .. } => {
                self.looking = state == ElementState::Pressed && !consumed;
            }
            WindowEvent::RedrawRequested => self.frame(),
            _ => {}
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if self.looking {
                self.cam.yaw += dx as f32 * LOOK_SPEED;
                self.cam.pitch = (self.cam.pitch - dy as f32 * LOOK_SPEED).clamp(-1.5, 1.5);
            }
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("run app");
}
