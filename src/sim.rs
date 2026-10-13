//! CPU-side world: moving, spinning spheres and boxes, a point light, and the
//! per-object history that the ray tracer queries at retarded times.

use glam::{Quat, Vec3};

/// Simulation tick. One history sample is recorded per tick.
pub const SIM_DT: f32 = 1.0 / 60.0;
/// History samples kept per object (~17 s at 60 Hz).
pub const HIST_CAP: usize = 1024;
/// Samples per bounding chunk; the shader skips whole chunks the ray can't touch.
pub const CHUNK: usize = 16;

/// Must match ROOM_MIN / ROOM_MAX in trace.wgsl.
pub const ROOM_MIN: Vec3 = Vec3::new(-9.0, 0.0, -9.0);
pub const ROOM_MAX: Vec3 = Vec3::new(9.0, 6.0, 9.0);

const GRAVITY: f32 = 3.0;

#[derive(Clone, Copy)]
pub enum Material {
    Diffuse(Vec3),
    Mirror(Vec3),
}

#[derive(Clone, Copy)]
pub enum Shape {
    Sphere(f32),
    /// Half-extents.
    Box(Vec3),
}

impl Shape {
    /// Bounding-sphere radius around the center.
    pub fn bound(&self) -> f32 {
        match *self {
            Shape::Sphere(r) => r,
            Shape::Box(h) => h.length(),
        }
    }
}

enum Motion {
    Still,
    Ballistic,
    Orbit { center: Vec3, radius: f32, speed: f32, phase: f32 },
    Oscillate { center: Vec3, axis: Vec3, amp: f32, omega: f32 },
}

pub struct Body {
    pub shape: Shape,
    pub material: Material,
    pos: Vec3,
    vel: Vec3,
    motion: Motion,
    rot: Quat,
    /// Angular velocity: axis * rad/s, in world space.
    spin: Vec3,
}

pub struct Light {
    pub on: bool,
    pub moving: bool,
    /// Flashlight (cone, held by the camera) instead of the ceiling lamp.
    pub flashlight: bool,
    /// Flashlight position and aim, set by the app before each step.
    pub hand: (Vec3, Vec3),
    pos: Vec3,
    dir: Vec3,
    angle: f32,
    orbit_radius: f32,
}

pub const LIGHT_RADIUS: f32 = 0.08;
const LIGHT_HOME: Vec3 = Vec3::new(0.0, 5.4, 0.0);
const LIGHT_ORBIT: f32 = 6.0;
const LIGHT_SPEED: f32 = 2.0;
/// cos of the flashlight cone's half-angle (18 degrees).
const FLASHLIGHT_COS: f32 = 0.951;

pub struct World {
    pub bodies: Vec<Body>,
    pub light: Light,
    pub time: f32,
    /// Fastest body speed seen in the last tick, to warn when it exceeds c.
    pub max_speed: f32,
    history: History,
}

impl World {
    pub fn new() -> Self {
        let bodies = vec![
            Body {
                shape: Shape::Sphere(0.5),
                material: Material::Diffuse(Vec3::new(0.9, 0.15, 0.1)),
                pos: Vec3::new(-4.5, 4.5, -3.0),
                vel: Vec3::new(2.2, 0.0, 1.4),
                motion: Motion::Ballistic,
                rot: Quat::IDENTITY,
                spin: Vec3::new(1.0, 2.5, 0.0),
            },
            Body {
                shape: Shape::Sphere(0.8),
                material: Material::Mirror(Vec3::splat(0.9)),
                pos: Vec3::ZERO,
                vel: Vec3::ZERO,
                motion: Motion::Orbit { center: Vec3::new(0.0, 1.0, -1.5), radius: 4.5, speed: 2.0, phase: 0.0 },
                rot: Quat::IDENTITY,
                spin: Vec3::ZERO,
            },
            Body {
                shape: Shape::Sphere(0.45),
                material: Material::Diffuse(Vec3::new(0.15, 0.3, 0.95)),
                pos: Vec3::ZERO,
                vel: Vec3::ZERO,
                motion: Motion::Oscillate {
                    center: Vec3::new(0.0, 0.6, 3.75),
                    axis: Vec3::X,
                    amp: 6.75,
                    omega: 0.5,
                },
                rot: Quat::IDENTITY,
                spin: Vec3::new(0.0, 1.5, 0.0),
            },
            Body {
                shape: Shape::Sphere(0.4),
                material: Material::Diffuse(Vec3::new(0.2, 0.85, 0.25)),
                pos: Vec3::new(5.25, 4.8, 4.5),
                vel: Vec3::ZERO,
                motion: Motion::Ballistic,
                rot: Quat::IDENTITY,
                spin: Vec3::new(0.0, 0.0, 2.0),
            },
            Body {
                shape: Shape::Sphere(0.3),
                material: Material::Diffuse(Vec3::new(0.95, 0.8, 0.1)),
                pos: Vec3::ZERO,
                vel: Vec3::ZERO,
                motion: Motion::Orbit { center: Vec3::new(0.0, 4.5, -1.5), radius: 6.75, speed: 3.0, phase: 2.0 },
                rot: Quat::IDENTITY,
                spin: Vec3::new(0.0, 3.0, 0.0),
            },
            // Spinner: the equator moves at 3.6 m/s, about 0.45c at the default c = 8.
            // The grid shears where the surface moves toward / away from the camera.
            Body {
                shape: Shape::Sphere(0.9),
                material: Material::Diffuse(Vec3::new(0.7, 0.3, 0.9)),
                pos: Vec3::new(-7.5, 2.2, -7.5),
                vel: Vec3::ZERO,
                motion: Motion::Still,
                rot: Quat::IDENTITY,
                spin: Vec3::new(0.25, 1.0, 0.15).normalize() * 4.0,
            },
            // Fast cube: peaks at 6.4 m/s (0.8c at c = 8) right in front of the camera.
            // Straight edges bend, and you see its back face as it passes.
            Body {
                shape: Shape::Box(Vec3::splat(0.6)),
                material: Material::Diffuse(Vec3::splat(0.9)),
                pos: Vec3::ZERO,
                vel: Vec3::ZERO,
                motion: Motion::Oscillate {
                    center: Vec3::new(0.0, 1.2, 5.8),
                    axis: Vec3::X,
                    amp: 7.5,
                    omega: 0.85,
                },
                rot: Quat::IDENTITY,
                spin: Vec3::ZERO,
            },
            // Tumbling brick bouncing around the room.
            Body {
                shape: Shape::Box(Vec3::new(0.7, 0.35, 0.5)),
                material: Material::Diffuse(Vec3::new(0.95, 0.85, 0.7)),
                pos: Vec3::new(-3.0, 3.0, 0.0),
                vel: Vec3::new(1.5, 0.0, -2.0),
                motion: Motion::Ballistic,
                rot: Quat::IDENTITY,
                spin: Vec3::new(1.2, 0.8, 0.4),
            },
        ];
        // Bodies plus the light.
        let objects = bodies.len() + 1;
        let mut world = World {
            bodies,
            light: Light {
                on: false,
                moving: false,
                flashlight: true,
                hand: (Vec3::ZERO, Vec3::NEG_Z),
                pos: LIGHT_HOME,
                dir: Vec3::NEG_Y,
                angle: 0.0,
                orbit_radius: 0.0,
            },
            time: 0.0,
            max_speed: 0.0,
            history: History::new(objects),
        };
        world.step_bodies(0.0);
        world.record();
        world
    }

    pub fn step(&mut self) {
        self.time += SIM_DT;
        self.step_bodies(SIM_DT);
        let l = &mut self.light;
        // The lamp eases in/out of its orbit; a jump would move the light faster than c.
        let target = if l.moving { LIGHT_ORBIT } else { 0.0 };
        l.orbit_radius += (target - l.orbit_radius).clamp(-LIGHT_SPEED * SIM_DT, LIGHT_SPEED * SIM_DT);
        if l.moving {
            l.angle += LIGHT_SPEED / LIGHT_ORBIT * SIM_DT;
        }
        if l.flashlight {
            (l.pos, l.dir) = l.hand;
        } else {
            l.pos = LIGHT_HOME + Vec3::new(l.angle.cos(), 0.0, l.angle.sin()) * l.orbit_radius;
            l.dir = Vec3::NEG_Y;
        }
        self.record();
    }

    fn step_bodies(&mut self, dt: f32) {
        let t = self.time;
        self.max_speed = 0.0;
        for b in &mut self.bodies {
            let prev = b.pos;
            match b.motion {
                Motion::Still => {}
                Motion::Ballistic => {
                    b.vel.y -= GRAVITY * dt;
                    b.pos += b.vel * dt;
                    // Bounce on the bounding sphere: boxes stop a little short of the walls.
                    let lo = ROOM_MIN + Vec3::splat(b.shape.bound());
                    let hi = ROOM_MAX - Vec3::splat(b.shape.bound());
                    for i in 0..3 {
                        if b.pos[i] < lo[i] {
                            b.pos[i] = lo[i];
                            b.vel[i] = b.vel[i].abs();
                        } else if b.pos[i] > hi[i] {
                            b.pos[i] = hi[i];
                            b.vel[i] = -b.vel[i].abs();
                        }
                    }
                }
                Motion::Orbit { center, radius, speed, phase } => {
                    let a = phase + speed / radius * t;
                    b.pos = center + Vec3::new(a.cos(), 0.0, a.sin()) * radius;
                }
                Motion::Oscillate { center, axis, amp, omega } => {
                    b.pos = center + axis * amp * (omega * t).sin();
                }
            }
            b.rot = (Quat::from_scaled_axis(b.spin * dt) * b.rot).normalize();
            if dt > 0.0 {
                // Fastest surface point (upper bound): translation plus spin at the bounding radius.
                let speed = (b.pos - prev).length() / dt + b.spin.length() * b.shape.bound();
                self.max_speed = self.max_speed.max(speed);
            }
        }
    }

    fn record(&mut self) {
        let mut row: Vec<Sample> =
            self.bodies.iter().map(|b| [[b.pos.x, b.pos.y, b.pos.z, 0.0], b.rot.to_array()]).collect();
        let (l, d) = (self.light.pos, self.light.dir);
        row.push([
            [l.x, l.y, l.z, if self.light.on { 1.0 } else { 0.0 }],
            [d.x, d.y, d.z, if self.light.flashlight { FLASHLIGHT_COS } else { 0.0 }],
        ]);
        self.history.push(&row);
    }

    pub fn history_len(&self) -> usize {
        self.history.len
    }

    /// History laid out for the GPU: per object, newest sample first, two vec4 per sample.
    pub fn history_gpu(&self, out: &mut Vec<[f32; 4]>) {
        self.history.linearize(out);
    }

    /// One bounding sphere per CHUNK segments per object, covering the object's sweep.
    pub fn chunks_gpu(&self, out: &mut Vec<[f32; 4]>) {
        let radii: Vec<f32> = self.bodies.iter().map(|b| b.shape.bound()).chain([LIGHT_RADIUS]).collect();
        self.history.chunk_bounds(&radii, out);
    }
}

/// One object at one tick: [position, payload] and [rotation quaternion]
/// (for the light: [position, on] and [direction, cone]).
type Sample = [[f32; 4]; 2];

/// Fixed-size ring buffer of samples for every object (bodies + light).
struct History {
    objects: usize,
    data: Vec<Sample>,
    head: usize,
    len: usize,
}

impl History {
    fn new(objects: usize) -> Self {
        History { objects, data: vec![[[0.0; 4]; 2]; objects * HIST_CAP], head: HIST_CAP - 1, len: 0 }
    }

    fn push(&mut self, row: &[Sample]) {
        self.head = (self.head + 1) % HIST_CAP;
        for (r, s) in row.iter().enumerate() {
            self.data[r * HIST_CAP + self.head] = *s;
        }
        self.len = (self.len + 1).min(HIST_CAP);
    }

    /// k = 0 is the newest sample.
    fn get(&self, obj: usize, k: usize) -> Sample {
        self.data[obj * HIST_CAP + (self.head + HIST_CAP - k) % HIST_CAP]
    }

    fn linearize(&self, out: &mut Vec<[f32; 4]>) {
        out.clear();
        for r in 0..self.objects {
            for k in 0..HIST_CAP {
                out.extend(if k < self.len { self.get(r, k) } else { [[0.0; 4]; 2] });
            }
        }
    }

    fn chunk_bounds(&self, radii: &[f32], out: &mut Vec<[f32; 4]>) {
        out.clear();
        for (r, &radius) in radii.iter().enumerate() {
            for j in 0..HIST_CAP / CHUNK {
                let first = j * CHUNK;
                if first + 1 >= self.len {
                    out.push([0.0, 0.0, 0.0, -1.0]);
                    continue;
                }
                let last = (first + CHUNK).min(self.len - 1);
                let mut lo = Vec3::splat(f32::MAX);
                let mut hi = Vec3::splat(f32::MIN);
                for k in first..=last {
                    let p = Vec3::from_slice(&self.get(r, k)[0][..3]);
                    lo = lo.min(p);
                    hi = hi.max(p);
                }
                let c = (lo + hi) * 0.5;
                out.push([c.x, c.y, c.z, (hi - lo).length() * 0.5 + radius]);
            }
        }
    }
}
