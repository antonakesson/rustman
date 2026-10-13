# rustman

A real-time GPU ray tracer where light travels at a slow, adjustable speed.

![rustman demo](docs/demo.gif)

Light moves at 8 m/s by default, so what you see is delayed: every surface shows up where it *was* when the light left it. Moving objects lag behind and stretch, reflections show the past, and the scene ahead of you is older than the scene next to you. Push `c` up and you're back to normal rendering. Pull it down and things fall apart.

## How it works

- **Rays are null lines in spacetime.** A ray from the camera is traced backwards in both space and time: `x(s) = o + d·s`, `t(s) = t0 − s/c`. A hit is where the ray meets an object at the moment the light left it.
- **Objects are world tubes.** The CPU simulates spinning, orbiting and bouncing spheres and boxes at 60 Hz and keeps ~17 s of position and rotation history per object. The shader intersects rays with each object's state at the *retarded time*.
- **Occlusion comes for free.** The nearest hit along the ray is also the latest in time, so ordinary nearest-hit tracing gives the correct answer.
- **Chunked history.** History is split into 16-sample chunks with bounding spheres, so the shader skips whole time ranges a ray can't reach.
- **Doppler shift.** Light paths are color-shifted by the relative motion of source and receiver. Physically correct strength puts most of the shift outside visible light at `c = 8`, so the default is turned down to 0.25 and adjustable.
- **Special relativity for the camera too.** Walking boosts each camera ray from your moving frame into the room's frame. The view crowds into a tunnel ahead of you (aberration), turns bluer and brighter ahead and redder and dimmer behind (your own Doppler shift, plus radiance scaling as D⁴). Moving objects are length-contracted along their velocity, and every Doppler factor carries its γ term. Press `O` to compare against classical light delay.
- Up to 4 bounces with mirrors, a moving lamp or a flashlight with shadows, plus an emission-time debug view that colors each pixel by how long ago its light was emitted.

Built with [wgpu](https://wgpu.rs) (one compute pass for tracing, one render pass to blit and draw the [egui](https://github.com/emilk/egui) overlay) and [winit](https://github.com/rust-windowing/winit). The tracer is a single WGSL shader, `shaders/trace.wgsl`, and it hot-reloads when you save it.

## How real is it?

The *rendering* follows special relativity: what you see is what light would actually show you. The *world* being rendered is simpler than that. Here's an honest breakdown.

**Physically correct**

- Light delay. Every pixel shows its surface at the retarded time, through mirrors and shadows as well.
- Aberration, Doppler shift and beaming from your own motion, using the exact Lorentz transformation for your current velocity. What you see at a given instant depends only on your velocity then, not your acceleration, so this is correct even while you speed up and slow down.
- Relativistic Doppler factors, γ terms included, at both ends of every light path: light → surface → mirrors → camera.
- Length contraction of moving objects along their velocity.

**Simplified or artistic**

- **The objects move by Newtonian rules.** Orbits, bounces and gravity are computed classically, and nothing stops an object from moving faster than light if you lower `c`. The panel warns you when that happens, and γ is capped at 10 so those objects still render.
- **Spin is rigid.** Rotating objects are contracted by their center's velocity, not point by point.
- **Doppler color is scaled down.** At full physical strength most shifts land in infrared or ultraviolet, so the default shows a quarter of the shift. Set the strength slider in the panel to 1 for the physical amount.
- **Ambient fill light is instant.** It only keeps the room visible in the dark, and `B` turns it off.
- **Your own clock doesn't slow down** while you move.
- Atoms couldn't exist if light were this slow. Every slow-light thought experiment ignores this, and so does rustman.

### Is it 4D?

In the spacetime sense, yes. Rays are traced through 3+1-dimensional flat (Minkowski) spacetime. Each ray is a light-like line through space *and* time, and each object is a *world tube*: the 4D shape it sweeps out over its recorded history. Every hit test is a 4D intersection between the two. It isn't a fourth *spatial* dimension, and spacetime isn't curved. Curved spacetime would mean gravity, which is general relativity.

## Running

Needs a recent stable Rust toolchain (edition 2024) and a GPU with Vulkan, DirectX 12 or Metal.

```sh
cargo run --release
```

## Controls

| Key | Action |
| --- | --- |
| `W` `A` `S` `D` / `Q` `E` | Move / down, up |
| Right mouse | Look |
| `Up` / `Down` | Speed of light ×1.25 / ÷1.25 |
| `I` | Infinite `c` (compare against normal rendering) |
| `F` | Light on / off |
| `L` | Flashlight / ceiling lamp |
| `M` | Lamp circles / stays still |
| `V` | Doppler color on / off |
| `[` / `]` | Doppler strength ÷1.5 / ×1.5 |
| `O` | Relativity on / off (aberration, beaming, length contraction) |
| `T` | Emission-time debug view |
| `B` | Ambient light on / off |
| `R` | Half / full resolution |
| `Space` | Pause |
| `H` | Hide / show the control panel |
| `Esc` | Quit |

## License

[MIT](LICENSE)
