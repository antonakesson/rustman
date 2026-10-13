// Fullscreen triangle that scales the traced image onto the swapchain.

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
    let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    var out: VOut;
    out.pos = vec4f(uv * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), 0.0, 1.0);
    out.uv = uv;
    return out;
}

@fragment
fn fs(v: VOut) -> @location(0) vec4f {
    return textureSample(tex, samp, v.uv);
}
