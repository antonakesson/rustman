// Finite-light-speed ray tracer.
//
// Every ray is a null line in spacetime: x(s) = o + d*s, t(s) = t0 - s/c.
// Objects are world tubes: spinning spheres and boxes swept along their recorded history.
// Nearest-hit in s is automatically latest-in-t, so occlusion needs no special care.
// Times are relative to the observer: t = 0 is "now", history sample k is at -k*dt.

struct Params {
    cam_pos: vec4f,   // w = c
    cam_right: vec4f, // pre-scaled by tan(fov/2) * aspect
    cam_up: vec4f,    // pre-scaled by tan(fov/2)
    cam_fwd: vec4f,
    misc: vec4f,      // x = history dt, y = light intensity, z = ambient, w = debug view (1 = emission time)
    counts: vec4u,    // x = bodies (the light is object x), y = valid samples, z = capacity
    fx: vec4f,        // x = Doppler color strength (0 = off, 1 = physical)
}

struct Obj {
    albedo: vec4f, // w = kind: 0 diffuse, 1 mirror, 2 light bulb
    shape: vec4f,  // w = shape: 0 sphere (x = radius), 1 box (xyz = half-extents)
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> objs: array<Obj>;
@group(0) @binding(2) var<storage, read> hist: array<vec4f>;
@group(0) @binding(3) var<storage, read> chunks: array<vec4f>;
@group(0) @binding(4) var out_tex: texture_storage_2d<rgba16float, write>;

const CHUNK: u32 = 16u;
const ROOM_MIN = vec3f(-9.0, 0.0, -9.0);
const ROOM_MAX = vec3f(9.0, 6.0, 9.0);
const EPS: f32 = 1e-3;
const MISS: f32 = 1e30;
const MAX_BOUNCES: u32 = 4u;
const KIND_DIFFUSE: u32 = 0u;
const KIND_MIRROR: u32 = 1u;
const KIND_LIGHT: u32 = 2u;
const KIND_ROOM: u32 = 3u;

fn light_speed() -> f32 { return P.cam_pos.w; }
fn hist_dt() -> f32 { return P.misc.x; }
// History: per object, newest first, two vec4 per sample:
// [position, payload] and [rotation quaternion] (light: [position, on] and [direction, cone]).
fn sample(obj: u32, k: u32) -> vec4f { return hist[(obj * P.counts.z + k) * 2u]; }
fn sample_rot(obj: u32, k: u32) -> vec4f { return hist[(obj * P.counts.z + k) * 2u + 1u]; }

fn quat_rotate(q: vec4f, v: vec3f) -> vec3f {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

fn quat_unrotate(q: vec4f, v: vec3f) -> vec3f {
    return quat_rotate(vec4f(-q.xyz, q.w), v);
}

fn quat_mul(a: vec4f, b: vec4f) -> vec4f {
    return vec4f(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

// Angular velocity that turns qb into qa over h seconds.
fn angular_velocity(qa: vec4f, qb: vec4f, h: f32) -> vec3f {
    var dq = quat_mul(qa, vec4f(-qb.xyz, qb.w));
    if (dq.w < 0.0) { dq = -dq; }
    let sn = length(dq.xyz);
    if (sn < 1e-7) { return vec3f(0.0); }
    return dq.xyz / sn * (2.0 * atan2(sn, dq.w) / h);
}

// Normalized lerp, taking the short way round.
fn quat_nlerp(a: vec4f, b: vec4f, t: f32) -> vec4f {
    return normalize(mix(a, select(b, -b, dot(a, b) < 0.0), t));
}

// Plain 3D ray vs sphere (d normalized). Returns (enter, exit), or an empty interval.
fn sphere_interval(o: vec3f, d: vec3f, c: vec3f, r: f32) -> vec2f {
    let m = o - c;
    let b = dot(m, d);
    let disc = b * b - (dot(m, m) - r * r);
    if (disc < 0.0) { return vec2f(MISS, -MISS); }
    let sq = sqrt(disc);
    return vec2f(-b - sq, -b + sq);
}

fn is_box(shape: vec4f) -> bool { return shape.w > 0.5; }

fn bound_radius(shape: vec4f) -> f32 {
    return select(shape.x, length(shape.xyz), is_box(shape));
}

// Static ray vs a shape centered at the origin of its own frame. d need not be
// normalized; the result is in units of d. Returns the entering s, or MISS.
fn shape_enter(shape: vec4f, o: vec3f, d: vec3f) -> f32 {
    if (!is_box(shape)) {
        let A = dot(d, d);
        let B = dot(o, d);
        let disc = B * B - A * (dot(o, o) - shape.x * shape.x);
        if (disc < 0.0) { return MISS; }
        return (-B - sqrt(disc)) / A;
    }
    // Slab test.
    let inv = 1.0 / d;
    let t1 = (-shape.xyz - o) * inv;
    let t2 = (shape.xyz - o) * inv;
    let tn = min(t1, t2);
    let tf = max(t1, t2);
    let s_in = max(max(tn.x, tn.y), tn.z);
    let s_out = min(min(tf.x, tf.y), tf.z);
    if (s_in > s_out || s_out < 0.0) { return MISS; }
    return s_in;
}

// Outward normal at a surface point, in the shape's frame.
fn shape_normal(shape: vec4f, local: vec3f) -> vec3f {
    if (!is_box(shape)) { return local / shape.x; }
    let q = abs(local) / shape.xyz;
    if (q.x >= q.y && q.x >= q.z) { return vec3f(sign(local.x), 0.0, 0.0); }
    if (q.y >= q.z) { return vec3f(0.0, sign(local.y), 0.0); }
    return vec3f(0.0, 0.0, sign(local.z));
}

struct ShapeHit {
    s: f32,
    local: vec3f, // hit point in the object's own frame
    w: f32,       // sample payload (light: on flag)
    rot: vec4f,   // orientation at the emission time
    vel: vec3f,   // world velocity of the hit surface point (translation + spin)
}

// First crossing of the null ray with an object's world tube.
// History segments are walked newest -> oldest, which is increasing s,
// so the first hit found is the nearest one for this object.
//
// Inside one segment the object moves linearly, so put the ray in its moving frame:
//   o' = R^T (o - pa - v*(t0 - ta)),  d' = R^T (d + v/c)
// Any static ray-shape test on (o', d') then returns the same s.
fn hit_shape(obj: u32, shape: vec4f, o: vec3f, d: vec3f, t0: f32, s_max: f32) -> ShapeHit {
    var res = ShapeHit(MISS, vec3f(0.0), 0.0, vec4f(0.0, 0.0, 0.0, 1.0), vec3f(0.0));
    let radius = bound_radius(shape);
    let valid = P.counts.y;
    if (valid < 2u) { return res; }
    let c = light_speed();
    let h = hist_dt();
    let nseg = valid - 1u;
    let k_first = u32(max(floor(-t0 / h), 0.0));
    if (k_first >= nseg) { return res; }
    let nchunks = (nseg + CHUNK - 1u) / CHUNK;
    let chunks_per_obj = P.counts.z / CHUNK;

    for (var j = k_first / CHUNK; j < nchunks; j++) {
        // s range covered by this chunk's time span
        let ch_lo = max((t0 + f32(j * CHUNK) * h) * c, 0.0);
        let ch_hi = (t0 + f32(j * CHUNK + CHUNK) * h) * c;
        if (ch_lo > s_max) { break; }
        let bs = chunks[obj * chunks_per_obj + j];
        let iv = sphere_interval(o, d, bs.xyz, bs.w);
        if (iv.y < ch_lo || iv.x > min(ch_hi, s_max)) { continue; }

        let k_end = min(j * CHUNK + CHUNK, nseg);
        for (var k = max(j * CHUNK, k_first); k < k_end; k++) {
            let ta = -f32(k) * h;
            let seg_lo = max((t0 - ta) * c, 0.0);
            let seg_hi = (t0 - ta + h) * c;
            if (seg_lo > s_max) { return res; }
            let sa = sample(obj, k);
            let pa = sa.xyz;
            let pb = sample(obj, k + 1u).xyz;
            // Linear motion inside the segment: center(t) = pa + v*(t - ta).
            // Substituting t = t0 - s/c, the offset from the center is m + w*s.
            let v = (pa - pb) / h;
            let m = o - pa - v * (t0 - ta);
            let w = d + v / c;
            // Bounding sphere first: |m + w*s|^2 = r^2, a quadratic in s.
            let A = dot(w, w);
            let B = dot(m, w);
            let C = dot(m, m) - radius * radius;
            let disc = B * B - A * C;
            if (disc < 0.0) { continue; }
            let sq = sqrt(disc);
            var s = (-B - sq) / A;
            if (s >= seg_hi || (-B + sq) / A < seg_lo) { continue; }
            // Orientation at s, as a fraction of the way from sample k to k+1.
            let qa = sample_rot(obj, k);
            let qb = sample_rot(obj, k + 1u);
            let s_ta = (t0 - ta) * c;
            var q = quat_nlerp(qa, qb, clamp((s - s_ta) / (h * c), 0.0, 1.0));
            if (is_box(shape)) {
                // Solve with the orientation at the bounding-sphere entry, then once more
                // with the orientation at that hit. Two passes are plenty at 60 Hz.
                let s1 = shape_enter(shape, quat_unrotate(q, m), quat_unrotate(q, w));
                if (s1 >= MISS) { continue; }
                q = quat_nlerp(qa, qb, clamp((s1 - s_ta) / (h * c), 0.0, 1.0));
                s = shape_enter(shape, quat_unrotate(q, m), quat_unrotate(q, w));
            }
            if (s >= seg_lo && s < seg_hi && s > EPS && s < s_max) {
                res.s = s;
                res.local = quat_unrotate(q, m + w * s);
                res.w = sa.w;
                res.rot = q;
                // The light's second row is its aim, not a rotation: it doesn't spin.
                var spin = vec3f(0.0);
                if (obj < P.counts.x) { spin = angular_velocity(qa, qb, h); }
                res.vel = v + cross(spin, quat_rotate(q, res.local));
                return res;
            }
        }
    }
    return res;
}

// Ray from inside the static room box. Returns (normal, s).
fn hit_room(o: vec3f, d: vec3f) -> vec4f {
    let inv = 1.0 / d;
    let t_exit = max((ROOM_MAX - o) * inv, (ROOM_MIN - o) * inv);
    var s = t_exit.x;
    var n = vec3f(-sign(d.x), 0.0, 0.0);
    if (t_exit.y < s) { s = t_exit.y; n = vec3f(0.0, -sign(d.y), 0.0); }
    if (t_exit.z < s) { s = t_exit.z; n = vec3f(0.0, 0.0, -sign(d.z)); }
    return vec4f(n, s);
}

// Lat/long checker in the sphere's own frame (l = unit local position), 8 x 16 cells,
// with a white cap on the +y pole so you can tell which way is up.
fn sphere_albedo(l: vec3f, base: vec3f) -> vec3f {
    if (l.y > 0.95) { return vec3f(0.95); }
    let lat = asin(clamp(l.y, -1.0, 1.0)) / 3.14159265 + 0.5;
    let lon = atan2(l.z, l.x) / 6.28318531 + 0.5;
    let checker = (i32(floor(lat * 8.0)) + i32(floor(lon * 16.0))) & 1;
    return select(base, base * 0.3, checker == 1);
}

// 6 x 6 checker on each box face in face-local UV, each face a different tint.
fn box_albedo(local: vec3f, n: vec3f, half: vec3f, base: vec3f) -> vec3f {
    var tints = array<vec3f, 6>(
        vec3f(1.0, 0.55, 0.5), vec3f(0.55, 1.0, 0.6),   // -x, +x
        vec3f(0.55, 0.65, 1.0), vec3f(1.0, 0.95, 0.5), // -y, +y
        vec3f(0.95, 0.6, 1.0), vec3f(0.5, 0.95, 1.0),  // -z, +z
    );
    let q = local / half;
    var uv = q.yz;
    var axis = 0u;
    if (abs(n.y) > 0.5) { uv = q.xz; axis = 1u; }
    if (abs(n.z) > 0.5) { uv = q.xy; axis = 2u; }
    let cell = vec2i(floor((uv + 1.0) * 3.0));
    let tinted = base * tints[axis * 2u + select(0u, 1u, dot(n, vec3f(1.0)) > 0.0)];
    return select(tinted, tinted * 0.3, ((cell.x + cell.y) & 1) == 1);
}

fn room_albedo(p: vec3f, n: vec3f) -> vec3f {
    if (n.y > 0.5) {
        let checker = (i32(floor(p.x)) + i32(floor(p.z))) & 1;
        return select(vec3f(0.75), vec3f(0.3), checker == 1);
    }
    if (n.y < -0.5) { return vec3f(0.8); }
    if (abs(n.x) > 0.5) {
        return select(vec3f(0.75, 0.25, 0.2), vec3f(0.2, 0.6, 0.25), n.x < 0.0);
    }
    return vec3f(0.75, 0.72, 0.65);
}

struct Emission {
    pos: vec3f,
    on: f32,
    dir: vec3f,
    cone: f32, // cos of the cone half-angle; 0 = omnidirectional
    vel: vec3f,
    ok: bool,
}

// Where/when was the light when it emitted the photon reaching p at time tp?
// f(k) = c*(tp - t_k) - |p - L_k| increases with k while the light moves slower
// than c, so binary-search the history for the sign change.
fn light_emission(p: vec3f, tp: f32) -> Emission {
    var e = Emission(vec3f(0.0), 0.0, vec3f(0.0), 0.0, vec3f(0.0), false);
    let obj = P.counts.x;
    let valid = P.counts.y;
    if (valid < 2u) { return e; }
    let c = light_speed();
    let h = hist_dt();
    var lo = 0u;
    var hi = valid - 1u;
    let f_hi = c * (tp + f32(hi) * h) - distance(p, sample(obj, hi).xyz);
    if (f_hi < 0.0) { return e; } // emitted before history began
    var f_lo = c * tp - distance(p, sample(obj, 0u).xyz);
    if (f_lo >= 0.0) {
        let s0 = sample(obj, 0u);
        let d0 = sample_rot(obj, 0u);
        let v0 = (s0.xyz - sample(obj, 1u).xyz) / h;
        return Emission(s0.xyz, s0.w, d0.xyz, d0.w, v0, true);
    }
    var f_h = f_hi;
    loop {
        if (hi - lo <= 1u) { break; }
        let mid = (lo + hi) / 2u;
        let f_mid = c * (tp + f32(mid) * h) - distance(p, sample(obj, mid).xyz);
        if (f_mid >= 0.0) { hi = mid; f_h = f_mid; } else { lo = mid; f_lo = f_mid; }
    }
    let a = f_lo / (f_lo - f_h);
    let s_lo = sample(obj, lo);
    let s_hi = sample(obj, hi);
    let d_lo = sample_rot(obj, lo);
    let d_hi = sample_rot(obj, hi);
    return Emission(
        mix(s_lo.xyz, s_hi.xyz, a),
        select(s_lo.w, s_hi.w, a > 0.5),
        normalize(mix(d_lo.xyz, d_hi.xyz, a)),
        select(d_lo.w, d_hi.w, a > 0.5),
        (s_lo.xyz - s_hi.xyz) / (f32(hi - lo) * h),
        true,
    );
}

struct Shade {
    ambient: vec3f,
    direct: vec3f,
    shift: f32, // Doppler factor of the direct light on its way in, light -> surface
}

// Diffuse shading at p, on a surface moving with velocity u.
fn shade_diffuse(p: vec3f, n: vec3f, u: vec3f, tp: f32, albedo: vec3f) -> Shade {
    // Ambient is instantaneous (non-physical); it only keeps the room visible.
    var sh = Shade(albedo * P.misc.z, vec3f(0.0), 1.0);
    let e = light_emission(p, tp);
    if (!e.ok || e.on < 0.5) { return sh; }
    let to_l = e.pos - p;
    let dist = length(to_l);
    let l = to_l / dist;
    let ndl = dot(n, l);
    if (ndl <= 0.0) { return sh; }
    // Cone test uses where the light was aimed at emission time.
    var spot = 1.0;
    if (e.cone > 0.0) {
        spot = smoothstep(e.cone, e.cone + 0.02, dot(-l, e.dir));
        if (spot <= 0.0) { return sh; }
    }
    // Shadow ray: the photon travelled light -> p, so walk back from p in time.
    let o = p + n * EPS;
    for (var i = 0u; i < P.counts.x; i++) {
        if (hit_shape(i, objs[i].shape, o, l, tp, dist - 0.1).s < MISS) { return sh; }
    }
    sh.direct = albedo * ndl * spot * P.misc.y / (dist * dist);
    // The photon travelled along -l: sent by the moving light, caught by the moving surface.
    sh.shift = doppler_out(e.vel, -l) * doppler_in(u, -l);
    return sh;
}

// Classical Doppler factors (received / emitted frequency) in the room frame, where light
// moves at c in every direction. `dir` is the photon's direction of travel.
fn doppler_out(emitter_vel: vec3f, dir: vec3f) -> f32 {
    return 1.0 / max(1.0 - dot(emitter_vel, dir) / light_speed(), 0.05);
}

fn doppler_in(receiver_vel: vec3f, dir: vec3f) -> f32 {
    return max(1.0 - dot(receiver_vel, dir) / light_speed(), 0.05);
}

// Doppler color. Each RGB channel is a narrow band (610, 550, 465 nm). A factor D moves
// every band by strength * log2(D) along log-frequency, then re-splits it between the
// neighboring channels. Bands pushed past 700 nm (infrared) or 400 nm (ultraviolet) fade
// out. D = 1 leaves the color exactly as it was.
const U_IR: f32 = -0.1986; // log2(610 / 700)
const U_R: f32 = 0.0;
const U_G: f32 = 0.1494;   // log2(610 / 550)
const U_B: f32 = 0.3916;   // log2(610 / 465)
const U_UV: f32 = 0.6088;  // log2(610 / 400)

fn band(u: f32) -> vec3f {
    if (u <= U_R) { return vec3f(clamp((u - U_IR) / (U_R - U_IR), 0.0, 1.0), 0.0, 0.0); }
    if (u <= U_G) { let a = (u - U_R) / (U_G - U_R); return vec3f(1.0 - a, a, 0.0); }
    if (u <= U_B) { let a = (u - U_G) / (U_B - U_G); return vec3f(0.0, 1.0 - a, a); }
    return vec3f(0.0, 0.0, clamp((U_UV - u) / (U_UV - U_B), 0.0, 1.0));
}

fn doppler_color(rgb: vec3f, shift: f32) -> vec3f {
    let x = P.fx.x * log2(shift);
    if (abs(x) < 1e-5) { return rgb; }
    return rgb.r * band(U_R + x) + rgb.g * band(U_G + x) + rgb.b * band(U_B + x);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let size = textureDimensions(out_tex);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(size) * 2.0 - 1.0;

    var o = P.cam_pos.xyz;
    var d = normalize(P.cam_fwd.xyz + uv.x * P.cam_right.xyz - uv.y * P.cam_up.xyz);
    var t0 = 0.0;
    var throughput = vec3f(1.0);
    var color = vec3f(0.0);
    var age = 0.0; // debug: how long ago the light left the last surface hit
    var shift = 1.0; // Doppler factor along the path so far (received at camera / emitted here)

    for (var bounce = 0u; bounce < MAX_BOUNCES; bounce++) {
        let room = hit_room(o, d);
        var s_best = room.w;
        var n = room.xyz;
        var kind = KIND_ROOM;
        var albedo = vec3f(0.0);
        var payload = 0.0;
        var best = ShapeHit(MISS, vec3f(0.0), 0.0, vec4f(0.0, 0.0, 0.0, 1.0), vec3f(0.0));
        var best_i = 0u;
        // Bodies plus the light bulb (object counts.x).
        for (var i = 0u; i <= P.counts.x; i++) {
            let sh = hit_shape(i, objs[i].shape, o, d, t0, s_best);
            if (sh.s < s_best) {
                s_best = sh.s;
                best = sh;
                best_i = i;
            }
        }
        let p = o + d * s_best;
        let tp = t0 - s_best / light_speed();
        if (best.s < MISS) {
            let shape = objs[best_i].shape;
            let nl = shape_normal(shape, best.local);
            n = quat_rotate(best.rot, nl);
            kind = u32(objs[best_i].albedo.w);
            albedo = objs[best_i].albedo.xyz;
            payload = best.w;
            if (kind == KIND_DIFFUSE) {
                if (is_box(shape)) {
                    albedo = box_albedo(best.local, nl, shape.xyz, albedo);
                } else {
                    albedo = sphere_albedo(nl, albedo);
                }
            }
        }
        age = -tp;
        // The photon leaves this surface toward the camera along -d. The room has zero velocity.
        shift *= doppler_out(best.vel, -d);

        if (kind == KIND_LIGHT) {
            // The bulb glows only if it was on when this light left it.
            color += doppler_color(throughput * mix(vec3f(0.05), vec3f(4.0, 3.6, 2.8), payload), shift);
            break;
        }
        if (kind == KIND_MIRROR) {
            throughput *= albedo;
            d = reflect(d, n);
            // ...and it arrived along the reflected ray, reversed.
            shift *= doppler_in(best.vel, -d);
            o = p + n * EPS;
            t0 = tp;
            continue;
        }
        if (kind == KIND_ROOM) { albedo = room_albedo(p, n); }
        let sh = shade_diffuse(p, n, best.vel, tp, albedo);
        color += doppler_color(throughput * sh.ambient, shift)
            + doppler_color(throughput * sh.direct, shift * sh.shift);
        break;
    }

    if (P.misc.w > 0.5) {
        textureStore(out_tex, gid.xy, vec4f(vec3f(age_gray(age)), 1.0));
        return;
    }
    textureStore(out_tex, gid.xy, vec4f(color / (1.0 + color), 1.0));
}

// Debug view, like a 1/w depth buffer but in time: white = light left just now,
// fading toward black with delay. Half brightness at AGE_HALF seconds; most
// contrast up close. Squared so the sRGB swapchain shows it perceptually linear.
const AGE_HALF: f32 = 1.0;

fn age_gray(age: f32) -> f32 {
    let v = 1.0 / (1.0 + max(age, 0.0) / AGE_HALF);
    return v * v;
}
