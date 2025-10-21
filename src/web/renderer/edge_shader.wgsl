struct VertIn {
    @location(0) quad_pos: vec2<f32>,         // [-1..1]
    @location(1) start: vec2<f32>,            // start of edge (in px)
    @location(2) end: vec2<f32>,              // end of edge (in px)
    @location(3) center: vec2<f32>,           // point ON curve at t=0.5 (in px)
    @location(4) end_shape: u32,              // The shape of the node pointed to, 0: Circle, 1: Rectangle
    @location(5) shape_dimensions: vec2<f32>, // The radius of a circle or the width and height of a rectangle
};

struct VertOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) v_uv: vec2<f32>,
    @location(1) v_start: vec2<f32>,
    @location(2) v_center_ctrl: vec2<f32>, // control point used by quadratic bezier
    @location(3) v_end: vec2<f32>,
    @location(4) v_mbr_min: vec2<f32>,
    @location(5) v_mbr_max: vec2<f32>,
    @location(6) v_end_shape: u32,
    @location(7) v_shape_dimensions: vec2<f32>,
};

@group(0) @binding(0)
var<uniform> u_resolution: vec4<f32>; // xy = screen size

const LINE_THICKNESS = 1.75;
const AA_SOFTNESS    = 1.5;

// Evaluate Bézier point
fn bezier(p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, t: f32) -> vec2<f32> {
    return mix(mix(p0, p1, t), mix(p1, p2, t), t);
}

@vertex
fn vs_edge_main(in: VertIn) -> VertOut {
    var out: VertOut;

    let p0 = in.start;
    let p2 = in.end;
    let on_mid = in.center; // point that must lie on the curve at t = 0.5

    // compute quadratic bezier control point that ensures B(0.5) == on_mid
    // derived from 0.25*p0 + 0.5*c + 0.25*p2 = on_mid  => c = 2*on_mid - 0.5*(p0 + p2)
    let ctrl = (4.0 * on_mid - p0 - p2) * 0.5;

    // find possible extrema using control point
    let denom_x = p0.x - 2.0 * ctrl.x + p2.x;
    let denom_y = p0.y - 2.0 * ctrl.y + p2.y;

    var min_p = min(p0, p2);
    var max_p = max(p0, p2);

    if (abs(denom_x) > 1e-5) {
        let tx = clamp((p0.x - ctrl.x) / denom_x, 0.0, 1.0);
        let bx = bezier(p0, ctrl, p2, tx);
        min_p = min(min_p, bx);
        max_p = max(max_p, bx);
    }
    if (abs(denom_y) > 1e-5) {
        let ty = clamp((p0.y - ctrl.y) / denom_y, 0.0, 1.0);
        let by = bezier(p0, ctrl, p2, ty);
        min_p = min(min_p, by);
        max_p = max(max_p, by);
    }

    // pad for thickness
    min_p -= vec2<f32>(LINE_THICKNESS);
    max_p += vec2<f32>(LINE_THICKNESS);

    // map quad to this rectangle
    let pos_px = mix(min_p, max_p, in.quad_pos * 0.5 + vec2<f32>(0.5));
    let ndc = (pos_px / u_resolution.xy) * 2.0 - vec2<f32>(1.0, 1.0);

    out.clip_position = vec4<f32>(ndc, 0.0, 1.0);
    out.v_uv = in.quad_pos * 0.5 + vec2<f32>(0.5);
    out.v_start = p0;
    out.v_center_ctrl = ctrl;
    out.v_end = p2;
    out.v_mbr_min = min_p;
    out.v_mbr_max = max_p;
    out.v_end_shape = in.end_shape;
    out.v_shape_dimensions = in.shape_dimensions;

    return out;
}

// arrow constants (pixels)
const ARROW_LENGTH_PX = 10.0;
const ARROW_WIDTH_PX  = 10.0;
const ARROW_AA_PX     = 1.5; // anti-alias softness in pixels

@fragment
fn fs_edge_main(in: VertOut) -> @location(0) vec4<f32> {
    let px = mix(in.v_mbr_min, in.v_mbr_max, in.v_uv); // this fragment's pixel position
    let p0 = in.v_start;
    let ctrl = in.v_center_ctrl;
    let p2 = in.v_end;

    // Precompute polynomial coefficients for quadratic: B(t) = a*t^2 + b*t + p0
    let a = p0 - 2.0 * ctrl + p2;
    let b = 2.0 * (ctrl - p0);

    var best_d2 = 1e12;
    var best_t = 0.0;
    for (var i = 0u; i <= 8u; i = i + 1u) {
        let t = f32(i) / 8.0;
        let bt = bezier(p0, ctrl, p2, t);
        let d2 = dot(bt - px, bt - px);
        if (d2 < best_d2) {
            best_d2 = d2;
            best_t = t;
        }
    }

    var t = best_t;
    for (var i = 0; i < 8; i = i + 1) {
        let bt  = a * t * t + b * t + p0;
        let dBt = 2.0 * a * t + b;
        let diff = bt - px;
        let f = dot(diff, dBt);
        let df = 2.0 * dot(dBt, dBt) + 2.0 * dot(diff, a);
        if (abs(df) < 1e-6) { break; }
        t = clamp(t - f / df, 0.0, 1.0);
    }

    let bt = bezier(p0, ctrl, p2, t);
    let dist = length(bt - px);

    // line alpha (anti-aliased)
    let line_alpha = 1.0 - smoothstep(LINE_THICKNESS - AA_SOFTNESS, LINE_THICKNESS, dist);

    // Arrow
    // tangent at t=1:
    let tangent = 2.0 * a + b; // derivative at t=1
    var dir = tangent;
    let dir_len = length(dir);
    if (dir_len < 1e-6) {
        dir = normalize(p2 - p0);
    } else {
        dir = dir / dir_len;
    }

    // find intersection point with node shape boundary
    var tip = p2;
    let shape_type = i32(in.v_end_shape);
    let dims = in.v_shape_dimensions;

    // Circle shape
    if (shape_type == 0) {
        let radius = dims.x;
        tip = p2 - dir * radius;
    }

    // Rectangle
    if (shape_type == 1) {
        let half_size = dims;
        // Ray-box intersection: find smallest positive t where |p2 - dir*t| hits the box
        let inv_dir = 1.0 / dir;
        let t1 = (-half_size - vec2<f32>(0.0)) * inv_dir;
        let t2 = (half_size - vec2<f32>(0.0)) * inv_dir;
        let tmin = max(min(t1.x, t2.x), min(t1.y, t2.y));
        let tmax = min(max(t1.x, t2.x), max(t1.y, t2.y));
        let t_hit = max(tmin, 0.0);
        tip = p2 - dir * t_hit;
    }

    // Define triangle: tip at intersection point, base behind tip
    let base_center = tip - dir * ARROW_LENGTH_PX;
    let perp = vec2<f32>(-dir.y, dir.x);
    let halfw = ARROW_WIDTH_PX * 0.5;
    let left = base_center + perp * halfw;
    let right = base_center - perp * halfw;

    let area_total = tri_area(tip, left, right);
    let area_sub = tri_area(px, left, right) + tri_area(tip, px, right) + tri_area(tip, left, px);
    let area_diff = abs(area_sub - area_total);

    var arrow_alpha = 0.0;
    if (area_total > 1e-5) {
        let normalized_diff = area_diff / area_total;
        arrow_alpha = clamp(1.0 - smoothstep(0.0, 0.06, normalized_diff), 0.0, 1.0);
    }

    let color = vec3<f32>(0.0);

    let final_alpha = max(line_alpha, arrow_alpha);

    return vec4<f32>(color, final_alpha);
}

// Simple point-in-triangle check
fn tri_area(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> f32 {
    return abs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)) * 0.5;
}