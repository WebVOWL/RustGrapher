struct VertIn {
    @location(0) quad_pos: vec2<f32>, // [-1..1]
    @location(1) start: vec2<f32>,    // start of edge (in px)
    @location(2) end: vec2<f32>,      // end of edge (in px)
    @location(3) center: vec2<f32>,   // point ON curve at t=0.5 (in px)
};

struct VertOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) v_uv: vec2<f32>,
    @location(1) v_start: vec2<f32>,
    @location(2) v_center_ctrl: vec2<f32>, // control point used by quadratic bezier
    @location(3) v_end: vec2<f32>,
    @location(4) v_mbr_min: vec2<f32>,
    @location(5) v_mbr_max: vec2<f32>,
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

    return out;
}

@fragment
fn fs_edge_main(in: VertOut) -> @location(0) vec4<f32> {
    let px = mix(in.v_mbr_min, in.v_mbr_max, in.v_uv);
    let p0 = in.v_start;
    let ctrl = in.v_center_ctrl;
    let p2 = in.v_end;

    // Precompute polynomial coefficients for quadratic: B(t) = a*t^2 + b*t + p0
    let a = p0 - 2.0 * ctrl + p2;
    let b = 2.0 * (ctrl - p0);
    let _c = p0 - px;

    // Initial sample (cheap safety fallback)
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

    // refine with Newton iteration around best_t
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

    let alpha = 1.0 - smoothstep(LINE_THICKNESS - AA_SOFTNESS,
                                 LINE_THICKNESS,
                                 dist);

    let color = vec3<f32>(0.0);
    return vec4<f32>(color, alpha);
}
