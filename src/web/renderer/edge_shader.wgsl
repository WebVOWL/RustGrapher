struct VertIn {
    @location(0) quad_pos: vec2<f32>,         // [-1..1]
    @location(1) start: vec2<f32>,            // start of edge (in px)
    @location(2) center: vec2<f32>,           // point ON curve at t=0.5 (in px)
    @location(3) end: vec2<f32>,              // end of edge (in px)
    @location(4) end_shape: u32,              // The shape of the node pointed to, 0: Circle, 1: Rectangle
    @location(5) shape_dimensions: vec2<f32>, // The radius of a circle or the width and height of a rectangle
    @location(6) line_type: u32,
};

struct VertOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) v_uv: vec2<f32>,
    @location(1) v_start: vec2<f32>,
    @location(2) v_center_ctrl: vec2<f32>, // control point used by quadratic bezier
    @location(3) v_end: vec2<f32>,
    @interpolate(flat) @location(4) v_mbr_min: vec2<f32>,
    @interpolate(flat) @location(5) v_mbr_max: vec2<f32>,
    @interpolate(flat) @location(6) v_end_shape: u32,
    @location(7) v_shape_dimensions: vec2<f32>,
    @interpolate(flat) @location(8) v_line_type: u32,

    // Precomputed quadratic coefficients and tangent
    @location(9) v_quad_a: vec2<f32>,
    @location(10) v_quad_b: vec2<f32>,
    @location(11) v_tangent_at1: vec2<f32>,
    @location(12) v_approx_len: f32,
};

@group(0) @binding(0)
var<uniform> u_resolution: vec4<f32>; // xy = screen size

// visual & geometry constants
const LINE_THICKNESS: f32 = 1.0;
const AA_SOFTNESS: f32 = 1.0;

// arrow constants (pixels)
const ARROW_LENGTH_PX: f32 = 10.0;
const ARROW_WIDTH_PX: f32 = 15.0;
const ARROW_AA: f32 = 0.1;
const NODE_RADIUS_PIX: f32 = 50.0;

// Tunable performance/precision
const SAMPLE_COUNT: u32 = 24u;   // lower = faster, higher = more accurate
const NR_ITER: i32 = 10;          // Newton-Raphson iterations

// Evaluate Bézier point (quadratic) directly using coefficients
fn bezier_point_from_coeffs(a: vec2<f32>, b: vec2<f32>, p0: vec2<f32>, t: f32) -> vec2<f32> {
    // a*t^2 + b*t + p0
    return a * t * t + b * t + p0;
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

    // quadratic coefficients: B(t) = a*t^2 + b*t + p0
    let a = p0 - 2.0 * ctrl + p2;
    let b = 2.0 * (ctrl - p0);

    // precompute tangent at t = 1 (derivative = 2*a*t + b => at t=1 => 2*a + b)
    let tangent_at1 = 2.0 * a + b;

    // approximate curve length cheaply: chord + two control segment
    let approx_len = distance(p0, ctrl) + distance(ctrl, p2);

    // find possible extrema using control point
    let denom_x = p0.x - 2.0 * ctrl.x + p2.x;
    let denom_y = p0.y - 2.0 * ctrl.y + p2.y;

    var min_p = min(p0, p2);
    var max_p = max(p0, p2);

    if (abs(denom_x) > 1e-5) {
        let tx = clamp((p0.x - ctrl.x) / denom_x, 0.0, 1.0);
        let bx = bezier_point_from_coeffs(a, b, p0, tx);
        min_p = min(min_p, bx);
        max_p = max(max_p, bx);
    }
    if (abs(denom_y) > 1e-5) {
        let ty = clamp((p0.y - ctrl.y) / denom_y, 0.0, 1.0);
        let by = bezier_point_from_coeffs(a, b, p0, ty);
        min_p = min(min_p, by);
        max_p = max(max_p, by);
    }

    // pad for thickness and arrow
    min_p -= vec2<f32>(LINE_THICKNESS + AA_SOFTNESS + ARROW_WIDTH_PX);
    max_p += vec2<f32>(LINE_THICKNESS + AA_SOFTNESS + ARROW_WIDTH_PX);

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
    out.v_line_type = in.line_type;

    // pass precomputed coefficients and tangent
    out.v_quad_a = a;
    out.v_quad_b = b;
    out.v_tangent_at1 = tangent_at1;
    out.v_approx_len = approx_len;

    return out;
}

// Distance from point to line segment
fn point_to_segment_dist(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let denom = dot(ab, ab);
    if (denom <= 1e-8) {
        return length(p - a);
    }
    let t = clamp(dot(p - a, ab) / denom, 0.0, 1.0);
    let closest = a + ab * t;
    return length(p - closest);
}

// fast triangle area via 0.5 * abs(cross)
fn tri_area_fast(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> f32 {
    return abs(((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x))) * 0.5;
}

// Distance from point to quadratic bezier curve using sampling + Newton-Raphson
fn dist_and_t_to_bezier(px: vec2<f32>, p0: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    // coarse sampling to get starting t
    var best_d2 = 1e12;
    var best_t = 0.0;

    for (var i: u32 = 0u; i <= SAMPLE_COUNT; i = i + 1u) {
        let t = f32(i) / f32(SAMPLE_COUNT);
        let bt = bezier_point_from_coeffs(a, b, p0, t);
        let d2 = dot(bt - px, bt - px);
        if (d2 < best_d2) {
            best_d2 = d2;
            best_t = t;
        }
    }

    // Newton-Raphson refine
    var t = best_t;
    for (var iter: i32 = 0; iter < NR_ITER; iter = iter + 1) {
        let bt = a * t * t + b * t + p0;
        let dBt = 2.0 * a * t + b;
        let diff = bt - px;
        let f = dot(diff, dBt);
        let df = 2.0 * dot(dBt, dBt) + 2.0 * dot(diff, a);
        if (abs(df) < 1e-6) { break; }
        let dt = f / df;
        if (abs(dt) < 1e-5) { break; }
        t = t - dt;
        // small safety clamp while refining
        if (t < -0.1) { t = -0.1; }
        if (t > 1.1)  { t = 1.1; }
    }

    let t_clamped = clamp(t, 0.0, 1.0);
    let bt = a * t_clamped * t_clamped + b * t_clamped + p0;
    let dist = length(bt - px);
    return vec2<f32>(dist, t_clamped);
}

@fragment
fn fs_edge_main(in: VertOut) -> @location(0) vec4<f32> {
    let px = mix(in.v_mbr_min, in.v_mbr_max, in.v_uv);
    let p0 = in.v_start;
    let ctrl = in.v_center_ctrl;
    let p2 = in.v_end;

    let a = in.v_quad_a;
    let b = in.v_quad_b;

    // Get distance and t to curve (t in [0..1])
    let dt = dist_and_t_to_bezier(px, p0, a, b);
    let dist = dt.x;
    let t_closest = dt.y;

    let EPS_T: f32 = 0.02;
    if (t_closest < -EPS_T || t_closest > 1.0 + EPS_T) {
        discard;
    }

    // Anti-aliased line alpha
    var line_alpha = 1.0 - smoothstep(LINE_THICKNESS - AA_SOFTNESS, LINE_THICKNESS + AA_SOFTNESS, dist);

    // dashed/dotted pattern
    if (in.v_line_type == 1u || in.v_line_type == 2u || in.v_line_type == 4u) {
        let pattern_repeats = in.v_approx_len / 10.0;
        let dot_fraction = 0.6;
        let pattern_phase = fract(t_closest * pattern_repeats);
        let fade = 0.05;
        let dot_mask = smoothstep(0.0, fade, pattern_phase) * (1.0 - smoothstep(dot_fraction, dot_fraction + fade, pattern_phase));
        line_alpha *= dot_mask;
    }

    // Arrow calculation
    var dir = in.v_tangent_at1;
    var dir_len = length(dir);
    if (dir_len < 1e-6) {
        dir = normalize(p2 - p0);
    } else {
        dir = dir / dir_len;
    }

    var tip = p2;
    let shape_type = i32(in.v_end_shape);
    let dims = in.v_shape_dimensions;

    if (shape_type == 1) {
        // rectangle intersection fallback to shift tip inward
        let rect_size = vec2<f32>(0.9, 0.25 * dims.y);
        let half_size = rect_size;

        // avoid divide-by-zero; small epsilon
        let eps = 1e-6;
        var safe_dir = dir;
        if (abs(safe_dir.x) < eps) { safe_dir.x = sign(safe_dir.x) * eps; }
        if (abs(safe_dir.y) < eps) { safe_dir.y = sign(safe_dir.y) * eps; }

        let inv_dir = -1.0 / safe_dir;
        let t1 = (-half_size) * inv_dir;
        let t2 = (half_size) * inv_dir;
        let tmin = max(min(t1.x, t2.x), min(t1.y, t2.y));
        let tmax = min(max(t1.x, t2.x), max(t1.y, t2.y));

        var t_hit = 0.0;
        if (tmin <= tmax && tmax > 0.0) {
            t_hit = tmax;
        }

        tip = p2 - dir * t_hit;
    }

    // No arrow for disjoint edges
    if (in.v_line_type == 2u) {
        return vec4<f32>(vec3<f32>(0.0), line_alpha);
    }

    // Arrow triangle geometry
    let base_center = tip - dir * ARROW_LENGTH_PX;
    let perp = vec2<f32>(-dir.y, dir.x);
    let halfw = ARROW_WIDTH_PX * 0.5;
    let left = base_center + perp * halfw;
    let right = base_center - perp * halfw;

    // Use cross-based triangle area
    let area_total = tri_area_fast(tip, left, right);

    // Barycentric-like test via area difference
    let area_sub = tri_area_fast(px, left, right) + tri_area_fast(tip, px, right) + tri_area_fast(tip, left, px);
    let area_diff = abs(area_sub - area_total);

    var arrow_alpha: f32 = 1.0;
    if (area_total > 1e-5) {
        let normalized_diff = area_diff / area_total;
        arrow_alpha = 1.0 - smoothstep(0.0, ARROW_AA, normalized_diff);
    }

    var color = vec3<f32>(0.0);
    var arrow_color = vec3<f32>(0.0);

    // blue line for AllValuesFrom and SomeValuesFrom
    if (in.v_line_type == 3u) {
        color = vec3<f32>(0.4, 0.6, 0.8);
        arrow_color = vec3<f32>(0.4, 0.6, 0.8);
    }

    var inside_arrow = arrow_alpha > 0.0;

    // Make arrow white with black border for type 1
    if (in.v_line_type == 1u && inside_arrow) {
        // barycentric area weights
        let w0 = tri_area_fast(px, left, right) / area_total;
        let w1 = tri_area_fast(px, right, tip) / area_total;
        let w2 = tri_area_fast(px, tip, left) / area_total;

        let edge_dist = min(min(w0, w1), w2);
        let edge_thickness = 2.0;
        // normalized edge distance based on triangle size
        let border_smooth = smoothstep(0.0, edge_thickness / ARROW_WIDTH_PX, edge_dist);
        arrow_color = mix(vec3<f32>(0.0), vec3<f32>(1.0), border_smooth);
    } else if (in.v_line_type == 4u) {
        // diamond (two triangles) simplified: compute diamond centers and use same area test
        let diamond_width_px = ARROW_WIDTH_PX + 5.0;
        let diamond_length_px = ARROW_LENGTH_PX * 2.0;

        let diamond_tip = tip;
        let diamond_center = tip - dir * diamond_length_px * 0.5;
        let diamond_back = tip - dir * diamond_length_px;
        let perp_d = vec2<f32>(-dir.y, dir.x);
        let diamond_left = diamond_center + perp_d * diamond_width_px * 0.5;
        let diamond_right = diamond_center - perp_d * diamond_width_px * 0.5;

        // front triangle (tip, left, right)
        let area1_total = tri_area_fast(diamond_tip, diamond_left, diamond_right);
        let area1_sub = tri_area_fast(diamond_left, diamond_right, px) + tri_area_fast(diamond_tip, px, diamond_right) + tri_area_fast(diamond_tip, diamond_left, px);
        let area1_diff = abs(area1_sub - area1_total);

        // back triangle (back, left, right)
        let area2_total = tri_area_fast(diamond_back, diamond_left, diamond_right);
        let area2_sub = tri_area_fast(diamond_left, diamond_right, px) + tri_area_fast(diamond_back, px, diamond_right) + tri_area_fast(diamond_back, diamond_left, px);
        let area2_diff = abs(area2_sub - area2_total);

        var diamond_alpha: f32 = 0.0;
        if (area1_total > 1e-5) {
            let norm_diff1 = area1_diff / area1_total;
            diamond_alpha = max(diamond_alpha, 1.0 - smoothstep(0.0, ARROW_AA, norm_diff1));
        }
        if (area2_total > 1e-5) {
            let norm_diff2 = area2_diff / area2_total;
            diamond_alpha = max(diamond_alpha, 1.0 - smoothstep(0.0, ARROW_AA, norm_diff2));
        }

        if (diamond_alpha > 0.0) {
            // distance to diamond edges
            let dist1 = point_to_segment_dist(px, diamond_tip, diamond_left);
            let dist2 = point_to_segment_dist(px, diamond_tip, diamond_right);
            let dist3 = point_to_segment_dist(px, diamond_left, diamond_back);
            let dist4 = point_to_segment_dist(px, diamond_right, diamond_back);
            let min_edge_dist = min(min(dist1, dist2), min(dist3, dist4));

            let edge_thickness = 2.0;
            let border_smooth = smoothstep(0.0, edge_thickness, min_edge_dist);
            arrow_color = mix(vec3<f32>(0.0), vec3<f32>(1.0), border_smooth);
            arrow_alpha = diamond_alpha;
            inside_arrow = true;
        }
    }

    // Blend between arrow and line colors
    if (inside_arrow) {
        color = mix(color, arrow_color, arrow_alpha);
    }

    // Final composited alpha
    let final_alpha = max(line_alpha, arrow_alpha);
    return vec4<f32>(color, final_alpha);
}