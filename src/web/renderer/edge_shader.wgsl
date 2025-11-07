struct VertIn {
    @location(0) position: vec2<f32>,        // Position in pixel space
    @location(1) t_param: f32,               // Parameter along curve [0..1]
    @location(2) side: f32,                  // -1 or +1 for left/right
    @location(3) line_type: u32,             // Line style
    @location(4) end_shape_type: u32,        // 0: Circle, 1: Rectangle
    @location(5) end_shape_dim: vec2<f32>,   // Shape dimensions
    @location(6) curve_start: vec2<f32>,     // Start of curve
    @location(7) curve_end: vec2<f32>,       // End of curve
    @location(8) tangent_at_end: vec2<f32>,  // Tangent at t=1
    @location(9) ctrl: vec2<f32>,            // Control point for quadratic Bezier
};

struct VertOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) v_t: f32,
    @location(1) v_side: f32,
    @interpolate(flat) @location(2) v_line_type: u32,
    @interpolate(flat) @location(3) v_end_shape_type: u32,
    @location(4) v_end_shape_dim: vec2<f32>,
    @location(5) v_curve_start: vec2<f32>,
    @location(6) v_curve_end: vec2<f32>,
    @location(7) v_tangent_at_end: vec2<f32>,
    @location(8) v_position_px: vec2<f32>,   // Position in pixel space
    @location(9) v_ctrl: vec2<f32>,          // Control point for quadratic Bezier
};

struct ViewUniforms {
    resolution: vec2<f32>,
    pan: vec2<f32>,
    zoom: f32,
};

@group(0) @binding(0)
var<uniform> u_view: ViewUniforms;

// Arrow constants (pixels)
const ARROW_LENGTH_PX: f32 = 10.0;
const ARROW_WIDTH_PX: f32 = 15.0;
const AA_SOFTNESS: f32 = 2.5;
const DASH_LENGTH_PX: f32 = 20.0;
const LINE_THICKNESS = 1.5;

fn bezier_point(p0: vec2<f32>, ctrl: vec2<f32>, p2: vec2<f32>, t: f32) -> vec2<f32> {
    let t1 = 1.0 - t;
    return t1 * t1 * p0 + 2.0 * t1 * t * ctrl + t * t * p2;
}

@vertex
fn vs_edge_main(in: VertIn) -> VertOut {
    var out: VertOut;

    let world_pos = in.position;

    // View Transform for clip_position
    let world_rel = world_pos - u_view.pan;
    let world_rel_zoomed_px = world_rel * u_view.zoom;
    let screen_center_px = u_view.resolution * 0.5;
    let screen_offset_px = vec2<f32>(world_rel_zoomed_px.x, -world_rel_zoomed_px.y);
    let screen = screen_center_px + screen_offset_px;
    
    let ndc_x = (screen.x / u_view.resolution.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (screen.y / u_view.resolution.y) * 2.0;
    out.clip_position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);

    // Pass vertex attributes
    out.v_t = in.t_param;
    out.v_side = in.side;
    out.v_line_type = in.line_type;
    out.v_end_shape_type = in.end_shape_type;
    out.v_end_shape_dim = in.end_shape_dim;
    out.v_curve_start = in.curve_start;
    out.v_curve_end = in.curve_end;
    out.v_tangent_at_end = in.tangent_at_end;
    out.v_position_px = world_pos; // Pass WORLD position
    out.v_ctrl = in.ctrl;

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

// Fast triangle area via cross product
fn tri_area_fast(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> f32 {
    return abs(((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x))) * 0.5;
}

@fragment
fn fs_edge_main(in: VertOut) -> @location(0) vec4<f32> {
    let px = in.v_position_px; // World space
    let t = in.v_t;
    let tip = in.v_curve_end;
    var dir = normalize(in.v_tangent_at_end);

    // Compute center point on the curve for this fragment's t
    let center_pos = bezier_point(in.v_curve_start, in.v_ctrl, in.v_curve_end, t);
    let dist_to_center = length(px - center_pos);

    // Convert pixel-space constants to world-space
    let line_thickness_world = LINE_THICKNESS * 1.90;
    let aa_softness_world = AA_SOFTNESS / u_view.zoom;

    // AA smoothing across the curve thickness
    let half_thickness_world = line_thickness_world / 2.0;
    var line_alpha = 1.0 - smoothstep(half_thickness_world - aa_softness_world, 
                                    half_thickness_world + aa_softness_world, 
                                    dist_to_center);

    // Calculate physical distance from this fragment to the end point
    let dist_to_end = length(px - tip);

    // Apply dash/dot pattern
    if (in.v_line_type == 1u || in.v_line_type == 2u || in.v_line_type == 4u) {
        let chord_len = length(in.v_curve_end - in.v_curve_start);
        let pattern_scale = max(1.0, chord_len / DASH_LENGTH_PX);
        let dot_fraction = 0.6;
        let pattern_phase = fract(t * pattern_scale);
        let fade = aa_softness_world * 0.025;
        let dot_mask = smoothstep(0.0, fade, pattern_phase) * (1.0 - smoothstep(dot_fraction, dot_fraction + fade, pattern_phase));
        line_alpha *= dot_mask;
    }

    // --- Arrow Drawing Logic ---
    var arrow_alpha: f32 = 0.0;
    var color = vec3<f32>(0.0);
    var arrow_color = vec3<f32>(0.0);
    var inside_arrow = false;
    
    // Blue line for AllValuesFrom and SomeValuesFrom
    if (in.v_line_type == 3u) {
        color = vec3<f32>(0.4, 0.6, 0.8);
        arrow_color = vec3<f32>(0.4, 0.6, 0.8);
    }

    // No arrow for disjoint edges (type 2)
    if (in.v_line_type == 2u) {
        return vec4<f32>(vec3<f32>(0.0), line_alpha);
    }

    if (in.v_line_type == 4u) {
        // Diamond arrow for set operators
        let diamond_width_px = ARROW_WIDTH_PX + 5.0;
        let diamond_length_px = ARROW_LENGTH_PX * 2.0;

        let diamond_tip = tip;
        let diamond_center = tip - dir * diamond_length_px * 0.5;
        let diamond_back = tip - dir * diamond_length_px;
        let perp_d = vec2<f32>(-dir.y, dir.x);
        let diamond_left = diamond_center + perp_d * diamond_width_px * 0.5;
        let diamond_right = diamond_center - perp_d * diamond_width_px * 0.5;

        // Front triangle
        let area1_total = tri_area_fast(diamond_tip, diamond_left, diamond_right);
        let area1_sub = tri_area_fast(diamond_left, diamond_right, px) + 
                        tri_area_fast(diamond_tip, px, diamond_right) + 
                        tri_area_fast(diamond_tip, diamond_left, px);
        let area1_diff = abs(area1_sub - area1_total);

        // Back triangle
        let area2_total = tri_area_fast(diamond_back, diamond_left, diamond_right);
        let area2_sub = tri_area_fast(diamond_left, diamond_right, px) + 
                        tri_area_fast(diamond_back, px, diamond_right) + 
                        tri_area_fast(diamond_back, diamond_left, px);
        let area2_diff = abs(area2_sub - area2_total);

        var diamond_alpha: f32 = 0.0;
        if (area1_total > 1e-5) {
            let norm_diff1 = area1_diff / area1_total;
            diamond_alpha = max(diamond_alpha, 1.0 - smoothstep(0.0, aa_softness_world, norm_diff1));
        }
        if (area2_total > 1e-5) {
            let norm_diff2 = area2_diff / area2_total;
            diamond_alpha = max(diamond_alpha, 1.0 - smoothstep(0.0, aa_softness_world, norm_diff2));
        }

        if (diamond_alpha > 0.0) {
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
    } else {
        // Default triangle arrow
        let base_center = tip - dir * ARROW_LENGTH_PX;
        let perp = vec2<f32>(-dir.y, dir.x);
        let halfw = ARROW_WIDTH_PX * 0.5;
        let left = base_center + perp * halfw;
        let right = base_center - perp * halfw;

        // Use area-based triangle test
        let area_total = tri_area_fast(tip, left, right);
        let area_sub = tri_area_fast(px, left, right) + 
                       tri_area_fast(tip, px, right) + 
                       tri_area_fast(tip, left, px);
        let area_diff = abs(area_sub - area_total);

        if (area_total > 1e-5) {
            let normalized_diff = area_diff / area_total;
            arrow_alpha = 1.0 - smoothstep(0.0, aa_softness_world, normalized_diff);
        }
        inside_arrow = arrow_alpha > 0.0;

        // White arrow with black border for type 1 (SubclassOf)
        if (in.v_line_type == 1u && inside_arrow) {
            let w0 = tri_area_fast(px, left, right) / area_total;
            let w1 = tri_area_fast(px, right, tip) / area_total;
            let w2 = tri_area_fast(px, tip, left) / area_total;

            let edge_dist = min(min(w0, w1), w2);
            let edge_thickness = 2.0;
            let border_smooth = smoothstep(0.0, edge_thickness / ARROW_WIDTH_PX, edge_dist);
            arrow_color = mix(vec3<f32>(0.0), vec3<f32>(1.0), border_smooth);
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