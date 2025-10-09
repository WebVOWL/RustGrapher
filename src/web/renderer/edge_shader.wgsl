struct VertIn {
    @location(0) quad_pos: vec2<f32>, // [-1..1] corners of unit quad
    @location(1) start: vec2<f32>,    // start of line (in px)
    @location(2) end: vec2<f32>,      // end of line (in px)
};

struct VertOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) v_uv: vec2<f32>,
    @location(1) v_dir: vec2<f32>,
    @location(2) v_length: f32,
};

@group(0) @binding(0)
var<uniform> u_resolution: vec4<f32>; // xy = screen size

const LINE_THICKNESS = 3.0; // pixels
const AA_SOFTNESS = 1.5;    // pixels

@vertex
fn vs_edge_main(in: VertIn) -> VertOut {
    var out: VertOut;

    let dir = in.end - in.start;
    let length = length(dir);
    // guard against zero-length edges
    if (length <= 0.0001) {
        // collapse to start pos to avoid NaNs
        let px = in.start;
        let ndc_x = (px.x / u_resolution.x) * 2.0 - 1.0;
        let ndc_y = (px.y / u_resolution.y) * 2.0 - 1.0;
        out.clip_position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
        out.v_uv = in.quad_pos * 0.5 + vec2<f32>(0.5, 0.5);
        out.v_dir = vec2<f32>(0.0, 1.0);
        out.v_length = 0.0;
        return out;
    }

    let dir_norm = dir / length;
    let perp = vec2<f32>(-dir_norm.y, dir_norm.x);

    // shrink the quad along the line by half the thickness so endpoints don't overhang
    let half_len = 0.5 * (length - LINE_THICKNESS);
    // if line is shorter than thickness, fall back to tiny center quad
    let effective_half_len = max(0.0, half_len);

    // map quad_pos.x [-1..1] to [-effective_half_len..+effective_half_len]
    // and quad_pos.y [-1..1] to [-thickness/2..+thickness/2]
    let offset_px =
        dir_norm * (in.quad_pos.x * effective_half_len) +
        perp * (in.quad_pos.y * (LINE_THICKNESS * 0.5));

    let center = (in.start + in.end) * 0.5;
    let pos_px = center + offset_px;

    let ndc_x = (pos_px.x / u_resolution.x) * 2.0 - 1.0;
    let ndc_y = (pos_px.y / u_resolution.y) * 2.0 - 1.0;

    out.clip_position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.v_uv = in.quad_pos * 0.5 + vec2<f32>(0.5, 0.5);
    out.v_dir = dir_norm;
    out.v_length = length;

    return out;
}

@fragment
fn fs_edge_main(in: VertOut) -> @location(0) vec4<f32> {
    // distance from centerline in pixel units
    let dist_from_center = abs(in.v_uv.y - 0.5) * 2.0 * LINE_THICKNESS;

    // alpha falloff using smoothstep for anti-aliasing
    let alpha = 1.0 - smoothstep(LINE_THICKNESS - AA_SOFTNESS,
                                 LINE_THICKNESS,
                                 dist_from_center);

    let color = vec3<f32>(0.0, 0.0, 0.0); // black line
    return vec4<f32>(color, alpha);
}
