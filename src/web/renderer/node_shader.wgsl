struct VertIn {
    @location(0) quad_pos: vec2<f32>, // [-1..1] quad corner in local space
    @location(1) inst_pos: vec2<f32>, // per-instance node position in pixels
    @location(2) node_type: u32, // Type of node used when drawing
};

struct VertOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) v_uv: vec2<f32>, // 0..1 inside quad
    @location(1) node_type: u32,
};

@group(0) @binding(0)
var<uniform> u_resolution: vec4<f32>; // xy = pixel resolution

// per-instance radius fixed
const NODE_RADIUS_PIX = 24.0; // pixels

@vertex
fn vs_node_main(
    in: VertIn,
    @builtin(instance_index) instanceIndex: u32,
) -> VertOut {
    var out: VertOut;

    // fetch node position (in pixel coordinates) from per-instance attribute
    let pos_px: vec2<f32> = in.inst_pos;

    // quad_pos is [-1..1] so convert to offset in pixels
    let offset_px = in.quad_pos * vec2(NODE_RADIUS_PIX);

    // screen position in pixels
    let screen = pos_px + offset_px;

    // convert to NDC clip space: x -> [-1,1] left->right, y -> [-1,1] bottom->top
    let ndc_x = (screen.x / u_resolution.x) * 2.0 - 1.0;
    let ndc_y = (screen.y / u_resolution.y) * 2.0 - 1.0;
    out.clip_position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);

    // uv 0..1 for circle mask; quad_pos [-1..1] -> uv [0..1]
    out.v_uv = in.quad_pos * 0.5 + vec2<f32>(0.5, 0.5);

    out.node_type = in.node_type;

    return out;
}

@fragment
fn fs_node_main(in: VertOut) -> @location(0) vec4<f32> {
    // circle distance
    let d = distance(in.v_uv, vec2<f32>(0.5, 0.5));
    let r = 0.48;

    // parameters
    let border_thickness = 0.05;   // how thick the border ring is
    let edge_softness    = 0.03;   // anti-aliasing

    // smooth fill mask (circle inside without border)
    var fill_mask = 1.0 - smoothstep(r - border_thickness, r - border_thickness + edge_softness, d);

    // smooth border mask (ring around circle)
    var border_mask = smoothstep(r - border_thickness, r - border_thickness + edge_softness, d)
                    * (1.0 - smoothstep(r, r + edge_softness, d));

    // colors
    var fill_color = vec3<f32>(0.40724, 0.60383, 1.0);
    var col: vec3<f32>;

    switch in.node_type {
        case 1: {
            fill_color = vec3<f32>(0.03189, 0.13286, 0.60382);

            let border_color = vec3<f32>(0.0, 0.0, 0.0);
            let background = vec3<f32>(0.84, 0.87, 0.88);

            // blend smoothly: background -> border -> fill
            col = mix(background, border_color, border_mask);
            col = mix(col, fill_color, fill_mask);
        }
        case 2: {
            // polar angle based repeating pattern
            const PI = 3.14159265f;
            const DOT_COUNT = 14.0f;        // number of dots around the ring
            const DOT_RADIUS = 0.3f;        // half-width of each dot in pattern-space (0..0.5)
            const DOT_EDGE = 0.01f;         // softness of dot edges

            let center = vec2(0.5f, 0.5f);
            let dir = in.v_uv - center;
            let angle = atan2(dir.y, dir.x);               // -PI..PI
            let ang01 = angle / (2.0f * PI) + 0.5f;       // 0..1
            let p = fract(ang01 * DOT_COUNT);             // 0..1 per dot segment
            let distToDot = abs(p - 0.5f);                // distance from center of dot segment
            let dot_mask = 1.0f - smoothstep(DOT_RADIUS - DOT_EDGE, DOT_RADIUS + DOT_EDGE, distToDot);

            // apply dot mask to border mask so the ring becomes dotted
            border_mask *= dot_mask;

            fill_color = vec3<f32>(1.0, 1.0, 1.0);

            let border_color = vec3(0.0f, 0.0f, 0.0f);
            let background = vec3(0.84f, 0.87f, 0.88f);

            // blend smoothly: background -> border -> fill
            col = mix(background, border_color, border_mask);
            col = mix(col, fill_color, fill_mask);
        }
        default: {
            let border_color = vec3<f32>(0.0, 0.0, 0.0);
            let background = vec3<f32>(0.84, 0.87, 0.88);

            // blend smoothly: background -> border -> fill
            col = mix(background, border_color, border_mask);
            col = mix(col, fill_color, fill_mask);
        }
    }

    // smooth alpha (fill + border)
    let alpha = clamp(fill_mask + border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}