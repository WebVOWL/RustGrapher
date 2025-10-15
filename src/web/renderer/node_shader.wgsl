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
const NODE_RADIUS_PIX = 48.0; // pixels

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

    out.node_type = in.node_type;

    return out;
}

// parameters
const BORDER_THICKNESS = 0.03;   // how thick the border ring is
const EDGE_SOFTNESS    = 0.02;   // anti-aliasing
// polar angle based repeating pattern (dotted border)
const PI = 3.14159265;
const DOT_COUNT = 14.0;        // number of dots around the ring
const DOT_RADIUS = 0.3;        // half-width of each dot in pattern-space (0..0.5)
const DOT_EDGE = 0.01;         // softness of dot edges

@fragment
fn fs_node_main(in: VertOut) -> @location(0) vec4<f32> {
    // circle distance
    
    let v_uv = in.v_uv;

    return draw_node_by_type(in.node_type, v_uv);
}

fn draw_class(v_uv: vec2<f32>) -> vec4<f32> {
    let d = distance(v_uv, vec2<f32>(0.5, 0.5));
    let r = 0.48;
    // smooth fill mask (circle inside without border)
    var fill_mask = 1.0 - smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + EDGE_SOFTNESS, d);

    // smooth border mask (ring around circle)
    var border_mask = smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + EDGE_SOFTNESS, d)
                    * (1.0 - smoothstep(r, r + EDGE_SOFTNESS, d));

    let fill_color = vec3<f32>(0.40724, 0.60383, 1.0);

    let border_color = vec3<f32>(0.0, 0.0, 0.0);
    let background = vec3<f32>(0.84, 0.87, 0.88);

    // blend smoothly: background -> border -> fill
    var col = mix(background, border_color, border_mask);
    col = mix(col, fill_color, fill_mask);

    // blend smoothly: background -> border -> fill
    col = mix(background, border_color, border_mask);
    col = mix(col, fill_color, fill_mask);

    // smooth alpha (fill + border)
    let alpha = clamp(fill_mask + border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_subclass(v_uv: vec2<f32>)  -> vec4<f32> {
    let d = distance(v_uv, vec2<f32>(0.5, 0.5));
    let r = 0.48;
    // smooth fill mask (circle inside without border)
    var fill_mask = 1.0 - smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + EDGE_SOFTNESS, d);

    // smooth border mask (ring around circle)
    var border_mask = smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + EDGE_SOFTNESS, d)
                    * (1.0 - smoothstep(r, r + EDGE_SOFTNESS, d));

    let fill_color = vec3<f32>(0.03189, 0.13286, 0.60382);
    var col: vec3<f32>;

    let border_color = vec3<f32>(0.0, 0.0, 0.0);
    let background = vec3<f32>(0.84, 0.87, 0.88);

    // blend smoothly: background -> border -> fill
    col = mix(background, border_color, border_mask);
    col = mix(col, fill_color, fill_mask);

    // smooth alpha (fill + border)
    let alpha = clamp(fill_mask + border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_thing(v_uv: vec2<f32>)  -> vec4<f32> {
    let d = distance(v_uv, vec2<f32>(0.5, 0.5));
    let r = 0.48;
    // smooth fill mask (circle inside without border)
    var fill_mask = 1.0 - smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + EDGE_SOFTNESS, d);

    // smooth border mask (ring around circle)
    var border_mask = smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + EDGE_SOFTNESS, d)
                    * (1.0 - smoothstep(r, r + EDGE_SOFTNESS, d));

    let center = vec2(0.5, 0.5);
    let dir = v_uv - center;
    let angle = atan2(dir.y, dir.x);          // -PI..PI
    let ang01 = angle / (2.0 * PI) + 0.5;     // 0..1
    let p = fract(ang01 * DOT_COUNT);         // 0..1 per dot segment
    let distToDot = abs(p - 0.5);             // distance from center of dot segment
    let dot_mask = 1.0 - smoothstep(DOT_RADIUS - DOT_EDGE, DOT_RADIUS + DOT_EDGE, distToDot);

    // apply dot mask to border mask so the ring becomes dotted
    border_mask *= dot_mask;

    let fill_color = vec3<f32>(1.0, 1.0, 1.0);

    let border_color = vec3(0.0, 0.0, 0.0);
    let background = vec3(0.84, 0.87, 0.88);

    // blend smoothly: background -> border -> fill
    var col = mix(background, border_color, border_mask);
    col = mix(col, fill_color, fill_mask);

    // smooth alpha (fill + border)
    let alpha = clamp(fill_mask + border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_equivalent_class(v_uv: vec2<f32>)  -> vec4<f32> {
    let d = distance(v_uv, vec2<f32>(0.5, 0.5));
    let r = 0.48;

    let border_gap = 0.02;

    // radius of the inner border
    let inner_border_outer_r = r - BORDER_THICKNESS - border_gap;
    let inner_border_inner_r = inner_border_outer_r - BORDER_THICKNESS;

    // fill mask (everything inside the inner border)
    let fill_mask = 1.0 - smoothstep(inner_border_inner_r, inner_border_inner_r + EDGE_SOFTNESS, d);

    // inner border (fully opaque)
    let inner_border_mask = smoothstep(inner_border_inner_r, inner_border_inner_r + EDGE_SOFTNESS, d) * (1.0 - smoothstep(inner_border_outer_r, inner_border_outer_r + EDGE_SOFTNESS, d));

    // outer border
    let outer_border_mask = smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + EDGE_SOFTNESS, d) *
        (1.0 - smoothstep(r, r + EDGE_SOFTNESS, d));

    let fill_color = vec3<f32>(0.40724, 0.60383, 1.0);

    let border_color = vec3<f32>(0.0, 0.0, 0.0);
    let background = vec3<f32>(0.84, 0.87, 0.88);

    // blend smoothly: background -> border -> fill
    var col = mix(background, border_color, outer_border_mask);
    col = mix(col, border_color, inner_border_mask);
    col = mix(col, fill_color, fill_mask);

    // smooth alpha (fill + border)
    let alpha = clamp(fill_mask + outer_border_mask + inner_border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_disjoint_union(v_uv: vec2<f32>)  -> vec4<f32> {
    return vec4<f32>(0.0); // TODO: implement
}

fn draw_intersection_of(v_uv: vec2<f32>)  -> vec4<f32> {
    return vec4<f32>(0.0); // TODO: implement
}

fn draw_complement(v_uv: vec2<f32>)  -> vec4<f32> {
    return vec4<f32>(0.0); // TODO: implement
}

fn draw_deprecated_class(v_uv: vec2<f32>)  -> vec4<f32> {
    let d = distance(v_uv, vec2<f32>(0.5, 0.5));
    let r = 0.48;
    // smooth fill mask (circle inside without border)
    var fill_mask = 1.0 - smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + EDGE_SOFTNESS, d);

    // smooth border mask (ring around circle)
    var border_mask = smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + EDGE_SOFTNESS, d)
                    * (1.0 - smoothstep(r, r + EDGE_SOFTNESS, d));

    let fill_color = vec3<f32>(0.6038);

    let border_color = vec3<f32>(0.0, 0.0, 0.0);
    let background = vec3<f32>(0.84, 0.87, 0.88);

    // blend smoothly: background -> border -> fill
    var col = mix(background, border_color, border_mask);
    col = mix(col, fill_color, fill_mask);

    // blend smoothly: background -> border -> fill
    col = mix(background, border_color, border_mask);
    col = mix(col, fill_color, fill_mask);

    // smooth alpha (fill + border)
    let alpha = clamp(fill_mask + border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_anonymous_class(v_uv: vec2<f32>)  -> vec4<f32> {
    return vec4<f32>(0.0); // TODO: implement
}

fn draw_literal(v_uv: vec2<f32>)  -> vec4<f32> {
    return vec4<f32>(0.0); // TODO: implement
}

fn draw_rdfs_class(v_uv: vec2<f32>)  -> vec4<f32> {
    return vec4<f32>(0.0); // TODO: implement
}

fn draw_rdfs_resource(v_uv: vec2<f32>)  -> vec4<f32> {
    return vec4<f32>(0.0); // TODO: implement
}

fn draw_node_by_type(node_type: u32, v_uv: vec2<f32>) -> vec4<f32> {
    switch node_type {
        case 0: {
            return draw_class(v_uv);
        }
        case 1: {
            return draw_subclass(v_uv);
        }
        case 2: {
            return draw_thing(v_uv);
        }
        case 3: {
            return draw_equivalent_class(v_uv);
        }
        case 4: {
            return draw_disjoint_union(v_uv);
        }
        case 5: {
            return draw_intersection_of(v_uv);
        }
        case 6: {
            return draw_complement(v_uv);
        }
        case 7: {
            return draw_deprecated_class(v_uv);
        }
        case 8: {
            return draw_anonymous_class(v_uv);
        }
        case 9: {
            return draw_literal(v_uv);
        }
        case 10: {
            return draw_rdfs_class(v_uv);
        }
        case 11: {
            return draw_rdfs_resource(v_uv);
        }
        default: {
            return vec4(0.0);
        }
    }
}