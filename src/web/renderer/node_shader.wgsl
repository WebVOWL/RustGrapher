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
const BACKGROUND_COLOR = vec3<f32>(0.84, 0.87, 0.88);
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

    // blend smoothly: background -> border -> fill
    var col = mix(BACKGROUND_COLOR, border_color, border_mask);
    col = mix(col, fill_color, fill_mask);

    // blend smoothly: background -> border -> fill
    col = mix(BACKGROUND_COLOR, border_color, border_mask);
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

    // blend smoothly: background -> border -> fill
    col = mix(BACKGROUND_COLOR, border_color, border_mask);
    col = mix(col, fill_color, fill_mask);

    // smooth alpha (fill + border)
    let alpha = clamp(fill_mask + border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_thing(v_uv: vec2<f32>)  -> vec4<f32> {
    let d = distance(v_uv, vec2<f32>(0.5, 0.5));
    let r = 0.43;
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

    // blend smoothly: background -> border -> fill
    var col = mix(BACKGROUND_COLOR, border_color, border_mask);
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

    // blend smoothly: background -> border -> fill
    var col = mix(BACKGROUND_COLOR, border_color, outer_border_mask);
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

    // blend smoothly: background -> border -> fill
    var col = mix(BACKGROUND_COLOR, border_color, border_mask);
    col = mix(col, fill_color, fill_mask);

    // blend smoothly: background -> border -> fill
    col = mix(BACKGROUND_COLOR, border_color, border_mask);
    col = mix(col, fill_color, fill_mask);

    // smooth alpha (fill + border)
    let alpha = clamp(fill_mask + border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_anonymous_class(v_uv: vec2<f32>)  -> vec4<f32> {
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

    let fill_color = vec3<f32>(0.40724, 0.60383, 1.0);

    let border_color = vec3(0.0, 0.0, 0.0);

    // blend smoothly: background -> border -> fill
    var col = mix(BACKGROUND_COLOR, border_color, border_mask);
    col = mix(col, fill_color, fill_mask);

    // smooth alpha (fill + border)
    let alpha = clamp(fill_mask + border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_literal(v_uv: vec2<f32>)  -> vec4<f32> {
    let rect_center = vec2<f32>(0.5, 0.5);
    let rect_size = vec2(0.9, 0.25);
    let dot_count_rect = 11.0;
    let dot_radius_rect = 0.3;
    let fill_color = vec3<f32>(1.0, 0.6038, 0.0331);
    let border_color = vec3<f32>(0.0);
    let border_thickness_rect = 0.02;

    let p = v_uv - rect_center;

    let half_size = 0.5 * rect_size;

    let inside_x = abs(p.x) <= half_size.x;
    let inside_y = abs(p.y) <= half_size.y;

    let inside_rect = inside_x && inside_y;

    let inside_inner = abs(p.x) <= half_size.x - border_thickness_rect && abs(p.y) <= half_size.y - border_thickness_rect;

    // mask selection
    var fill_mask = 0.0;
    if(inside_inner) {
        fill_mask = 1.0;
    }
    var border_mask = 0.0;
    if(inside_rect && !inside_inner) {
        border_mask = 1.0;
    }

    // perimeter coordinate
    let width = 2.0 * half_size.x;
    let height = 2.0 * half_size.y;
    let perim = 2.0 * (width + height);

    // nearest boundary point
    var bp = p;
    if(abs(p.x) > half_size.x - border_thickness_rect && abs(p.x) > abs(p.y)) {
        bp.x = sign(p.x) * half_size.x;
    } else if(abs(p.y) > half_size.y - border_thickness_rect) {
        bp.y = sign(p.y) * half_size.y;
    }

    // convert boundary point to perimeter offset
    var offset = 0.0;
    if(abs(bp.y - half_size.y) < 0.0001) {         // top
        offset = bp.x + half_size.x;
    } else if(abs(bp.x - half_size.x) < 0.0001) {  // right
        offset = width + (half_size.y - bp.y);
    } else if(abs(bp.y + half_size.y) < 0.0001) {  // bottom
        offset = width + height + (half_size.x - bp.x);
    } else {                                       // left
        offset = width * 2 + height + (bp.y + half_size.y);
    }

    let t = (offset / perim) % 1.0;

    // dot pattern along perimeter
    let seg = fract(t * dot_count_rect);
    let dist_to_dot = abs(seg - 0.5);
    let dot_mask = 1.0 - smoothstep(dot_radius_rect - EDGE_SOFTNESS, dot_radius_rect + EDGE_SOFTNESS, dist_to_dot);

    // apply to border only
    border_mask *= dot_mask;

    // composite
    var col = BACKGROUND_COLOR;
    col = mix(col, border_color, border_mask);
    col = mix(col, fill_color, fill_mask);

    return vec4<f32>(col, 1.0);
}

fn draw_rdfs_class(v_uv: vec2<f32>)  -> vec4<f32> {
    let d = distance(v_uv, vec2<f32>(0.5, 0.5));
    let r = 0.48;
    // smooth fill mask (circle inside without border)
    var fill_mask = 1.0 - smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + EDGE_SOFTNESS, d);

    // smooth border mask (ring around circle)
    var border_mask = smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + EDGE_SOFTNESS, d)
                    * (1.0 - smoothstep(r, r + EDGE_SOFTNESS, d));

    let fill_color = vec3<f32>(0.604, 0.3185, 0.604);

    let border_color = vec3<f32>(0.0, 0.0, 0.0);

    // blend smoothly: background -> border -> fill
    var col = mix(BACKGROUND_COLOR, border_color, border_mask);
    col = mix(col, fill_color, fill_mask);

    // blend smoothly: background -> border -> fill
    col = mix(BACKGROUND_COLOR, border_color, border_mask);
    col = mix(col, fill_color, fill_mask);

    // smooth alpha (fill + border)
    let alpha = clamp(fill_mask + border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_rdfs_resource(v_uv: vec2<f32>)  -> vec4<f32> {
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

    let fill_color = vec3<f32>(0.604, 0.3185, 0.604);

    let border_color = vec3(0.0, 0.0, 0.0);

    // blend smoothly: background -> border -> fill
    var col = mix(BACKGROUND_COLOR, border_color, border_mask);
    col = mix(col, fill_color, fill_mask);

    // smooth alpha (fill + border)
    let alpha = clamp(fill_mask + border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
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