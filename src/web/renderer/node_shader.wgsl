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
// colors
const BACKGROUND_COLOR = vec3<f32>(0.93, 0.94, 0.95);
const LIGHT_BLUE = vec3<f32>(0.67, 0.8, 1.0);
const DARK_BLUE = vec3<f32>(0.2, 0.4, 0.8);
const RDFS_COLOR = vec3<f32>(0.8, 0.6, 0.8);
const LITERAL_COLOR = vec3<f32>(1.0, 0.8, 0.2);
const BORDER_COLOR = vec3<f32>(0.0);
const DEPRECATED_COLOR = vec3<f32>(0.6038);
const SET_COLOR = vec3<f32>(0.4, 0.6, 0.8);

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

    let fill_color = LIGHT_BLUE;

    // blend smoothly: background -> border -> fill
    var col = mix(BACKGROUND_COLOR, BORDER_COLOR, border_mask);
    col = mix(col, fill_color, fill_mask);

    // smooth alpha (fill + border)
    let alpha = clamp(fill_mask + border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_external_class(v_uv: vec2<f32>) -> vec4<f32> {
    let d = distance(v_uv, vec2<f32>(0.5, 0.5));
    let r = 0.48;
    // smooth fill mask (circle inside without border)
    var fill_mask = 1.0 - smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + EDGE_SOFTNESS, d);

    // smooth border mask (ring around circle)
    var border_mask = smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + EDGE_SOFTNESS, d)
                    * (1.0 - smoothstep(r, r + EDGE_SOFTNESS, d));

    let fill_color = DARK_BLUE;
    var col: vec3<f32>;

    // blend smoothly: background -> border -> fill
    col = mix(BACKGROUND_COLOR, BORDER_COLOR, border_mask);
    col = mix(col, fill_color, fill_mask);

    // smooth alpha (fill + border)
    let alpha = clamp(fill_mask + border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_thing(v_uv: vec2<f32>) -> vec4<f32> {
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

    let fill_color = vec3<f32>(1.0);

    // blend smoothly: background -> border -> fill
    var col = mix(BACKGROUND_COLOR, BORDER_COLOR, border_mask);
    col = mix(col, fill_color, fill_mask);

    // smooth alpha (fill + border)
    let alpha = clamp(fill_mask + border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_equivalent_class(v_uv: vec2<f32>) -> vec4<f32> {
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

    let fill_color = LIGHT_BLUE;

    var col = mix(BACKGROUND_COLOR, BORDER_COLOR, outer_border_mask);
    col = mix(col, BORDER_COLOR, inner_border_mask);
    col = mix(col, fill_color, fill_mask);

    // smooth alpha (fill + border)
    let alpha = clamp(fill_mask + outer_border_mask + inner_border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_union(v_uv: vec2<f32>) -> vec4<f32> {
    let r = 0.48;
    let border_gap = 0.35;

    // outer circle
    let d_outer = distance(v_uv, vec2<f32>(0.5, 0.5));

    // radius of the inner border
    let inner_border_outer_r = r - BORDER_THICKNESS * 0.7 - border_gap;
    let inner_border_inner_r = inner_border_outer_r - BORDER_THICKNESS * 0.7;

    // positions for two overlapping inner circles
    let offset = 0.05;
    let c1 = vec2<f32>(0.5 - offset, 0.5);
    let c2 = vec2<f32>(0.5 + offset, 0.5);

    let d1 = distance(v_uv, c1);
    let d2 = distance(v_uv, c2);

    // borders
    let inner_border_1 =
        smoothstep(inner_border_inner_r, inner_border_inner_r + EDGE_SOFTNESS, d1) *
        (1.0 - smoothstep(inner_border_outer_r, inner_border_outer_r + EDGE_SOFTNESS, d1));

    let inner_border_2 =
        smoothstep(inner_border_inner_r, inner_border_inner_r + EDGE_SOFTNESS, d2) *
        (1.0 - smoothstep(inner_border_outer_r, inner_border_outer_r + EDGE_SOFTNESS, d2));

    // Combine borders additively
    let inner_border_mask = min(inner_border_1 + inner_border_2, 1.0);

    // outer region
    let outer_fill_mask =
        1.0 - smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + EDGE_SOFTNESS, d_outer);

    let outer_border_mask =
        smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + EDGE_SOFTNESS, d_outer) *
        (1.0 - smoothstep(r, r + EDGE_SOFTNESS, d_outer));

    // inner circle masks
    let inner_fill_1 = 1.0 - smoothstep(inner_border_inner_r, inner_border_inner_r + EDGE_SOFTNESS, d1);
    let inner_fill_2 = 1.0 - smoothstep(inner_border_inner_r, inner_border_inner_r + EDGE_SOFTNESS, d2);

    // Combine fills, but reduce intensity where borders exist
    let inner_fill_combined = max(inner_fill_1, inner_fill_2);
    let inner_fill_mask = inner_fill_combined * (1.0 - inner_border_mask);

    // colors
    let inner_fill_color = SET_COLOR;
    let outer_fill_color = LIGHT_BLUE;

    // layering
    var col = mix(BACKGROUND_COLOR, BORDER_COLOR, outer_border_mask);
    col = mix(col, outer_fill_color, outer_fill_mask);
    col = mix(col, BORDER_COLOR, inner_border_mask);
    col = mix(col, inner_fill_color, inner_fill_mask);

    // alpha
    let alpha = clamp(inner_fill_mask + outer_fill_mask + inner_border_mask + outer_border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_intersection_of(v_uv: vec2<f32>)  -> vec4<f32> {
    let r = 0.48;
    let border_gap = 0.35;

    // outer circle
    let d_outer = distance(v_uv, vec2<f32>(0.5, 0.5));

    // radius of the inner border
    let inner_border_outer_r = r - BORDER_THICKNESS * 0.7 - border_gap;
    let inner_border_inner_r = inner_border_outer_r - BORDER_THICKNESS * 0.7;

    // positions for two overlapping inner circles
    let offset = 0.05;
    let c1 = vec2<f32>(0.5 - offset, 0.5);
    let c2 = vec2<f32>(0.5 + offset, 0.5);
    let d1 = distance(v_uv, c1);
    let d2 = distance(v_uv, c2);

    // borders
    let inner_border_1 =
        smoothstep(inner_border_inner_r, inner_border_inner_r + EDGE_SOFTNESS, d1) *
        (1.0 - smoothstep(inner_border_outer_r, inner_border_outer_r + EDGE_SOFTNESS, d1));
    let inner_border_2 =
        smoothstep(inner_border_inner_r, inner_border_inner_r + EDGE_SOFTNESS, d2) *
        (1.0 - smoothstep(inner_border_outer_r, inner_border_outer_r + EDGE_SOFTNESS, d2));

    // Combine borders additively
    let inner_border_mask = min(inner_border_1 + inner_border_2, 1.0);

    // outer region
    let outer_fill_mask =
        1.0 - smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + EDGE_SOFTNESS, d_outer);
    let outer_border_mask =
        smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + EDGE_SOFTNESS, d_outer) *
        (1.0 - smoothstep(r, r + EDGE_SOFTNESS, d_outer));

    // inner circle masks
    let inner_fill_1 = 1.0 - smoothstep(inner_border_inner_r, inner_border_inner_r + EDGE_SOFTNESS, d1);
    let inner_fill_2 = 1.0 - smoothstep(inner_border_inner_r, inner_border_inner_r + EDGE_SOFTNESS, d2);
    
    // Calculate overlapping region (where both circles are filled)
    let overlap_mask = min(inner_fill_1, inner_fill_2);
    
    // Calculate non-overlapping regions (exclusive OR)
    let non_overlap_mask = max(inner_fill_1, inner_fill_2) - overlap_mask;
    
    // Reduce fill intensity where borders exist
    let overlap_final = overlap_mask * (1.0 - inner_border_mask);
    let non_overlap_final = non_overlap_mask * (1.0 - inner_border_mask);
    
    // colors
    let inner_fill_color = SET_COLOR;
    let outer_fill_color = LIGHT_BLUE;
    // layering
    var col = mix(BACKGROUND_COLOR, BORDER_COLOR, outer_border_mask);
    col = mix(col, outer_fill_color, outer_fill_mask);
    col = mix(col, BORDER_COLOR, inner_border_mask);
    col = mix(col, outer_fill_color, non_overlap_final);
    col = mix(col, inner_fill_color, overlap_final);
    // alpha
    let alpha = clamp(overlap_final + non_overlap_final + outer_fill_mask + inner_border_mask + outer_border_mask, 0.0, 1.0);
    return vec4<f32>(col, alpha);
}

fn draw_complement(v_uv: vec2<f32>) -> vec4<f32> {
    let d = distance(v_uv, vec2<f32>(0.5, 0.5));
    let r = 0.48;
    let border_gap = 0.35;

    // radius of the inner border
    let inner_border_outer_r = r - BORDER_THICKNESS * 0.8 - border_gap;
    let inner_border_inner_r = inner_border_outer_r - BORDER_THICKNESS * 0.8;

    // masks
    let inner_fill_mask = 1.0 - smoothstep(inner_border_inner_r, inner_border_inner_r + EDGE_SOFTNESS, d);

    // solid outer fill area between inner and outer borders
    let outer_fill_mask =
        smoothstep(inner_border_outer_r, inner_border_outer_r + EDGE_SOFTNESS, d) *
        (1.0 - smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + EDGE_SOFTNESS, d));

    // borders
    let inner_border_mask =
        smoothstep(inner_border_inner_r, inner_border_inner_r + EDGE_SOFTNESS, d) *
        (1.0 - smoothstep(inner_border_outer_r, inner_border_outer_r + EDGE_SOFTNESS, d));

    let outer_border_mask =
        smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + EDGE_SOFTNESS, d) *
        (1.0 - smoothstep(r, r + EDGE_SOFTNESS, d));

    // colors
    let inner_fill_color = SET_COLOR;
    let outer_fill_color = LIGHT_BLUE;

    // layering
    var col = mix(BACKGROUND_COLOR, BORDER_COLOR, outer_border_mask);
    col = mix(col, outer_fill_color, outer_fill_mask);
    col = mix(col, BORDER_COLOR, inner_border_mask);
    col = mix(col, inner_fill_color, inner_fill_mask);

    // combined alpha
    let alpha = clamp(inner_fill_mask + outer_fill_mask + inner_border_mask + outer_border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_deprecated_class(v_uv: vec2<f32>) -> vec4<f32> {
    let d = distance(v_uv, vec2<f32>(0.5, 0.5));
    let r = 0.48;
    // smooth fill mask (circle inside without border)
    var fill_mask = 1.0 - smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + EDGE_SOFTNESS, d);

    // smooth border mask (ring around circle)
    var border_mask = smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + EDGE_SOFTNESS, d)
                    * (1.0 - smoothstep(r, r + EDGE_SOFTNESS, d));

    let fill_color = DEPRECATED_COLOR;

    // blend smoothly: background -> border -> fill
    var col = mix(BACKGROUND_COLOR, BORDER_COLOR, border_mask);
    col = mix(col, fill_color, fill_mask);

    // smooth alpha (fill + border)
    let alpha = clamp(fill_mask + border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_anonymous_class(v_uv: vec2<f32>) -> vec4<f32> {
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

    let fill_color = LIGHT_BLUE;

    // blend smoothly: background -> border -> fill
    var col = mix(BACKGROUND_COLOR, BORDER_COLOR, border_mask);
    col = mix(col, fill_color, fill_mask);

    // smooth alpha (fill + border)
    let alpha = clamp(fill_mask + border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_literal(v_uv: vec2<f32>) -> vec4<f32> {
    let rect_center = vec2<f32>(0.5, 0.5);
    let rect_size = vec2(0.9, 0.25);
    let dot_count_rect = 11.0;
    let dot_radius_rect = 0.3;
    let fill_color = LITERAL_COLOR;
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
    col = mix(col, BORDER_COLOR, border_mask);
    col = mix(col, fill_color, fill_mask);

    let alpha = clamp(fill_mask + border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_rdfs_class(v_uv: vec2<f32>) -> vec4<f32> {
    let d = distance(v_uv, vec2<f32>(0.5, 0.5));
    let r = 0.48;
    // smooth fill mask (circle inside without border)
    var fill_mask = 1.0 - smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + EDGE_SOFTNESS, d);

    // smooth border mask (ring around circle)
    var border_mask = smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + EDGE_SOFTNESS, d)
                    * (1.0 - smoothstep(r, r + EDGE_SOFTNESS, d));

    let fill_color = RDFS_COLOR;

    // blend smoothly: background -> border -> fill
    var col = mix(BACKGROUND_COLOR, BORDER_COLOR, border_mask);
    col = mix(col, fill_color, fill_mask);

    // smooth alpha (fill + border)
    let alpha = clamp(fill_mask + border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_rdfs_resource(v_uv: vec2<f32>) -> vec4<f32> {
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

    let fill_color = RDFS_COLOR;

    // blend smoothly: background -> border -> fill
    var col = mix(BACKGROUND_COLOR, BORDER_COLOR, border_mask);
    col = mix(col, fill_color, fill_mask);

    // smooth alpha (fill + border)
    let alpha = clamp(fill_mask + border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_datatype(v_uv: vec2<f32>) -> vec4<f32> {
    let rect_center = vec2<f32>(0.5, 0.5);
    let rect_size = vec2(0.9, 0.25);
    let fill_color = LITERAL_COLOR;
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

    // composite
    var col = BACKGROUND_COLOR;
    col = mix(col, BORDER_COLOR, border_mask);
    col = mix(col, fill_color, fill_mask);

    let alpha = clamp(fill_mask + border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_node_by_type(node_type: u32, v_uv: vec2<f32>) -> vec4<f32> {
    switch node_type {
        case 0: {return draw_class(v_uv);}
        case 1: {return draw_external_class(v_uv);}
        case 2: {return draw_thing(v_uv);}
        case 3: {return draw_equivalent_class(v_uv);}
        case 4: {return draw_union(v_uv);}
        case 5: {return draw_union(v_uv);}
        case 6: {return draw_intersection_of(v_uv);}
        case 7: {return draw_complement(v_uv);}
        case 8: {return draw_deprecated_class(v_uv);}
        case 9: {return draw_anonymous_class(v_uv);}
        case 10: {return draw_literal(v_uv);}
        case 11: {return draw_rdfs_class(v_uv);}
        case 12: {return draw_rdfs_resource(v_uv);}
        case 13: {return draw_datatype(v_uv);}
        default: {return draw_class(v_uv);}
    }
}