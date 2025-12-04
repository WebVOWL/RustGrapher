struct VertIn {
    @location(0) quad_pos: vec2<f32>,         // [-1..1] quad corner in local space
    @location(1) inst_pos: vec2<f32>,         // per-instance node position in pixels
    @location(2) node_type: u32,              // Type of node used when drawing
    @location(3) shape: u32,                  // The shape of the node, 0: Circle, 1: Rectangle
    @location(4) shape_dimensions: vec2<f32>, // The radius of a circle or the width and height of a rectangle
    @location(5) hovered: u32,                // 1 when hovered, 0 otherwise
};

struct VertOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) v_uv: vec2<f32>, // 0..1 inside quad
    @interpolate(flat) @location(1) v_node_type: u32,
    @interpolate(flat) @location(2) v_shape: u32,
    @location(3) v_shape_dimensions: vec2<f32>,
    @interpolate(flat) @location(4) v_hovered: u32,
};

struct ViewUniforms {
    resolution: vec2<f32>,
    pan: vec2<f32>,
    zoom: f32,
};

@group(0) @binding(0)
var<uniform> u_view: ViewUniforms;

// per-instance radius fixed
const NODE_RADIUS_PIX = 50.0; // pixels

@vertex
fn vs_node_main(
    in: VertIn,
    @builtin(instance_index) instanceIndex: u32,
) -> VertOut {
    var out: VertOut;

    // compute non-uniform scale for shape geometry
    var scale_xy = vec2<f32>(in.shape_dimensions.x, in.shape_dimensions.y);
    if (in.shape == 0u) {
        // circle -> use same x and y
        scale_xy = vec2<f32>(in.shape_dimensions.x, in.shape_dimensions.x);
    }

    // World-space point 
    let world_pos = in.inst_pos + in.quad_pos * (NODE_RADIUS_PIX * scale_xy);

    // View Transform
    let world_rel = world_pos - u_view.pan;
    let world_rel_zoomed_px = world_rel * u_view.zoom;
    let screen_center_px = u_view.resolution * 0.5;
    let screen_offset_px = vec2<f32>(world_rel_zoomed_px.x, -world_rel_zoomed_px.y);
    let screen = screen_center_px + screen_offset_px;
    
    let ndc_x = (screen.x / u_view.resolution.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (screen.y / u_view.resolution.y) * 2.0;
    out.clip_position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);

    let aspect = vec2<f32>(in.quad_pos.x, in.quad_pos.y);
    out.v_uv = aspect * 0.5 + vec2<f32>(0.5, 0.5);

    out.v_node_type = in.node_type;
    out.v_shape = in.shape;
    out.v_shape_dimensions = in.shape_dimensions;
    out.v_hovered = in.hovered;

    return out;
}

// parameters
const BORDER_THICKNESS = 0.03;   // how thick the border ring is
const EDGE_SOFTNESS    = 0.02;   // anti-aliasing
const RECT_SCALE = vec2(0.9, 0.25);
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
const DATATYPE_PROPERTY_COLOR = vec3<f32>(0.6039, 0.7960, 0.4039);
const HIGHLIGHTED_COLOR = vec3<f32>(1.0, 0.0, 0.0);

@fragment
fn fs_node_main(in: VertOut) -> @location(0) vec4<f32> {
    return draw_node_by_type(in.v_node_type, in.v_uv, in.v_shape_dimensions, in.v_hovered);
}

fn draw_class(v_uv: vec2<f32>, hovered: u32) -> vec4<f32> {
    let d = distance(v_uv, vec2<f32>(0.5, 0.5));
    let r = 0.48;
    let aa_softness_world = EDGE_SOFTNESS / u_view.zoom;
    // smooth fill mask (circle inside without border)
    var fill_mask = 1.0 - smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + aa_softness_world, d);

    // smooth border mask (ring around circle)
    var border_mask = smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + aa_softness_world, d)
                    * (1.0 - smoothstep(r, r + aa_softness_world, d));

    var fill_color = LIGHT_BLUE;
    if (hovered == 1u) {
        fill_color = HIGHLIGHTED_COLOR;
    }

    // blend smoothly: background -> border -> fill
    var col = mix(BACKGROUND_COLOR, BORDER_COLOR, border_mask);
    col = mix(col, fill_color, fill_mask);

    // smooth alpha (fill + border)
    let alpha = clamp(fill_mask + border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_external_class(v_uv: vec2<f32>, hovered: u32) -> vec4<f32> {
    let d = distance(v_uv, vec2<f32>(0.5, 0.5));
    let r = 0.48;
    let aa_softness_world = EDGE_SOFTNESS / u_view.zoom;

    // smooth fill mask (circle inside without border)
    var fill_mask = 1.0 - smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + aa_softness_world, d);

    // smooth border mask (ring around circle)
    var border_mask = smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + aa_softness_world, d)
                    * (1.0 - smoothstep(r, r + aa_softness_world, d));

    var fill_color = DARK_BLUE;
    if (hovered == 1u) {
        fill_color = HIGHLIGHTED_COLOR;
    }
    
    var col: vec3<f32>;

    // blend smoothly: background -> border -> fill
    col = mix(BACKGROUND_COLOR, BORDER_COLOR, border_mask);
    col = mix(col, fill_color, fill_mask);

    // smooth alpha (fill + border)
    let alpha = clamp(fill_mask + border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_thing(v_uv: vec2<f32>, hovered: u32) -> vec4<f32> {
    let d = distance(v_uv, vec2<f32>(0.5, 0.5));
    let r = 0.48;
    let aa_softness_world = EDGE_SOFTNESS / u_view.zoom;

    // smooth fill mask (circle inside without border)
    var fill_mask = 1.0 - smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + aa_softness_world, d);

    // smooth border mask (ring around circle)
    var border_mask = smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + aa_softness_world, d)
                    * (1.0 - smoothstep(r, r + aa_softness_world, d));

    let center = vec2(0.5, 0.5);
    let dir = v_uv - center;
    let angle = atan2(dir.y, dir.x);          // -PI..PI
    let ang01 = angle / (2.0 * PI) + 0.5;     // 0..1
    let p = fract(ang01 * DOT_COUNT);         // 0..1 per dot segment
    let distToDot = abs(p - 0.5);             // distance from center of dot segment
    let dot_mask = 1.0 - smoothstep(DOT_RADIUS - DOT_EDGE, DOT_RADIUS + DOT_EDGE, distToDot);

    // apply dot mask to border mask so the ring becomes dotted
    border_mask *= dot_mask;

    var fill_color = vec3<f32>(1.0);
    if (hovered == 1u) {
        fill_color = HIGHLIGHTED_COLOR;
    }

    // blend smoothly: background -> border -> fill
    var col = mix(BACKGROUND_COLOR, BORDER_COLOR, border_mask);
    col = mix(col, fill_color, fill_mask);

    // smooth alpha (fill + border)
    let alpha = clamp(fill_mask + border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_equivalent_class(v_uv: vec2<f32>, hovered: u32) -> vec4<f32> {
    let d = distance(v_uv, vec2<f32>(0.5, 0.5));
    let r = 0.48;
    let aa_softness_world = EDGE_SOFTNESS / u_view.zoom;

    let border_gap = 0.02;

    // radius of the inner border
    let inner_border_outer_r = r - BORDER_THICKNESS - border_gap;
    let inner_border_inner_r = inner_border_outer_r - BORDER_THICKNESS;

    // fill mask (everything inside the inner border)
    let fill_mask = 1.0 - smoothstep(inner_border_inner_r, inner_border_inner_r + aa_softness_world, d);

    // inner border (fully opaque)
    let inner_border_mask = smoothstep(inner_border_inner_r, inner_border_inner_r + aa_softness_world, d) * (1.0 - smoothstep(inner_border_outer_r, inner_border_outer_r + aa_softness_world, d));

    // outer border
    let outer_border_mask = smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + aa_softness_world, d) *
        (1.0 - smoothstep(r, r + aa_softness_world, d));

    var fill_color = LIGHT_BLUE;
    if (hovered == 1u) {
        fill_color = HIGHLIGHTED_COLOR;
    }

    var col = mix(BACKGROUND_COLOR, BORDER_COLOR, outer_border_mask);
    col = mix(col, BORDER_COLOR, inner_border_mask);
    col = mix(col, fill_color, fill_mask);

    // smooth alpha (fill + border)
    let alpha = clamp(fill_mask + outer_border_mask + inner_border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_union(v_uv: vec2<f32>, hovered: u32) -> vec4<f32> {
    let r = 0.48;
    let border_gap = 0.35;
    let aa_softness_world = EDGE_SOFTNESS / u_view.zoom;

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
        smoothstep(inner_border_inner_r, inner_border_inner_r + aa_softness_world, d1) *
        (1.0 - smoothstep(inner_border_outer_r, inner_border_outer_r + aa_softness_world, d1));

    let inner_border_2 =
        smoothstep(inner_border_inner_r, inner_border_inner_r + aa_softness_world, d2) *
        (1.0 - smoothstep(inner_border_outer_r, inner_border_outer_r + aa_softness_world, d2));

    // Combine borders additively
    let inner_border_mask = min(inner_border_1 + inner_border_2, 1.0);

    // outer region
    let outer_fill_mask =
        1.0 - smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + aa_softness_world, d_outer);

    let outer_border_mask =
        smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + aa_softness_world, d_outer) *
        (1.0 - smoothstep(r, r + aa_softness_world, d_outer));

    // inner circle masks
    let inner_fill_1 = 1.0 - smoothstep(inner_border_inner_r, inner_border_inner_r + aa_softness_world, d1);
    let inner_fill_2 = 1.0 - smoothstep(inner_border_inner_r, inner_border_inner_r + aa_softness_world, d2);

    // Combine fills, but reduce intensity where borders exist
    let inner_fill_combined = max(inner_fill_1, inner_fill_2);
    let inner_fill_mask = inner_fill_combined * (1.0 - inner_border_mask);

    // colors
    let inner_fill_color = SET_COLOR;
    var outer_fill_color = LIGHT_BLUE;
    if (hovered == 1u) {
        outer_fill_color = HIGHLIGHTED_COLOR;
    }

    // layering
    var col = mix(BACKGROUND_COLOR, BORDER_COLOR, outer_border_mask);
    col = mix(col, outer_fill_color, outer_fill_mask);
    col = mix(col, BORDER_COLOR, inner_border_mask);
    col = mix(col, inner_fill_color, inner_fill_mask);

    // alpha
    let alpha = clamp(inner_fill_mask + outer_fill_mask + inner_border_mask + outer_border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_intersection_of(v_uv: vec2<f32>, hovered: u32)  -> vec4<f32> {
    let r = 0.48;
    let border_gap = 0.35;
    let aa_softness_world = EDGE_SOFTNESS / u_view.zoom;

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
        smoothstep(inner_border_inner_r, inner_border_inner_r + aa_softness_world, d1) *
        (1.0 - smoothstep(inner_border_outer_r, inner_border_outer_r + aa_softness_world, d1));
    let inner_border_2 =
        smoothstep(inner_border_inner_r, inner_border_inner_r + aa_softness_world, d2) *
        (1.0 - smoothstep(inner_border_outer_r, inner_border_outer_r + aa_softness_world, d2));

    // Combine borders additively
    let inner_border_mask = min(inner_border_1 + inner_border_2, 1.0);

    // outer region
    let outer_fill_mask =
        1.0 - smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + aa_softness_world, d_outer);
    let outer_border_mask =
        smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + aa_softness_world, d_outer) *
        (1.0 - smoothstep(r, r + aa_softness_world, d_outer));

    // inner circle masks
    let inner_fill_1 = 1.0 - smoothstep(inner_border_inner_r, inner_border_inner_r + aa_softness_world, d1);
    let inner_fill_2 = 1.0 - smoothstep(inner_border_inner_r, inner_border_inner_r + aa_softness_world, d2);
    
    // Calculate overlapping region (where both circles are filled)
    let overlap_mask = min(inner_fill_1, inner_fill_2);
    
    // Calculate non-overlapping regions (exclusive OR)
    let non_overlap_mask = max(inner_fill_1, inner_fill_2) - overlap_mask;
    
    // Reduce fill intensity where borders exist
    let overlap_final = overlap_mask * (1.0 - inner_border_mask);
    let non_overlap_final = non_overlap_mask * (1.0 - inner_border_mask);
    
    // colors
    let inner_fill_color = SET_COLOR;
    var outer_fill_color = LIGHT_BLUE;
    if (hovered == 1u) {
        outer_fill_color = HIGHLIGHTED_COLOR;
    }

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

fn draw_complement(v_uv: vec2<f32>, hovered: u32) -> vec4<f32> {
    let d = distance(v_uv, vec2<f32>(0.5, 0.5));
    let r = 0.48;
    let border_gap = 0.35;
    let aa_softness_world = EDGE_SOFTNESS / u_view.zoom;

    // radius of the inner border
    let inner_border_outer_r = r - BORDER_THICKNESS * 0.8 - border_gap;
    let inner_border_inner_r = inner_border_outer_r - BORDER_THICKNESS * 0.8;

    // masks
    let inner_fill_mask = 1.0 - smoothstep(inner_border_inner_r, inner_border_inner_r + aa_softness_world, d);

    // solid outer fill area between inner and outer borders
    let outer_fill_mask =
        smoothstep(inner_border_outer_r, inner_border_outer_r + aa_softness_world, d) *
        (1.0 - smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + aa_softness_world, d));

    // borders
    let inner_border_mask =
        smoothstep(inner_border_inner_r, inner_border_inner_r + aa_softness_world, d) *
        (1.0 - smoothstep(inner_border_outer_r, inner_border_outer_r + aa_softness_world, d));

    let outer_border_mask =
        smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + aa_softness_world, d) *
        (1.0 - smoothstep(r, r + aa_softness_world, d));

    // colors
    let inner_fill_color = SET_COLOR;
    var outer_fill_color = LIGHT_BLUE;
    if (hovered == 1u) {
        outer_fill_color = HIGHLIGHTED_COLOR;
    }

    // layering
    var col = mix(BACKGROUND_COLOR, BORDER_COLOR, outer_border_mask);
    col = mix(col, outer_fill_color, outer_fill_mask);
    col = mix(col, BORDER_COLOR, inner_border_mask);
    col = mix(col, inner_fill_color, inner_fill_mask);

    // combined alpha
    let alpha = clamp(inner_fill_mask + outer_fill_mask + inner_border_mask + outer_border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_deprecated_class(v_uv: vec2<f32>, hovered: u32) -> vec4<f32> {
    let d = distance(v_uv, vec2<f32>(0.5, 0.5));
    let r = 0.48;
    let aa_softness_world = EDGE_SOFTNESS / u_view.zoom;

    // smooth fill mask (circle inside without border)
    var fill_mask = 1.0 - smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + aa_softness_world, d);

    // smooth border mask (ring around circle)
    var border_mask = smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + aa_softness_world, d)
                    * (1.0 - smoothstep(r, r + aa_softness_world, d));

    var fill_color = DEPRECATED_COLOR;
    if (hovered == 1u) {
        fill_color = HIGHLIGHTED_COLOR;
    }

    // blend smoothly: background -> border -> fill
    var col = mix(BACKGROUND_COLOR, BORDER_COLOR, border_mask);
    col = mix(col, fill_color, fill_mask);

    // smooth alpha (fill + border)
    let alpha = clamp(fill_mask + border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_anonymous_class(v_uv: vec2<f32>, hovered: u32) -> vec4<f32> {
    let d = distance(v_uv, vec2<f32>(0.5, 0.5));
    let r = 0.48;
    let aa_softness_world = EDGE_SOFTNESS / u_view.zoom;

    // smooth fill mask (circle inside without border)
    var fill_mask = 1.0 - smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + aa_softness_world, d);

    // smooth border mask (ring around circle)
    var border_mask = smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + aa_softness_world, d)
                    * (1.0 - smoothstep(r, r + aa_softness_world, d));

    let center = vec2(0.5, 0.5);
    let dir = v_uv - center;
    let angle = atan2(dir.y, dir.x);          // -PI..PI
    let ang01 = angle / (2.0 * PI) + 0.5;     // 0..1
    let p = fract(ang01 * DOT_COUNT);         // 0..1 per dot segment
    let distToDot = abs(p - 0.5);             // distance from center of dot segment
    let dot_mask = 1.0 - smoothstep(DOT_RADIUS - DOT_EDGE, DOT_RADIUS + DOT_EDGE, distToDot);

    // apply dot mask to border mask so the ring becomes dotted
    border_mask *= dot_mask;

    var fill_color = LIGHT_BLUE;
    if (hovered == 1u) {
        fill_color = HIGHLIGHTED_COLOR;
    }

    // blend smoothly: background -> border -> fill
    var col = mix(BACKGROUND_COLOR, BORDER_COLOR, border_mask);
    col = mix(col, fill_color, fill_mask);

    // smooth alpha (fill + border)
    let alpha = clamp(fill_mask + border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_literal(v_uv: vec2<f32>, shape_dimensions: vec2<f32>, hovered: u32) -> vec4<f32> {
    let rect_center = vec2<f32>(0.5, 0.5);
    var rect_size = RECT_SCALE;
    let dot_count_rect = 11.0;
    let dot_radius_rect = 0.3;
    var fill_color = LITERAL_COLOR;
    if (hovered == 1u) {
        fill_color = HIGHLIGHTED_COLOR;
    }
    let border_thickness_rect = 0.02;
    let aa_softness_world = EDGE_SOFTNESS / u_view.zoom;

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
    let dot_mask = 1.0 - smoothstep(dot_radius_rect - aa_softness_world, dot_radius_rect + aa_softness_world, dist_to_dot);

    // apply to border only
    border_mask *= dot_mask;

    // composite
    var col = BACKGROUND_COLOR;
    col = mix(col, BORDER_COLOR, border_mask);
    col = mix(col, fill_color, fill_mask);

    let alpha = clamp(fill_mask + border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_rdfs_class(v_uv: vec2<f32>, hovered: u32) -> vec4<f32> {
    let d = distance(v_uv, vec2<f32>(0.5, 0.5));
    let r = 0.48;
    let aa_softness_world = EDGE_SOFTNESS / u_view.zoom;

    // smooth fill mask (circle inside without border)
    var fill_mask = 1.0 - smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + aa_softness_world, d);

    // smooth border mask (ring around circle)
    var border_mask = smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + aa_softness_world, d)
                    * (1.0 - smoothstep(r, r + aa_softness_world, d));

    var fill_color = RDFS_COLOR;
    if (hovered == 1u) {
        fill_color = HIGHLIGHTED_COLOR;
    }

    // blend smoothly: background -> border -> fill
    var col = mix(BACKGROUND_COLOR, BORDER_COLOR, border_mask);
    col = mix(col, fill_color, fill_mask);

    // smooth alpha (fill + border)
    let alpha = clamp(fill_mask + border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_rdfs_resource(v_uv: vec2<f32>, hovered: u32) -> vec4<f32> {
    let d = distance(v_uv, vec2<f32>(0.5, 0.5));
    let r = 0.48;
    let aa_softness_world = EDGE_SOFTNESS / u_view.zoom;

    // smooth fill mask (circle inside without border)
    var fill_mask = 1.0 - smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + aa_softness_world, d);

    // smooth border mask (ring around circle)
    var border_mask = smoothstep(r - BORDER_THICKNESS, r - BORDER_THICKNESS + aa_softness_world, d)
                    * (1.0 - smoothstep(r, r + aa_softness_world, d));

    let center = vec2(0.5, 0.5);
    let dir = v_uv - center;
    let angle = atan2(dir.y, dir.x);          // -PI..PI
    let ang01 = angle / (2.0 * PI) + 0.5;     // 0..1
    let p = fract(ang01 * DOT_COUNT);         // 0..1 per dot segment
    let distToDot = abs(p - 0.5);             // distance from center of dot segment
    let dot_mask = 1.0 - smoothstep(DOT_RADIUS - DOT_EDGE, DOT_RADIUS + DOT_EDGE, distToDot);

    // apply dot mask to border mask so the ring becomes dotted
    border_mask *= dot_mask;

    var fill_color = RDFS_COLOR;
    if (hovered == 1u) {
        fill_color = HIGHLIGHTED_COLOR;
    }

    // blend smoothly: background -> border -> fill
    var col = mix(BACKGROUND_COLOR, BORDER_COLOR, border_mask);
    col = mix(col, fill_color, fill_mask);

    // smooth alpha (fill + border)
    let alpha = clamp(fill_mask + border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_datatype(v_uv: vec2<f32>, shape_dimensions: vec2<f32>, hovered: u32) -> vec4<f32> {
    let rect_center = vec2<f32>(0.5, 0.5);
    let rect_size = RECT_SCALE;
    let border_thickness = BORDER_THICKNESS * 0.5; // controls visible border width
    let aa_softness_world = EDGE_SOFTNESS / u_view.zoom;

    var fill_color = LITERAL_COLOR;
    if (hovered == 1u) {
        fill_color = HIGHLIGHTED_COLOR;
    }

    // Convert to local space relative to rectangle center
    let p = abs(v_uv - rect_center);
    let half_size = 0.5 * rect_size;

    // Signed distance to rectangle boundary (negative inside)
    let dist = max(p.x - half_size.x, p.y - half_size.y);

    // Smooth alpha mask for outer edge (anti-aliasing)
    let outer_alpha = 1.0 - smoothstep(0.0, aa_softness_world, dist);

    // Smooth mask for *inner border* (offset by thickness)
    let inner_alpha = 1.0 - smoothstep(-border_thickness, -border_thickness + aa_softness_world, dist);

    // Border is the difference between outer and inner areas
    let border_mask = outer_alpha - inner_alpha;

    // Blend background -> border -> fill
    var col = mix(BACKGROUND_COLOR, BORDER_COLOR, border_mask);
    col = mix(col, fill_color, inner_alpha);

    let alpha = clamp(outer_alpha, 0.0, 1.0);
    return vec4<f32>(col, alpha);
}

fn draw_property(v_uv: vec2<f32>, shape_dimensions: vec2<f32>, base_color: vec3<f32>, hovered: u32) -> vec4<f32> {
    let rect_center = vec2<f32>(0.5, 0.5);
    var rect_size = RECT_SCALE;

    let p = v_uv - rect_center;

    let half_size = 0.5 * rect_size;

    let inside_x = abs(p.x) <= half_size.x;
    let inside_y = abs(p.y) <= half_size.y;

    let inside_rect = inside_x && inside_y;
    // mask selection
    var fill_mask = 0.0;
    if(inside_rect) {
        fill_mask = 1.0;
    }

    var fill_color = base_color;
    if (hovered == 1u) {
        fill_color = HIGHLIGHTED_COLOR;
    }

    // composite
    var col = BACKGROUND_COLOR;
    col = mix(col, fill_color, fill_mask);

    let alpha = clamp(fill_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_inverse_property(v_uv: vec2<f32>, shape_dimensions: vec2<f32>, hovered: u32) -> vec4<f32> {
    var fill_color = LIGHT_BLUE;
    if (hovered == 1u) {
        fill_color = HIGHLIGHTED_COLOR;
    }

    let rect_center1 = vec2<f32>(0.5, 0.32);
    let rect_center2 = vec2<f32>(0.5, 0.68);
    var rect_size = RECT_SCALE;

    let p1 = v_uv - rect_center1;
    let p2 = v_uv - rect_center2;

    let half_size = 0.5 * rect_size;

    let inside_x1 = abs(p1.x) <= half_size.x;
    let inside_y1 = abs(p1.y) <= half_size.y;
    let inside_x2 = abs(p2.x) <= half_size.x;
    let inside_y2 = abs(p2.y) <= half_size.y;

    let inside_rect1 = inside_x1 && inside_y1;
    let inside_rect2 = inside_x2 && inside_y2;
    // mask selection
    var fill_mask = 0.0;
    if(inside_rect1 || inside_rect2) {
        fill_mask = 1.0;
    }

    // composite
    var col = BACKGROUND_COLOR;
    col = mix(col, fill_color, fill_mask);

    let alpha = clamp(fill_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_disjoint_with(v_uv: vec2<f32>, shape_dimensions: vec2<f32>, hovered: u32) -> vec4<f32> {
    let rect_center = vec2<f32>(0.5, 0.5);
    var rect_size = RECT_SCALE;
    let aa_softness_world = EDGE_SOFTNESS / u_view.zoom;

    let p = v_uv - rect_center;

    let half_size = 0.5 * rect_size;

    // perimeter coordinate
    let width = 2.0 * half_size.x;
    let height = 2.0 * half_size.y;
    let perim = 2.0 * (width + height);

    // positions for two inner circles
    let circle_r = 0.10;
    let offset = 0.15;
    let c1 = vec2<f32>(0.5 - offset, 0.5);
    let c2 = vec2<f32>(0.5 + offset, 0.5);

    let d1 = distance(v_uv, c1);
    let d2 = distance(v_uv, c2);

    // Inner circle fill and border masks
    let inner_fill_1 = 1.0 - smoothstep(circle_r, circle_r + aa_softness_world, d1);
    let inner_fill_2 = 1.0 - smoothstep(circle_r, circle_r + aa_softness_world, d2);

    // borders
    let border_outer_r = circle_r + BORDER_THICKNESS * 0.4;
    let border_inner_r = circle_r - BORDER_THICKNESS * 0.4;

    let border_1 =
        smoothstep(border_inner_r, border_inner_r + aa_softness_world, d1) *
        (1.0 - smoothstep(border_outer_r, border_outer_r + aa_softness_world, d1));
    let border_2 =
        smoothstep(border_inner_r, border_inner_r + aa_softness_world, d2) *
        (1.0 - smoothstep(border_outer_r, border_outer_r + aa_softness_world, d2));

    let inner_fill_mask = clamp(inner_fill_1 + inner_fill_2, 0.0, 1.0);
    let inner_border_mask = clamp(border_1 + border_2, 0.0, 1.0);

    let inside_x = abs(p.x) <= half_size.x;
    let inside_y = abs(p.y) <= half_size.y;

    let inside_rect = inside_x && inside_y;
    // mask selection
    var outer_fill_mask = 0.0;
    if(inside_rect) {
        outer_fill_mask = 1.0;
    }

    // colors
    let inner_fill_color = SET_COLOR;
    var outer_fill_color = LIGHT_BLUE;
    if (hovered == 1u) {
        outer_fill_color = HIGHLIGHTED_COLOR;
    }

    // composite
    var col = BACKGROUND_COLOR;
    col = mix(col, outer_fill_color, outer_fill_mask);
    col = mix(col, inner_fill_color, inner_fill_mask);
    col = mix(col, BORDER_COLOR, inner_border_mask);

    let alpha = clamp(outer_fill_mask + inner_fill_mask + inner_border_mask, 0.0, 1.0);

    return vec4<f32>(col, alpha);
}

fn draw_node_by_type(node_type: u32, v_uv: vec2<f32>, shape_dimensions: vec2<f32>, hovered: u32) -> vec4<f32> {
    switch node_type {
        case 0: {return draw_class(v_uv, hovered);}
        case 1: {return draw_external_class(v_uv, hovered);}
        case 2: {return draw_thing(v_uv, hovered);}
        case 3: {return draw_equivalent_class(v_uv, hovered);}
        case 4: {return draw_union(v_uv, hovered);}
        case 5: {return draw_union(v_uv, hovered);}
        case 6: {return draw_intersection_of(v_uv, hovered);}
        case 7: {return draw_complement(v_uv, hovered);}
        case 8: {return draw_deprecated_class(v_uv, hovered);}
        case 9: {return draw_anonymous_class(v_uv, hovered);}
        case 10: {return draw_literal(v_uv, shape_dimensions, hovered);}
        case 11: {return draw_rdfs_class(v_uv, hovered);}
        case 12: {return draw_rdfs_resource(v_uv, hovered);}
        case 13: {return draw_datatype(v_uv, shape_dimensions, hovered);}
        case 14: {return draw_property(v_uv, shape_dimensions, LIGHT_BLUE, hovered);}
        case 15: {return draw_property(v_uv, shape_dimensions, DATATYPE_PROPERTY_COLOR, hovered);}
        case 16: {return draw_property(v_uv, shape_dimensions, vec3<f32>(1.0), hovered);}
        case 17: {return draw_inverse_property(v_uv, shape_dimensions, hovered);}
        case 18: {return draw_disjoint_with(v_uv, shape_dimensions, hovered);}
        case 19: {return draw_property(v_uv, shape_dimensions, RDFS_COLOR, hovered);}
        case 20: {return draw_property(v_uv, shape_dimensions, DEPRECATED_COLOR, hovered);}
        case 21: {return draw_property(v_uv, shape_dimensions, DARK_BLUE, hovered);}
        case 22: {return draw_property(v_uv, shape_dimensions, LIGHT_BLUE, hovered);}
        default: {return vec4<f32>(0.0);}
    }
}