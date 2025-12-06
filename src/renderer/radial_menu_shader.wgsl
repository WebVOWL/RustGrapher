struct VertexInput {
    @location(0) position: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Required to be 16-byte aligned by WebGL
struct ViewUniforms {
    resolution: vec2<f32>,
    pan: vec2<f32>,
    zoom: f32,
    _padding: vec2<f32>
};

struct MenuUniforms {
    center: vec2<f32>,       // World position of the node center
    radius_inner: f32,       // Inner radius in pixels
    radius_outer: f32,       // Outer radius in pixels
    hovered_segment: i32,    // -1: None, 0: Top, 1: Bottom
    _padding: vec3<u32>,
};

@group(0) @binding(0)
var<uniform> view: ViewUniforms;

@group(1) @binding(0)
var<uniform> menu: MenuUniforms;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    input: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;

    // quad expansion relative to world center
    let size = menu.radius_outer * 2.5; 
    let world_pos = menu.center + (input.position * size);

    // World to Screen transformation
    let world_rel = world_pos - view.pan;
    let world_rel_zoomed = world_rel * view.zoom;
    let screen_center = view.resolution * 0.5;
    let screen_pos = screen_center + vec2<f32>(world_rel_zoomed.x, -world_rel_zoomed.y);

    let ndc_x = (screen_pos.x / view.resolution.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (screen_pos.y / view.resolution.y) * 2.0;

    out.clip_position = vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    out.uv = input.position; // -1 to 1
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Calculate distance in pixels relative to zoom
    let dist_factor = length(in.uv);
    let pixel_dist = dist_factor * (menu.radius_outer * 2.5) * view.zoom;

    let r_inner = menu.radius_inner * view.zoom;
    let r_outer = menu.radius_outer * view.zoom;

    // Discard outside the ring
    if (pixel_dist < r_inner || pixel_dist > r_outer) {
        discard;
    }

    // Determine Angle for sectors
    let angle = atan2(in.uv.y, in.uv.x);
    
    var segment = 0; // Top
    if (angle < 0.0) {
        segment = 1; // Bottom
    }

    // Styling
    var color = vec3<f32>(0.2, 0.2, 0.2); // Dark grey default
    var alpha = 0.8;

    // Hover effect
    if (segment == menu.hovered_segment) {
        color = vec3<f32>(0.3, 0.5, 0.9); // Blue highlight
        alpha = 0.95;
    }

    // Separator line
    if (abs(in.uv.y) < 0.02) {
        color = vec3<f32>(0.0, 0.0, 0.0);
    }

    // Soft edges
    let edge_width = 1.5;
    let alpha_edge = smoothstep(r_inner, r_inner + edge_width, pixel_dist) * (1.0 - smoothstep(r_outer - edge_width, r_outer, pixel_dist));
    
    return vec4<f32>(color, alpha * alpha_edge);
}