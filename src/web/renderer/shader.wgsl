struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(model.position, 1.0);
    return out;
}

// Add uniform resolution (using vec4 for 16-byte alignment)
@group(0) @binding(0)
var<uniform> u_resolution: vec4<f32>;

@fragment
fn fs_main(
    @builtin(position) pos: vec4<f32>,
) -> @location(0) vec4<f32> {
    // use the uniform resolution (xy)
    var st = pos.xy / u_resolution.xy;
    var aspect = u_resolution.x / u_resolution.y;

    // correct aspect correction using u_resolution
    st.x *= aspect;

    // define nodes in normalized (0..1) space (x will be multiplied by aspect)
    // (0,0) is defined as top right (left hand rule)

    const N = 3;
    var nodes = array<vec2<f32>, N>();
    nodes[0] = vec2(0.3 * aspect, 0.7);
    nodes[1] = vec2(0.5 * aspect, 0.3);
    nodes[2] = vec2(0.7 * aspect, 0.7);

    const E = 3;
    var edges = array<vec2<i32>, E>();
    edges[0] = vec2(0,1);
    edges[1] = vec2(1,2);
    edges[2] = vec2(2,0);

    // background
    var color = vec3(0.8387990119, 0.8713671192, 0.8796223963);

    // draw edges
    const edge_thickness = 0.002; // in aspect-corrected space
    const edge_aa = 0.001;
    for (var i = 0; i < E; i++) {
        var a = nodes[edges[i].x];
        var b = nodes[edges[i].y];
        var d = seg_dist(st, a, b);
        var mask = 1.0 - smoothstep(edge_thickness - edge_aa, edge_thickness + edge_aa, d);
        color = mix(color, vec3(0.0), mask);
    }

    // draw nodes
    const node_radius = 0.03;
    const node_edge = 0.002;
    for (var i = 0; i < N; i++) {
        var cpos = nodes[i];
        // fill
        var m = circle_mask(st, cpos, node_radius, node_edge);
        // Default node color from WebVOWL
        color = mix(color, vec3(0.4072402119, 0.6038273389, 1.0), m);
        // draw border
        var border = circle_mask(st, cpos, node_radius + node_edge, node_edge * 0.6) - m;
        color = mix(color, vec3(0.0), max(0.0, border));
    }
    
    return vec4(color, 1.0);
}

// distance from point p to segment a-b
fn seg_dist(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    var pa = p - a;
    var ba = b - a;
    var ba_len2 = dot(ba, ba);
    // division by zero
    if ba_len2 == 0.0 {
        return length(pa);
    }
    var h = clamp(dot(pa, ba) / ba_len2, 0.0, 1.0);
    var projection = a + ba * h;
    return length(p - projection);
}

// circle mask centered at c
fn circle_mask(p: vec2<f32>, c: vec2<f32>, r: f32, edge: f32) -> f32 {
    var d = length(p - c);
    return 1.0 - smoothstep(r - edge, r + edge, d);
}