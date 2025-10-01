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

    // correct aspect correction using u_resolution
    st.x *= u_resolution.x / u_resolution.y;
    st = st * 2.0 - 1.0;

    var d = length(abs(st) - 0.3);

    var color = vec4(vec3(step(0.2, d)), 1.0);

    return color;
}

fn circle(_st: vec2<f32>, radius: f32) -> f32 {
    var dist = _st - vec2(0.5);
	return 1.0 - smoothstep(radius - (radius * 0.01),
                         radius + (radius * 0.01),
                         dot(dist,dist) * 4.0);
}