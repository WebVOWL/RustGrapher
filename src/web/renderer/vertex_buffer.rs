use wgpu::util::DeviceExt; // for create_buffer_init

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub quad_pos: [f32; 2],
}

impl Vertex {
    // location 0 -> Float32x2
    const ATTRIBS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x2];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;

        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

pub const VERTICES: &[Vertex] = &[
    Vertex {
        quad_pos: [-1.0, -1.0],
    },
    Vertex {
        quad_pos: [1.0, -1.0],
    },
    Vertex {
        quad_pos: [-1.0, 1.0],
    },
    Vertex {
        quad_pos: [-1.0, 1.0],
    },
    Vertex {
        quad_pos: [1.0, -1.0],
    },
    Vertex {
        quad_pos: [1.0, 1.0],
    },
];

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Instance {
    pub position: [f32; 2],
}

impl Instance {
    // location 1 -> Float32x2, instance step mode
    const ATTRIBS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![1 => Float32x2];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBS,
        }
    }
}

/// Create an instance vertex buffer containing node positions (vec2 f32).
pub fn create_instance_buffer(device: &wgpu::Device, positions: &[[f32; 2]]) -> wgpu::Buffer {
    // positions are stored as contiguous vec2<f32> matching Instance.position
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("instance_node_positions_buffer"),
        contents: bytemuck::cast_slice(positions),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    })
}

// --- Helpers for node positions storage buffer (SSBO) ---
// The shader expects a read-only storage buffer at group(0) binding(1)
// containing an array<vec2<f32>> of node positions (in pixels).

/// Create a storage buffer containing node positions (vec2 f32).
pub fn create_node_positions_buffer(device: &wgpu::Device, positions: &[[f32; 2]]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("node_positions_buffer"),
        contents: bytemuck::cast_slice(positions),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    })
}

/// Create (or update) a bind group that binds the node positions buffer at binding = 1.
/// Note: the provided bind_group_layout must include a binding=1 entry matching a storage buffer.
pub fn create_node_positions_bind_group(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    node_positions_buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 1, // matches shader: @group(0) @binding(1) var<storage, read> node_positions
            resource: node_positions_buffer.as_entire_binding(),
        }],
        label: Some("node_positions_bind_group"),
    })
}
