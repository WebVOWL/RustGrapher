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
    // first triangle
    Vertex {
        quad_pos: [-1.0, -1.0],
    }, // bottom-left
    Vertex {
        quad_pos: [1.0, -1.0],
    }, // bottom-right
    Vertex {
        quad_pos: [1.0, 1.0],
    }, // top-right
    // second triangle
    Vertex {
        quad_pos: [-1.0, -1.0],
    }, // bottom-left
    Vertex {
        quad_pos: [1.0, 1.0],
    }, // top-right
    Vertex {
        quad_pos: [-1.0, 1.0],
    }, // top-left
];

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NodeInstance {
    pub position: [f32; 2],
}

impl NodeInstance {
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

/// Create an instance vertex buffer containing node positions.
pub fn create_node_instance_buffer(device: &wgpu::Device, positions: &[[f32; 2]]) -> wgpu::Buffer {
    // positions are stored as contiguous vec2<f32> matching Instance.position
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("instance_node_positions_buffer"),
        contents: bytemuck::cast_slice(positions),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    })
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EdgeInstance {
    pub start: [f32; 2],
    pub end: [f32; 2],
}

impl EdgeInstance {
    // locations 1 and 2: start, end
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![1 => Float32x2, 2 => Float32x2];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBS,
        }
    }
}

pub fn create_edge_instance_buffer(device: &wgpu::Device, edges: &[[[f32; 2]; 2]]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("edge_instance_buffer"),
        contents: bytemuck::cast_slice(edges),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    })
}
