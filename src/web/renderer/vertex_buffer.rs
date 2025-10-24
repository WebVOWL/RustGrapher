use wgpu::util::DeviceExt;

use crate::web::renderer::{node_shape::NodeShape, node_types::NodeType};

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
    pub node_type: u32,
    pub shape_type: u32,     // 0 = Circle, 1 = Rectangle
    pub shape_dim: [f32; 2], // [r, _] for Circle or [w, h] for Rectangle
}

impl NodeInstance {
    const ATTRIBS: [wgpu::VertexAttribute; 4] =
        wgpu::vertex_attr_array![1 => Float32x2, 2 => Uint32, 3 => Uint32, 4 => Float32x2];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBS,
        }
    }
}

pub fn build_node_instances(
    device: &wgpu::Device,
    positions: &[[f32; 2]],
    node_types: &[NodeType],
    node_shapes: &[NodeShape],
) -> Vec<NodeInstance> {
    let mut node_instances: Vec<NodeInstance> = vec![];
    for (i, pos) in positions.iter().enumerate() {
        let (shape_type, shape_dim) = match node_shapes[i] {
            NodeShape::Circle { r } => (0, [r, 0.0]),
            NodeShape::Rectangle { w, h } => (1, [w, h]),
        };
        node_instances.push(NodeInstance {
            position: *pos,
            node_type: node_types[i] as u32,
            shape_type: shape_type,
            shape_dim,
        });
    }
    node_instances
}

/// Create an instance vertex buffer containing node positions.
pub fn create_node_instance_buffer(
    device: &wgpu::Device,
    positions: &[[f32; 2]],
    node_types: &[NodeType],
    node_shapes: &[NodeShape],
) -> wgpu::Buffer {
    let node_instances = build_node_instances(device, positions, node_types, node_shapes);

    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("instance_node_buffer"),
        contents: bytemuck::cast_slice(&node_instances),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    })
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EdgeInstance {
    pub start: [f32; 2],
    pub center: [f32; 2],
    pub end: [f32; 2],
    pub shape_type: u32,     // 0 = Circle, 1 = Rectangle
    pub shape_dim: [f32; 2], // [r, _] for Circle or [w, h] for Rectangle
    pub line_type: u32,
}

impl EdgeInstance {
    const ATTRIBS: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![
        1 => Float32x2,
        2 => Float32x2,
        3 => Float32x2,
        4 => Uint32,
        5 => Float32x2,
        6 => Uint32,
    ];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBS,
        }
    }
}

pub fn build_edge_instances(
    device: &wgpu::Device,
    edges: &[[usize; 3]],
    node_positions: &[[f32; 2]],
    node_shapes: &[NodeShape],
    node_types: &[NodeType],
) -> Vec<EdgeInstance> {
    let mut edge_instances = Vec::with_capacity(edges.len());

    for &[start_idx, center_idx, end_idx] in edges {
        let mut start = node_positions[start_idx];
        let center = node_positions[center_idx];
        let mut end = node_positions[end_idx];
        let line_type = match node_types[center_idx] {
            NodeType::SubclassOf => 1,   // Dotted line with white arrow
            NodeType::DisjointWith => 2, // Dotted line with no arrow
            _ => 0,                      // Solid line with black arrow
        };

        let (shape_type, shape_dim) = match node_shapes[end_idx] {
            NodeShape::Circle { r } => (0, [r, 0.0]),
            NodeShape::Rectangle { w, h } => (1, [w, h]),
        };
        // TODO: handle symmetric properties
        // if start_idx == end_idx {
        //
        // }

        edge_instances.push(EdgeInstance {
            start,
            center,
            end,
            shape_type,
            shape_dim,
            line_type,
        });
    }
    edge_instances
}

pub fn create_edge_instance_buffer(
    device: &wgpu::Device,
    edges: &[[usize; 3]],
    node_positions: &[[f32; 2]],
    node_shapes: &[NodeShape],
    node_types: &[NodeType],
) -> wgpu::Buffer {
    let edge_instances =
        build_edge_instances(device, edges, node_positions, node_shapes, node_types);

    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("edge_instance_buffer"),
        contents: bytemuck::cast_slice(&edge_instances),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    })
}
