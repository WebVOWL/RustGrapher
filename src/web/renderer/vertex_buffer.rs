use web_sys::js_sys::Math::atan2;
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
            NodeType::ValuesFrom => 3,   // Blue line and arrow
            _ => match node_types[start_idx] {
                NodeType::Union
                | NodeType::DisjointUnion
                | NodeType::Complement
                | NodeType::Intersection => 4,
                _ => 0, // Solid line with black arrow
            },
        };

        let (shape_type, shape_dim) = match node_shapes[end_idx] {
            NodeShape::Circle { r } => (0, [r, 0.0]),
            NodeShape::Rectangle { w, h } => (1, [w, h]),
        };
        let radius_pix = 50.0;
        let mut start_shape = node_shapes[start_idx];
        let mut end_shape = node_shapes[end_idx];
        // Handle symmetric properties
        if start_idx == end_idx {
            let node_center = node_positions[start_idx];

            if let NodeShape::Circle { r } = node_shapes[start_idx] {
                // Calculate angle from start node to center node (control point)
                let dx = center[0] - node_center[0];
                let dy = center[1] - node_center[1];
                let angle = atan2(dy as f64, dx as f64) as f32; // radians

                // Offset direction perpendicular to angle
                let offset_angle = angle + std::f32::consts::FRAC_PI_2;
                let offset_x = offset_angle.cos() * radius_pix * 0.5;
                let offset_y = offset_angle.sin() * radius_pix * 0.5;

                // Move start and end points along the rotated direction.
                start = [
                    node_center[0] + offset_x + r * radius_pix / 4.0 * angle.cos(),
                    node_center[1] + offset_y + r * radius_pix / 4.0 * angle.sin(),
                ];
                end = [
                    node_center[0] - offset_x + r * radius_pix / 4.0 * angle.cos(),
                    node_center[1] - offset_y + r * radius_pix / 4.0 * angle.sin(),
                ];
                let half_shape = match start_shape {
                    NodeShape::Circle { r } => NodeShape::Circle { r: r / 2.0 },
                    NodeShape::Rectangle { w, h } => NodeShape::Rectangle { w: w / 2.0, h },
                };
                start_shape = half_shape;
                end_shape = half_shape;
            };
        }
        let start_center = start;
        let end_center = end; // same as start_center for self-loop

        // Direction vectors from node center -> control point
        let dir_start = [center[0] - start_center[0], center[1] - start_center[1]];
        let dir_end = [center[0] - end_center[0], center[1] - end_center[1]];

        // Lengths
        let start_len = (dir_start[0] * dir_start[0] + dir_start[1] * dir_start[1]).sqrt();
        let end_len = (dir_end[0] * dir_end[0] + dir_end[1] * dir_end[1]).sqrt();

        // Normalized directions, guard against zero-length
        let dir_start_n = if start_len > 1e-6 {
            [dir_start[0] / start_len, dir_start[1] / start_len]
        } else {
            [0.0, -1.0] // fallback direction
        };
        let dir_end_n = if end_len > 1e-6 {
            [dir_end[0] / end_len, dir_end[1] / end_len]
        } else {
            [0.0, 1.0] // opposite fallback to separate start/end a bit
        };

        // Move start point to perimeter of its node
        if start_idx != center_idx {
            start = match start_shape {
                NodeShape::Circle { r } => [
                    start_center[0] + dir_start_n[0] * r * radius_pix,
                    start_center[1] + dir_start_n[1] * r * radius_pix,
                ],
                NodeShape::Rectangle { w, h } => {
                    // Project ray onto rectangle perimeter
                    let dx = dir_start_n[0];
                    let dy = dir_start_n[1];
                    let mut scale = f32::INFINITY;

                    if dx.abs() > 1e-6 {
                        // Effective half-width: (w * radius_pix) * 0.9
                        scale = scale.min((w * 0.9 / 2.0) / dx.abs());
                    }
                    if dy.abs() > 1e-6 {
                        // Effective half-height: (w * radius_pix) * 0.25
                        scale = scale.min((h * 0.25 / 2.0) / dy.abs());
                    }

                    if !scale.is_finite() {
                        [start_center[0], start_center[1]]
                    } else {
                        [
                            start_center[0] + dir_start_n[0] * scale * radius_pix,
                            start_center[1] + dir_start_n[1] * scale * radius_pix,
                        ]
                    }
                }
            };
        }

        // Move end point to perimeter of its node
        if end_idx != center_idx {
            end = match end_shape {
                NodeShape::Circle { r } => [
                    end_center[0] + dir_end_n[0] * r * radius_pix,
                    end_center[1] + dir_end_n[1] * r * radius_pix,
                ],
                NodeShape::Rectangle { w, h } => {
                    let dx = dir_end_n[0];
                    let dy = dir_end_n[1];
                    let mut scale = f32::INFINITY;

                    if dx.abs() > 1e-6 {
                        scale = scale.min((w * 0.9) / dx.abs());
                    }
                    if dy.abs() > 1e-6 {
                        scale = scale.min((h * 0.25) / dy.abs());
                    }

                    if !scale.is_finite() {
                        [end_center[0], end_center[1]]
                    } else {
                        [
                            end_center[0] + dir_end_n[0] * scale * radius_pix,
                            end_center[1] + dir_end_n[1] * scale * radius_pix,
                        ]
                    }
                }
            };
        }

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
