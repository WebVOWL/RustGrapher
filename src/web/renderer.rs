pub mod elements;
pub mod events;
mod node_shape;
mod vertex_buffer;

use crate::web::{
    prelude::EVENT_DISPATCHER,
    quadtree::QuadTree,
    renderer::{
        elements::{
            characteristic::Characteristic, element_type::ElementType, generic::*, owl::*, rdf::*,
            rdfs::*,
        },
        events::RenderEvent,
        graph_data::GraphDisplayData,
        node_shape::NodeShape,
    },
    simulator::{Simulator, components::nodes::Position, ressources::events::SimulatorEvent},
};
use glam::Vec2;
use glyphon::{
    Attrs, Buffer as GlyphBuffer, Cache, Color, Family, FontSystem, Metrics, Resolution, Shaping,
    SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};
use log::info;
use specs::{Join, ReaderId, WorldExt};
use std::{cmp::min, collections::HashMap, sync::Arc};
use vertex_buffer::{MenuUniforms, NodeInstance, VERTICES, Vertex, ViewUniforms};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use web_time::{Instant, SystemTime};
use wgpu::util::DeviceExt;
use winit::event::MouseButton;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::EventLoopExtWebSys;
use winit::{dpi::PhysicalPosition, event::MouseScrollDelta};
use winit::{event_loop::ActiveEventLoop, keyboard::KeyCode, window::Window};

pub struct RadialMenuState {
    pub active: bool,
    pub target_node_index: usize,
    pub center_world: Vec2,
    pub radius_inner: f32,
    pub radius_outer: f32,
    pub hovered_segment: i32, // -1 None, 0 Top (Freeze), 1 Bottom (Subgraph)
    pub menu_buffers: Option<[GlyphBuffer; 2]>,
}

pub struct State {
    // #[cfg(target_arch = "wasm32")]
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    num_vertices: u32,
    view_uniform_buffer: wgpu::Buffer,
    view_uniforms: ViewUniforms, // Store CPU-side copy
    // bind group for group(0): binding 0 = resolution uniform only
    bind_group0: wgpu::BindGroup,
    // instance buffer with node positions (array<vec2<f32>>), bound as vertex buffer slot 1
    node_instance_buffer: wgpu::Buffer,
    // number of instances (length of node positions)
    num_instances: u32,
    edge_pipeline: wgpu::RenderPipeline,
    edge_vertex_buffer: wgpu::Buffer,
    num_edge_vertices: u32,
    arrow_pipeline: wgpu::RenderPipeline,
    arrow_vertex_buffer: wgpu::Buffer,
    num_arrow_vertices: u32,

    // Node and edge coordinates in pixels
    positions: Vec<[f32; 2]>,
    labels: Vec<String>,
    edges: Vec<[usize; 3]>,
    solitary_edges: Vec<[usize; 3]>,
    node_types: Vec<ElementType>,
    node_shapes: Vec<NodeShape>,
    cardinalities: Vec<(u32, (String, Option<String>))>,
    characteristics: HashMap<usize, String>,
    simulator: Simulator<'static, 'static>,
    paused: bool,
    hovered_index: i32,

    // User input
    cursor_position: Option<Vec2>,
    node_dragged: bool,
    pan_active: bool,
    last_pan_position: Option<Vec2>, // Screen space
    pan: Vec2,
    zoom: f32,
    click_start_pos: Option<Vec2>,

    // Events
    reader_id: ReaderId<RenderEvent>,

    // Performance
    last_fps_time: Instant,
    fps_counter: u32,

    // Glyphon resources are initialized lazily when we have a non-zero surface.
    font_system: Option<FontSystem>,
    swash_cache: Option<SwashCache>,
    viewport: Option<Viewport>,
    atlas: Option<TextAtlas>,
    text_renderer: Option<TextRenderer>,
    // one glyphon buffer per node containing its text (created when glyphon is initialized)
    text_buffers: Option<Vec<GlyphBuffer>>,
    cardinality_text_buffers: Option<Vec<(usize, GlyphBuffer)>>,
    pub window: Arc<Window>,

    // Radial menu
    radial_menu_pipeline: wgpu::RenderPipeline,
    radial_menu_bind_group: wgpu::BindGroup,
    radial_menu_uniform_buffer: wgpu::Buffer,
    radial_menu_state: RadialMenuState,
}

impl State {
    pub async fn new(window: Arc<Window>, graph: GraphDisplayData) -> anyhow::Result<Self> {
        // Check if we can use WebGPU (as of this writing it's only enabled in some browsers)
        let is_webgpu_enabled = wgpu::util::is_browser_webgpu_supported().await;

        // Pick appropriate render backends
        // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
        let backends = if is_webgpu_enabled {
            wgpu::Backends::BROWSER_WEBGPU
        } else if cfg!(target_arch = "wasm32") {
            wgpu::Backends::GL
        } else {
            wgpu::Backends::PRIMARY
        };

        info!("Building render state (WebGPU={is_webgpu_enabled})");

        let size = window.inner_size();

        // The instance is a handle to our GPU
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: backends,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                // WebGL doesn't support all of wgpu's features, so if
                // we're building for a browser not supporting WebGPU,
                // we'll have to disable some.
                required_limits: if cfg!(target_arch = "wasm32") && !is_webgpu_enabled {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        // Always configure surface, even if size is zero
        let mut config = config;
        if size.width == 0 || size.height == 0 {
            config.width = 1;
            config.height = 1;
        }
        surface.configure(&device, &config);
        let surface_configured = true;

        let shader =
            device.create_shader_module(wgpu::include_wgsl!("./renderer/node_shader.wgsl"));

        // Create a bind group layout for group(0): binding 0 = uniform (resolution)
        let resolution_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("resolution_bind_group_layout"),
                entries: &[
                    // binding 0: resolution uniform (vec4<f32>) used in vertex shader
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Create View uniform buffer
        let view_uniforms = ViewUniforms {
            resolution: [size.width as f32, size.height as f32],
            pan: [0.0, 0.0],
            zoom: 1.0,
            _padding: [0.0, 0.0, 0.0],
        };

        let view_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("View Uniform Buffer"),
            contents: bytemuck::cast_slice(&[view_uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let hovered_index = -1;

        let mut labels = graph.labels;

        let mut elements = graph.node_types;

        let mut positions = vec![];

        let mut node_shapes = vec![];
        for (i, node_type) in elements.iter().enumerate() {
            match node_type {
                // Circle
                NodeType::Class
                | NodeType::ExternalClass
                | NodeType::EquivalentClass
                | NodeType::Union
                | NodeType::DisjointUnion
                | NodeType::Intersection
                | NodeType::Complement
                | NodeType::DeprecatedClass
                | NodeType::AnonymousClass
                | NodeType::RdfsClass
                | NodeType::RdfsResource
                | NodeType::NoDraw => {
                    node_shapes.push(NodeShape::Circle { r: 1.0 });
                }
                NodeType::Thing => {
                    node_shapes.push(NodeShape::Circle { r: 0.7 });
                }
                // Rectangle
                NodeType::Literal
                | NodeType::Datatype
                | NodeType::ObjectProperty
                | NodeType::DatatypeProperty
                | NodeType::SubclassOf
                | NodeType::InverseProperty
                | NodeType::DisjointWith
                | NodeType::RdfProperty
                | NodeType::DeprecatedProperty
                | NodeType::ExternalProperty
                | NodeType::ValuesFrom => {
                    node_shapes.push(NodeShape::Rectangle { w: 1.0, h: 1.0 });
                }
            }
            positions.push([
                f32::fract(f32::sin(i as f32) * 12345.6789),
                f32::fract(f32::sin(i as f32) * 98765.4321),
            ]);
        }
        if positions.len() == 0 {
            positions.push([0.0, 0.0]);
            labels.push("".to_string());
            node_shapes.push(NodeShape::Circle { r: 0.0 });
            elements.push(NodeType::NoDraw);
        }

        let edges = if graph.edges.len() > 0 {
            graph.edges
        } else {
            vec![[0, 0, 0]]
        };

        let cardinalities = graph.cardinalities;

        let mut characteristics = graph.characteristics;

        // FontSystem instance for text measurement
        let mut font_system =
            FontSystem::new_with_fonts(core::iter::once(glyphon::fontdb::Source::Binary(
                Arc::new(include_bytes!("../../assets/DejaVuSans.ttf").to_vec()),
            )));
        font_system.db_mut().set_sans_serif_family("DejaVu Sans");

        let mut node_shapes = node_shapes;
        let mut labels = labels;

        // iterate over labels and update the width of corresponding rectangle nodes
        for (i, label_text) in labels.clone().iter().enumerate() {
            // Set fixed size for disjoint property
            if let Some(ElementType::Owl(OwlType::Edge(OwlEdge::DisjointWith))) = elements.get(i) {
                node_shapes[i] = NodeShape::Rectangle { w: 0.75, h: 0.75 };
                continue;
            }
            // check if the node is a rectangle and get a mutable reference to its properties
            if label_text.is_empty() {
                continue;
            }

            // temporary buffer to measure the text
            let scale = window.scale_factor() as f32;
            let mut temp_buffer =
                glyphon::Buffer::new(&mut font_system, Metrics::new(12.0 * scale, 12.0 * scale));

            temp_buffer.set_text(
                &mut font_system,
                &label_text,
                &Attrs::new(),
                Shaping::Advanced,
            );

            // Compute max line width using layout runs
            let text_width = temp_buffer
                .layout_runs()
                .map(|run| run.line_w)
                .fold(0.0, f32::max);

            temp_buffer.shape_until_scroll(&mut font_system, false);
            let mut capped_width = 44.0 * scale;
            let mut max_lines = 0;
            match node_shapes.get_mut(i) {
                Some(NodeShape::Rectangle { w, .. }) => {
                    let new_width_pixels = text_width;
                    *w = f32::min(new_width_pixels / (capped_width * 2.0) * 1.05, 2.0);
                    if matches!(
                        elements[i],
                        ElementType::Owl(OwlType::Edge(OwlEdge::InverseOf))
                    ) {
                        continue;
                    }
                    max_lines = 1;
                    capped_width *= 4.0;
                }
                Some(NodeShape::Circle { r }) => match elements[i] {
                    ElementType::Owl(OwlType::Node(node)) => match node {
                        OwlNode::EquivalentClass => continue,
                        OwlNode::Complement
                        | OwlNode::DisjointUnion
                        | OwlNode::UnionOf
                        | OwlNode::IntersectionOff => {
                            max_lines = 1;
                            capped_width = 79.0 * scale;
                        }
                        _ => {
                            max_lines = 2;
                            capped_width *= *r * 2.0 - 0.1;
                        }
                    },
                },
                None => {}
            }
            let current_text = label_text.clone();

            temp_buffer.set_wrap(&mut font_system, glyphon::Wrap::Word);
            temp_buffer.set_size(&mut font_system, Some(capped_width), None);

            // Initial shape
            temp_buffer.set_text(
                &mut font_system,
                &current_text,
                &Attrs::new(),
                Shaping::Advanced,
            );
            temp_buffer.shape_until_scroll(&mut font_system, false);

            fn line_count(buffer: &glyphon::Buffer) -> usize {
                buffer.layout_runs().count()
            }

            if line_count(&temp_buffer) > max_lines {
                let mut low = 0;
                let mut high = current_text.len();
                let mut truncated = current_text.clone();

                while low < high {
                    let mid = (low + high) / 2;
                    let candidate = format!("{}…", &current_text[..mid]);

                    temp_buffer.set_text(
                        &mut font_system,
                        &candidate,
                        &Attrs::new(),
                        Shaping::Advanced,
                    );
                    temp_buffer.shape_until_scroll(&mut font_system, false);
                    let lines = line_count(&temp_buffer);

                    if lines > max_lines {
                        high = mid;
                    } else {
                        truncated = candidate.clone();
                        low = mid + 1;
                    }
                }

                labels[i] = truncated;
            }
        }

        // Combine positions and types into NodeInstance entries
        let node_instance_buffer = vertex_buffer::create_node_instance_buffer(
            &device,
            &positions,
            &elements,
            &node_shapes,
            &hovered_index,
        );
        let num_instances = positions.len() as u32;

        // Create bind group 0 with only the resolution uniform (binding 0)
        let bind_group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &resolution_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &view_uniform_buffer,
                    offset: 0,
                    size: None,
                }),
            }],
            label: Some("group0_bind_group"),
        });

        // Include the bind group layout in the pipeline layout
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&resolution_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Node Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_node_main"),
                // include both the per-vertex quad buffer and the per-instance positions buffer
                buffers: &[Vertex::desc(), NodeInstance::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_node_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let num_vertices = VERTICES.len() as u32;

        let (edge_vertex_buffer, num_edge_vertices, arrow_vertex_buffer, num_arrow_vertices) =
            vertex_buffer::create_edge_vertex_buffer(
                &device,
                &edges,
                &positions,
                &node_shapes,
                &elements,
                1.0,
                &hovered_index,
            );

        let edge_shader =
            device.create_shader_module(wgpu::include_wgsl!("./renderer/edge_shader.wgsl"));

        let edge_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Edge Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &edge_shader,
                entry_point: Some("vs_edge_main"),
                buffers: &[vertex_buffer::EdgeVertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &edge_shader,
                entry_point: Some("fs_edge_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        let arrow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Arrow Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &edge_shader,
                entry_point: Some("vs_edge_main"),
                buffers: &[vertex_buffer::EdgeVertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &edge_shader,
                entry_point: Some("fs_edge_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        // Exclude properties without neighbors from simulator
        let mut neighbor_map: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for [start, center, end] in &edges {
            let mut neighbors = neighbor_map.get(&(*start, *end));
            if neighbors.is_none() {
                neighbors = neighbor_map.get(&(*end, *start));
            };
            match neighbors {
                Some(cur_neighbors) => {
                    let mut new_neighbors = cur_neighbors.clone();
                    new_neighbors.push(*center);
                    neighbor_map.insert(
                        (usize::min(*start, *end), usize::max(*start, *end)),
                        new_neighbors,
                    );
                }
                None => {
                    neighbor_map.insert(
                        (usize::min(*start, *end), usize::max(*start, *end)),
                        vec![*center],
                    );
                }
            }
        }
        let mut solitary_edges: Vec<[usize; 3]> = vec![];
        for [start, center, end] in &edges {
            let num_neighbors = neighbor_map
                .get(&(usize::min(*start, *end), usize::max(*start, *end)))
                .unwrap()
                .len();
            if num_neighbors < 2
                || (matches!(elements[*center], NodeType::InverseProperty) && num_neighbors <= 2)
            {
                solitary_edges.push([*start, *center, *end]);
            }
        }

        let mut sim_nodes = Vec::with_capacity(positions.len());
        for pos in positions.iter() {
            sim_nodes.push(Vec2::new(pos[0], pos[1]));
        }

        let mut sim_edges = Vec::with_capacity(edges.len());
        for [start, center, end] in &edges {
            sim_edges.push([*start as u32, *center as u32]);
            sim_edges.push([*center as u32, *end as u32]);
        }

        let mut sim_sizes = Vec::with_capacity(positions.len());
        for node_shape in node_shapes.clone() {
            match node_shape {
                NodeShape::Circle { r } => sim_sizes.push(r),
                NodeShape::Rectangle { w, .. } => sim_sizes.push(w),
            }
        }

        let simulator = Simulator::builder().build(sim_nodes, sim_edges, sim_sizes);

        // Radial menu setup

        let menu_shader =
            device.create_shader_module(wgpu::include_wgsl!("./renderer/radial_menu_shader.wgsl"));

        let menu_uniforms = MenuUniforms {
            center: [0.0, 0.0],
            radius_inner: 0.0,
            radius_outer: 0.0,
            hovered_segment: -1,
            _padding: [0; 7],
        };

        let radial_menu_uniform_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Menu Uniform Buffer"),
                contents: bytemuck::cast_slice(&[menu_uniforms]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let menu_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("menu_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let radial_menu_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &menu_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: radial_menu_uniform_buffer.as_entire_binding(),
            }],
            label: Some("menu_bind_group"),
        });

        let menu_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Menu Pipeline Layout"),
            bind_group_layouts: &[&resolution_bind_group_layout, &menu_bind_group_layout],
            push_constant_ranges: &[],
        });

        let radial_menu_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Radial Menu Pipeline"),
            layout: Some(&menu_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &menu_shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &menu_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let radial_menu_state = RadialMenuState {
            active: false,
            target_node_index: 0,
            center_world: Vec2::ZERO,
            radius_inner: 0.0,
            radius_outer: 0.0,
            hovered_segment: -1,
            menu_buffers: None,
        };

        // Glyphon: do not create heavy glyphon resources unless we have a non-zero surface.
        // Initialize them lazily below (or on first resize).
        let font_system = None;
        let swash_cache = None;
        let viewport = None;
        let atlas = None;
        let text_renderer = None;
        let text_buffers = None;
        let cardinality_text_buffers = None;

        // Create one text buffer per node with sample labels
        // text_buffers are created when glyphon is initialized (lazy).

        // If the surface is already configured (non-zero initial size), initialize glyphon now.
        // Helper below will create FontSystem, SwashCache, Viewport, TextAtlas, TextRenderer and buffers.

        let reader_id = EVENT_DISPATCHER
            .rend_chan
            .write()
            .unwrap()
            .register_reader();

        let mut state = Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: surface_configured,
            render_pipeline,
            vertex_buffer,
            num_vertices,
            view_uniform_buffer,
            view_uniforms,
            bind_group0,
            node_instance_buffer,
            num_instances,
            edge_pipeline,
            edge_vertex_buffer,
            num_edge_vertices,
            arrow_pipeline,
            arrow_vertex_buffer,
            num_arrow_vertices,
            positions: positions.to_vec(),
            labels,
            edges: edges.to_vec(),
            solitary_edges,
            node_types: elements.to_vec(),
            node_shapes,
            cardinalities,
            characteristics,
            simulator,
            paused: false,
            hovered_index,
            reader_id,
            last_fps_time: Instant::now(),
            fps_counter: 0,
            cursor_position: None,
            node_dragged: false,
            pan_active: false,
            last_pan_position: None,
            pan: Vec2::ZERO,
            zoom: 1.0,
            click_start_pos: None,
            font_system,
            swash_cache,
            viewport,
            atlas,
            text_renderer,
            text_buffers,
            cardinality_text_buffers,
            window,
            radial_menu_pipeline,
            radial_menu_bind_group,
            radial_menu_uniform_buffer,
            radial_menu_state,
        };

        if surface_configured {
            state.init_glyphon();
        }

        Ok(state)
    }
    // --- Coordinate Conversion Helpers ---

    /// Converts screen-space pixel coordinates (Y-down) to world-space coordinates (Y-up)
    fn screen_to_world(&self, screen_pos: Vec2) -> Vec2 {
        let screen_center = Vec2::new(
            self.config.width as f32 / 2.0,
            self.config.height as f32 / 2.0,
        );
        let screen_offset_px = screen_pos - screen_center;
        let world_rel_zoomed = Vec2::new(screen_offset_px.x, -screen_offset_px.y);
        let world_rel = world_rel_zoomed / self.zoom;
        world_rel + self.pan
    }

    /// Converts world-space coordinates (Y-up) to screen-space pixel coordinates (Y-down)
    fn world_to_screen(&self, world_pos: Vec2) -> Vec2 {
        let screen_center = Vec2::new(
            self.config.width as f32 / 2.0,
            self.config.height as f32 / 2.0,
        );
        let world_rel = world_pos - self.pan;
        let screen_offset_px = Vec2::new(world_rel.x, -world_rel.y) * self.zoom;
        screen_center + screen_offset_px
    }

    // Initialize glyphon resources and create one text buffer per node.
    fn init_glyphon(&mut self) {
        if self.font_system.is_some() {
            return; // already initialized
        }

        // Embed font bytes into the binary
        const DEFAULT_FONT_BYTES: &'static [u8] = include_bytes!("../../assets/DejaVuSans.ttf");

        let mut font_system = FontSystem::new_with_fonts(core::iter::once(
            glyphon::fontdb::Source::Binary(Arc::new(DEFAULT_FONT_BYTES.to_vec())),
        ));
        font_system.db_mut().set_sans_serif_family("DejaVu Sans");
        let swash_cache = SwashCache::new();

        let cache = Cache::new(&self.device);
        let viewport = Viewport::new(&self.device, &cache);

        let mut atlas = TextAtlas::new(&self.device, &self.queue, &cache, self.config.format);
        let text_renderer = TextRenderer::new(
            &mut atlas,
            &self.device,
            wgpu::MultisampleState::default(),
            None,
        );
        let scale = self.window.scale_factor() as f32;
        let mut text_buffers: Vec<GlyphBuffer> = Vec::new();
        for (i, label) in self.labels.clone().iter().enumerate() {
            let font_px = 12.0 * scale; // font size in physical pixels
            let line_px = 12.0 * scale;
            let mut buf = GlyphBuffer::new(&mut font_system, Metrics::new(font_px, line_px));
            // per-label size (in physical pixels)
            let (label_width, label_height) = match self.node_shapes[i] {
                NodeShape::Rectangle { w, .. } => {
                    // Calculate physical pixel width from shape's width multiplier
                    let mut height = match self.node_types[i] {
                        ElementType::Owl(OwlType::Edge(OwlEdge::InverseOf)) => 48.0,
                        _ => 12.0,
                    };
                    if self.characteristics.contains_key(&i) {
                        height += 24.0;
                    };
                    (w * 85.0 * scale, height * scale)
                }
                NodeShape::Circle { r } => {
                    let mut height = match self.node_types[i] {
                        ElementType::Owl(OwlType::Node(node)) => match node {
                            OwlNode::ExternalClass
                            | OwlNode::DeprecatedClass
                            | OwlNode::Complement
                            | OwlNode::EquivalentClass
                            | OwlNode::DisjointUnion
                            | OwlNode::IntersectionOff
                            | OwlNode::UnionOf => 36.0,
                            _ => 24.0,
                        },
                        ElementType::Owl(OwlType::Edge(edge)) => match edge {
                            OwlEdge::DisjointWith => 36.0,
                            _ => 24.0,
                        },
                        _ => 24.0,
                    };
                    if self.characteristics.contains_key(&i) {
                        height += 24.0;
                    };
                    let width = match self.node_types[i] {
                        ElementType::Owl(OwlType::Node(node)) => match node {
                            OwlNode::UnionOf
                            | OwlNode::DisjointUnion
                            | OwlNode::Complement
                            | OwlNode::IntersectionOff => 75.0,
                            _ => 85.0,
                        },
                        _ => 85.0,
                    };
                    (width * scale * r, height * scale)
                }
            };
            buf.set_size(&mut font_system, Some(label_width), Some(label_height));
            buf.set_wrap(&mut font_system, glyphon::Wrap::Word);
            // sample label using the NodeType
            let attrs = &Attrs::new().family(Family::SansSerif);
            let node_type_metrics = Metrics::new(font_px - 3.0, line_px);
            let mut owned_spans: Vec<(String, Attrs)> = Vec::new();
            match self.node_types[i] {
                ElementType::Owl(OwlType::Node(node)) => {
                    match node {
                        OwlNode::EquivalentClass => {
                            // TODO: Update when handling equivalent classes from ontology
                            let mut parts: Vec<&str> = label.split('\n').collect();
                            let label1 = parts.get(0).map_or("", |v| *v).to_string();
                            let eq_labels = parts.split_off(1);
                            if !eq_labels.is_empty() {
                                owned_spans.push((label1, attrs.clone()));
                                owned_spans.push(("\n".to_string(), attrs.clone()));
                                for (idx, eq) in eq_labels.iter().enumerate() {
                                    let mut s = eq.to_string();
                                    if idx + 1 < eq_labels.len() {
                                        s.push_str(", ");
                                    }
                                    owned_spans.push((s, attrs.clone()));
                                }
                            } else {
                                owned_spans.push((label1, attrs.clone()));
                            }
                        }
                        OwlNode::ExternalClass => {
                            owned_spans.push((label.to_string(), attrs.clone()));
                            owned_spans.push((
                                "\n(external)".to_string(),
                                attrs.clone().metrics(node_type_metrics),
                            ));
                        }
                        OwlNode::DeprecatedClass => {
                            owned_spans.push((label.to_string(), attrs.clone()));
                            owned_spans.push((
                                "\n(deprecated)".to_string(),
                                attrs.clone().metrics(node_type_metrics),
                            ));
                        }
                        OwlNode::Thing => {
                            owned_spans.push(("Thing".to_string(), attrs.clone()));
                        }
                        OwlNode::Complement => {
                            owned_spans.push((label.to_string(), attrs.clone()));
                            owned_spans.push(("\n\n¬".to_string(), attrs.clone()));
                        }
                        OwlNode::DisjointUnion => {
                            owned_spans.push((label.to_string(), attrs.clone()));
                            owned_spans.push(("\n\n1".to_string(), attrs.clone()));
                        }
                        OwlNode::IntersectionOff => {
                            owned_spans.push((label.to_string(), attrs.clone()));
                            owned_spans.push(("\n\n∩".to_string(), attrs.clone()));
                        }
                        OwlNode::UnionOf => {
                            owned_spans.push((label.to_string(), attrs.clone()));
                            owned_spans.push(("\n\n∪".to_string(), attrs.clone()));
                        }
                        _ => {
                            owned_spans.push((label.to_string(), attrs.clone()));
                        }
                    }
                }
                ElementType::Owl(OwlType::Edge(edge)) => match edge {
                    OwlEdge::InverseOf => {
                        if let Some(chs) = self.characteristics.get(&i) {
                            let (ch1, ch2) = chs.split_once("\n").unwrap_or((chs, ""));
                            let labels_vec: Vec<&str> = label.split('\n').collect();
                            let label1 = labels_vec.get(0).map_or("", |v| *v).to_string();
                            owned_spans.push((label1, attrs.clone()));
                            owned_spans.push((
                                format!("\n({})\n\n", ch1),
                                attrs.clone().metrics(node_type_metrics),
                            ));
                            let label2 = labels_vec.get(1).map_or("", |v| *v).to_string();
                            owned_spans.push((label2, attrs.clone()));
                            owned_spans.push((
                                format!("\n({})", ch2),
                                attrs.clone().metrics(node_type_metrics),
                            ));
                        } else {
                            let labels_vec: Vec<&str> = label.split('\n').collect();
                            let mut label1 = labels_vec.get(0).map_or("", |v| *v).to_string();
                            label1.push_str("\n\n\n");
                            let label2 = labels_vec.get(1).map_or("", |v| *v).to_string();
                            owned_spans.push((label1, attrs.clone()));
                            owned_spans.push((label2, attrs.clone()));
                        }
                    }
                    OwlEdge::DisjointWith => {
                        owned_spans.push((label.to_string(), attrs.clone()));
                        owned_spans.push((
                            "\n\n(disjoint)".to_string(),
                            attrs.clone().metrics(node_type_metrics),
                        ));
                    }
                    _ => {
                        owned_spans.push((label.to_string(), attrs.clone()));
                    }
                },
                ElementType::Rdfs(RdfsType::Edge(edge)) => match edge {
                    RdfsEdge::SubclassOf => {
                        owned_spans.push(("Subclass of".to_string(), attrs.clone()));
                    }
                    _ => {
                        owned_spans.push((label.to_string(), attrs.clone()));
                    }
                },
                _ => {
                    owned_spans.push((label.to_string(), attrs.clone()));
                }
            }

            // Append characteristic as a small parenthesized suffix if present.
            if !matches!(
                self.node_types[i],
                ElementType::Owl(OwlType::Edge(OwlEdge::InverseOf))
            ) {
                if let Some(ch) = self.characteristics.get(&i) {
                    owned_spans.push((
                        format!("\n({})", ch),
                        attrs.clone().metrics(node_type_metrics),
                    ));
                }
            }

            let spans: Vec<(&str, Attrs)> = owned_spans
                .iter()
                .map(|(s, a)| (s.as_str(), a.clone()))
                .collect();

            buf.set_rich_text(
                &mut font_system,
                spans,
                &attrs,
                Shaping::Advanced,
                Some(glyphon::cosmic_text::Align::Center),
            );
            buf.shape_until_scroll(&mut font_system, false);
            text_buffers.push(buf);
        }

        // cardinalities
        let mut cardinality_buffers: Vec<(usize, GlyphBuffer)> = Vec::new();
        for (edge_u32, (cardinality_min, cardinality_max)) in self.cardinalities.iter() {
            let edge_idx = *edge_u32 as usize;
            let font_px = 12.0 * scale;
            let line_px = 12.0 * scale;
            let mut buf = GlyphBuffer::new(&mut font_system, Metrics::new(font_px, line_px));
            let label_width = 48.0 * scale;
            let label_height = 24.0 * scale;
            buf.set_size(&mut font_system, Some(label_width), Some(label_height));

            let attrs = &Attrs::new().family(Family::SansSerif);
            let cardinality_text = match cardinality_max {
                Some(max) => format!("{}..{}", cardinality_min, max),
                None => format!("{}", cardinality_min),
            };
            let spans = vec![(cardinality_text.as_str(), attrs.clone())];
            buf.set_rich_text(
                &mut font_system,
                spans,
                &attrs,
                Shaping::Advanced,
                Some(glyphon::cosmic_text::Align::Center),
            );
            buf.shape_until_scroll(&mut font_system, false);

            cardinality_buffers.push((edge_idx, buf));
        }

        // Initialize Radial Menu Buffers
        let scale = self.window.scale_factor() as f32;
        let menu_attrs = Attrs::new()
            .family(Family::SansSerif)
            .weight(glyphon::Weight::BOLD);
        let menu_metrics = Metrics::new(14.0 * scale, 14.0 * scale);

        let mut buf_freeze = GlyphBuffer::new(&mut font_system, menu_metrics);
        buf_freeze.set_text(&mut font_system, "Freeze", &menu_attrs, Shaping::Advanced);
        buf_freeze.set_size(&mut font_system, Some(200.0 * scale), None);
        buf_freeze.shape_until_scroll(&mut font_system, false);

        let mut buf_sub = GlyphBuffer::new(&mut font_system, menu_metrics);
        buf_sub.set_text(&mut font_system, "Subgraph", &menu_attrs, Shaping::Advanced);
        buf_sub.set_size(&mut font_system, Some(200.0 * scale), None);
        buf_sub.shape_until_scroll(&mut font_system, false);

        self.radial_menu_state.menu_buffers = Some([buf_freeze, buf_sub]);

        self.font_system = Some(font_system);
        self.swash_cache = Some(swash_cache);
        self.viewport = Some(viewport);
        self.atlas = Some(atlas);
        self.text_renderer = Some(text_renderer);
        self.text_buffers = Some(text_buffers);
        self.cardinality_text_buffers = Some(cardinality_buffers);
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            let max_size = self.device.limits().max_texture_dimension_2d;
            self.config.width = min(width, max_size);
            self.config.height = min(height, max_size);
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;

            // Initialize glyphon now if not already done.
            if self.font_system.is_none() {
                self.init_glyphon();
            }

            EVENT_DISPATCHER.sim_chan.write().unwrap().single_write(
                SimulatorEvent::WindowResized {
                    width: width,
                    height: height,
                },
            );
        }
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.window.request_redraw();

        self.fps_counter += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_fps_time);

        if elapsed.as_secs_f32() >= 1.0 {
            let fps = self.fps_counter as f32 / elapsed.as_secs_f32();
            info!("FPS: {:.2}", fps);

            // Reset counters
            self.last_fps_time = now;
            self.fps_counter = 0;
        }

        if !self.is_surface_configured {
            return Ok(());
        }

        let scale = self.window.scale_factor() as f32;
        let vp_h_px = self.config.height as f32 * scale as f32;
        let vp_w_px = self.config.width as f32 * scale as f32;

        // PRE-CALCULATE LAYOUTS

        // Radial Menu Layout
        struct MenuLayout {
            top_rect: (f32, f32), // left, top
            bot_rect: (f32, f32), // left, top
            color_top: Color,
            color_bot: Color,
        }

        let menu_layout = if self.radial_menu_state.active {
            let menu = &self.radial_menu_state;
            let radius_mid = (menu.radius_inner + menu.radius_outer) / 2.0;

            let world_offset_y = radius_mid;

            let top_pos_world = menu.center_world + Vec2::new(0.0, world_offset_y);
            let bot_pos_world = menu.center_world - Vec2::new(0.0, world_offset_y);

            let top_screen = self.world_to_screen(top_pos_world);
            let bot_screen = self.world_to_screen(bot_pos_world);

            let color_top = if menu.hovered_segment == 0 {
                Color::rgb(255, 255, 255)
            } else {
                Color::rgb(50, 50, 50)
            };
            let color_bot = if menu.hovered_segment == 1 {
                Color::rgb(255, 255, 255)
            } else {
                Color::rgb(50, 50, 50)
            };

            Some((top_screen, bot_screen, color_top, color_bot))
        } else {
            None
        };

        struct LabelLayout {
            buffer_index: usize,
            left: f32,
            top: f32,
            scale_factor: f32,
            bounds: TextBounds,
        }
        let mut node_layouts: Vec<LabelLayout> = Vec::with_capacity(self.positions.len());

        if let Some(text_buffers) = self.text_buffers.as_ref() {
            for (i, buf) in text_buffers.iter().enumerate() {
                // Skip hidden nodes
                if i >= self.node_types.len() {
                    continue;
                }

                // Node logical coords
                let node_logical = match self.node_types[i] {
                    NodeType::InverseProperty => {
                        Vec2::new(self.positions[i][0], self.positions[i][1] + 18.0)
                    }
                    NodeType::Complement
                    | NodeType::DisjointUnion
                    | NodeType::Union
                    | NodeType::Intersection => {
                        Vec2::new(self.positions[i][0], self.positions[i][1] + 24.0)
                    }
                    _ => Vec2::new(self.positions[i][0], self.positions[i][1]),
                };

                let screen_pos_logical = self.world_to_screen(node_logical);
                let y_offset = if self.characteristics.contains_key(&i) {
                    6.0 * self.zoom
                } else {
                    0.0
                };
                let node_x_px = screen_pos_logical.x * scale;
                let node_y_px = screen_pos_logical.y * scale - y_offset;

                let (label_w_opt, label_h_opt) = buf.size();
                let label_w = label_w_opt.unwrap_or(96.0) as f32;
                let label_h = label_h_opt.unwrap_or(24.0) as f32;

                let scaled_label_w = label_w * self.zoom;
                let scaled_label_h = label_h * self.zoom;

                let left = node_x_px - scaled_label_w * 0.5;
                let line_height = 8.0;
                let top = match self.node_types[i] {
                    ElementType::Owl(OwlType::Node(OwlNode::EquivalentClass)) => {
                        node_y_px - 2.0 * line_height * self.zoom
                    }
                    _ => node_y_px - (line_height * scale) * self.zoom,
                };

                let right = left + scaled_label_w;
                let bottom = top + scaled_label_h;

                // Cull
                if right < 0.0 || left > vp_w_px || bottom < 0.0 || top > vp_h_px {
                    continue;
                }

                node_layouts.push(LabelLayout {
                    buffer_index: i,
                    left,
                    top,
                    scale_factor: self.zoom,
                    bounds: TextBounds {
                        left: left as i32,
                        top: top as i32,
                        right: right as i32,
                        bottom: bottom as i32,
                    },
                });
            }
        }

        // Cardinality Layouts
        let mut card_layouts: Vec<LabelLayout> = Vec::new();
        if let Some(card_buffers) = self.cardinality_text_buffers.as_ref() {
            for (list_idx, (edge_idx, buf)) in card_buffers.iter().enumerate() {
                if *edge_idx >= self.edges.len() {
                    continue;
                }

                let edge = self.edges[*edge_idx];
                let center_idx = edge[1];
                let end_idx = edge[2];

                let center_log_world =
                    Vec2::new(self.positions[center_idx][0], self.positions[center_idx][1]);
                let end_log_world =
                    Vec2::new(self.positions[end_idx][0], self.positions[end_idx][1]);

                let dir_world_x = center_log_world.x - end_log_world.x;
                let dir_world_y = center_log_world.y - end_log_world.y;

                let radius_world: f32 = 50.0;
                let padding_world = 15.0;
                let padding_world_rect = 45.0;

                let (offset_world_x, offset_world_y) = match self.node_shapes[end_idx] {
                    NodeShape::Circle { r } => (
                        (radius_world + padding_world) * r,
                        (radius_world + padding_world) * r,
                    ),
                    NodeShape::Rectangle { w, h } => {
                        let half_w_world = (w * 0.9 / 2.0) * radius_world;
                        let half_h_world = (h * 0.25 / 2.0) * radius_world;
                        let nx = if dir_world_x.abs() > 1e-6 {
                            dir_world_x.signum()
                        } else {
                            0.0
                        };
                        let ny = if dir_world_y.abs() > 1e-6 {
                            dir_world_y.signum()
                        } else {
                            0.0
                        };

                        let len_sq = dir_world_x * dir_world_x + dir_world_y * dir_world_y;
                        let len_world = len_sq.sqrt().max(1.0);
                        let nx_world = dir_world_x / len_world;
                        let ny_world = dir_world_y / len_world;

                        let tx = if nx_world.abs() > 1e-6 {
                            half_w_world / nx_world.abs()
                        } else {
                            f32::INFINITY
                        };
                        let ty = if ny_world.abs() > 1e-6 {
                            half_h_world / ny_world.abs()
                        } else {
                            f32::INFINITY
                        };
                        let t = tx.min(ty);
                        let dist = t + padding_world_rect;
                        (dist, dist)
                    }
                };

                let len_sq = dir_world_x * dir_world_x + dir_world_y * dir_world_y;
                let len_world = len_sq.sqrt().max(1.0);
                let nx_world = dir_world_x / len_world;
                let ny_world = dir_world_y / len_world;

                let card_world_x = end_log_world.x + nx_world * offset_world_x;
                let card_world_y = end_log_world.y + ny_world * offset_world_y;

                let card_screen_phys =
                    self.world_to_screen(Vec2::new(card_world_x, card_world_y)) * scale;
                let card_x_px = card_screen_phys.x;
                let card_y_px = card_screen_phys.y;

                let (label_w_opt, label_h_opt) = buf.size();
                let label_w = label_w_opt.unwrap_or(48.0) as f32;
                let label_h = label_h_opt.unwrap_or(24.0) as f32;
                let scaled_label_w = label_w * self.zoom;
                let scaled_label_h = label_h * self.zoom;

                let left = card_x_px - scaled_label_w * 0.5;
                let top = card_y_px - scaled_label_h * 0.5;
                let right = left + scaled_label_w;
                let bottom = top + scaled_label_h;

                if right < 0.0 || left > vp_w_px || bottom < 0.0 || top > vp_h_px {
                    continue;
                }

                card_layouts.push(LabelLayout {
                    buffer_index: list_idx,
                    left,
                    top,
                    scale_factor: self.zoom,
                    bounds: TextBounds {
                        left: left as i32,
                        top: top as i32,
                        right: right as i32,
                        bottom: bottom as i32,
                    },
                });
            }
        }

        // CONSTRUCT TEXT AREAS

        let mut areas: Vec<TextArea> = Vec::new();

        // Radial Menu Areas
        if let Some((top_screen, bot_screen, c_top, c_bot)) = menu_layout {
            if let Some(buffers) = &self.radial_menu_state.menu_buffers {
                // Buffer 0: Freeze (Top)
                let raw_width_0 = buffers[0]
                    .layout_runs()
                    .map(|run| run.line_w)
                    .fold(0.0, f32::max);

                // Scale text by zoom
                let text_scale = self.zoom;

                // Calculate centered position
                let left_0 = top_screen.x * scale - (raw_width_0 * text_scale) / 2.0;

                // Move text up slightly to center in the ring sector
                let top_0 = top_screen.y * scale - (10.0 * scale * text_scale);

                areas.push(TextArea {
                    buffer: &buffers[0],
                    left: left_0,
                    top: top_0,
                    scale: text_scale,
                    bounds: TextBounds {
                        left: 0,
                        top: 0,
                        right: i32::MAX,
                        bottom: i32::MAX,
                    },
                    default_color: c_top,
                    custom_glyphs: &[],
                });

                // Buffer 1: Subgraph (Bottom)
                let raw_width_1 = buffers[1]
                    .layout_runs()
                    .map(|run| run.line_w)
                    .fold(0.0, f32::max);

                let left_1 = bot_screen.x * scale - (raw_width_1 * text_scale) / 2.0;
                let top_1 = bot_screen.y * scale - (10.0 * scale * text_scale);

                areas.push(TextArea {
                    buffer: &buffers[1],
                    left: left_1,
                    top: top_1,
                    scale: text_scale,
                    bounds: TextBounds {
                        left: 0,
                        top: 0,
                        right: i32::MAX,
                        bottom: i32::MAX,
                    },
                    default_color: c_bot,
                    custom_glyphs: &[],
                });
            }
        }

        // Node Label Areas
        if let Some(text_buffers) = self.text_buffers.as_ref() {
            for layout in node_layouts {
                areas.push(TextArea {
                    buffer: &text_buffers[layout.buffer_index],
                    left: layout.left,
                    top: layout.top,
                    scale: layout.scale_factor,
                    bounds: layout.bounds,
                    default_color: Color::rgb(0, 0, 0),
                    custom_glyphs: &[],
                });
            }
        }

        // Cardinality Areas
        if let Some(card_buffers) = self.cardinality_text_buffers.as_ref() {
            for layout in card_layouts {
                areas.push(TextArea {
                    buffer: &card_buffers[layout.buffer_index].1,
                    left: layout.left,
                    top: layout.top,
                    scale: layout.scale_factor,
                    bounds: layout.bounds,
                    default_color: Color::rgb(0, 0, 0),
                    custom_glyphs: &[],
                });
            }
        }

        // RENDER

        // Prepare glyphon
        if let (
            Some(font_system),
            Some(swash_cache),
            Some(viewport),
            Some(atlas),
            Some(text_renderer),
        ) = (
            self.font_system.as_mut(),
            self.swash_cache.as_mut(),
            self.viewport.as_mut(),
            self.atlas.as_mut(),
            self.text_renderer.as_mut(),
        ) {
            viewport.update(
                &self.queue,
                Resolution {
                    width: vp_w_px as u32,
                    height: vp_h_px as u32,
                },
            );
            atlas.trim();
            if let Err(e) = text_renderer.prepare(
                &self.device,
                &self.queue,
                font_system,
                atlas,
                viewport,
                areas,
                swash_cache,
            ) {
                log::error!("glyphon prepare failed: {:?}", e);
            }
        }

        // Setup Render Pass
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.93,
                            g: 0.94,
                            b: 0.95,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            // Edges
            render_pass.set_pipeline(&self.edge_pipeline);
            render_pass.set_vertex_buffer(0, self.edge_vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &self.bind_group0, &[]);
            render_pass.draw(0..self.num_edge_vertices, 0..1);

            // Arrows
            render_pass.set_pipeline(&self.arrow_pipeline);
            render_pass.set_vertex_buffer(0, self.arrow_vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &self.bind_group0, &[]);
            render_pass.draw(0..self.num_arrow_vertices, 0..1);

            // Nodes
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.node_instance_buffer.slice(..));
            render_pass.set_bind_group(0, &self.bind_group0, &[]);

            if self.hovered_index >= 0 {
                // Draw 0..hovered_index
                render_pass.draw(0..self.num_vertices, 0..self.hovered_index as u32);
                // Draw (hovered_index+1)..num_instances
                render_pass.draw(
                    0..self.num_vertices,
                    (self.hovered_index as u32 + 1)..self.num_instances,
                );
                // Draw hovered node last (on top)
                render_pass.draw(
                    0..self.num_vertices,
                    self.hovered_index as u32..(self.hovered_index as u32 + 1),
                );
            } else {
                // No hover, draw all normally
                render_pass.draw(0..self.num_vertices, 0..self.num_instances);
            }

            // Radial Menu
            if self.radial_menu_state.active {
                render_pass.set_pipeline(&self.radial_menu_pipeline);
                render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                render_pass.set_bind_group(0, &self.bind_group0, &[]);
                render_pass.set_bind_group(1, &self.radial_menu_bind_group, &[]);
                render_pass.draw(0..6, 0..1);
            }

            // Text
            if let (Some(atlas), Some(viewport), Some(text_renderer)) = (
                self.atlas.as_mut(),
                self.viewport.as_ref(),
                self.text_renderer.as_mut(),
            ) {
                text_renderer
                    .render(atlas, viewport, &mut render_pass)
                    .unwrap();
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    pub fn update(&mut self) {
        self.handle_external_events();
        if !self.paused {
            self.simulator.tick();
        }

        // Update uniforms
        self.view_uniforms.pan = self.pan.to_array();
        self.view_uniforms.zoom = self.zoom;
        self.view_uniforms.resolution = [self.config.width as f32, self.config.height as f32];
        self.queue.write_buffer(
            &self.view_uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.view_uniforms]),
        );

        let positions = self.simulator.world.read_storage::<Position>();
        let entities = self.simulator.world.entities();
        for (i, (_, position)) in (&entities, &positions).join().enumerate() {
            self.positions[i] = [position.0.x, position.0.y];
        }

        for [start, center, end] in &self.solitary_edges {
            if *start == *end {
                continue;
            }
            let center_x = (self.positions[*start][0] + self.positions[*end][0]) / 2.0;
            let center_y = (self.positions[*start][1] + self.positions[*end][1]) / 2.0;
            self.positions[*center] = [center_x, center_y];
        }

        if self.radial_menu_state.active {
            let idx = self.radial_menu_state.target_node_index;
            if idx < self.positions.len() {
                self.radial_menu_state.center_world =
                    Vec2::new(self.positions[idx][0], self.positions[idx][1]);
            }
        }

        self.hovered_index = self.update_hover();

        let node_instances = vertex_buffer::build_node_instances(
            &self.positions,
            &self.node_types,
            &self.node_shapes,
            &self.hovered_index,
        );

        let (edge_vertices, arrow_vertices) = vertex_buffer::build_line_and_arrow_vertices(
            &self.edges,
            &self.positions,
            &self.node_shapes,
            &self.node_types,
            self.zoom,
            &self.hovered_index,
        );

        // Update menu hover state
        if self.radial_menu_state.active {
            if let Some(cursor) = self.cursor_position {
                // Determine sector
                let world_cursor = self.screen_to_world(cursor);
                let diff = world_cursor - self.radial_menu_state.center_world;

                let angle = diff.y.atan2(diff.x);
                let segment = if angle >= 0.0 { 0 } else { 1 };

                // Visual distance check
                self.radial_menu_state.hovered_segment = segment;

                // Update Uniform
                let uniforms = MenuUniforms {
                    center: self.radial_menu_state.center_world.to_array(),
                    radius_inner: self.radial_menu_state.radius_inner,
                    radius_outer: self.radial_menu_state.radius_outer,
                    hovered_segment: segment,
                    _padding: [0; 7],
                };
                self.queue.write_buffer(
                    &self.radial_menu_uniform_buffer,
                    0,
                    bytemuck::cast_slice(&[uniforms]),
                );
            }
        }

        self.queue.write_buffer(
            &self.edge_vertex_buffer,
            0,
            bytemuck::cast_slice(&edge_vertices),
        );
        self.queue.write_buffer(
            &self.arrow_vertex_buffer,
            0,
            bytemuck::cast_slice(&arrow_vertices),
        );
        self.queue.write_buffer(
            &self.node_instance_buffer,
            0,
            bytemuck::cast_slice(&node_instances),
        );
    }

    pub fn handle_external_events(&mut self) {
        for event in EVENT_DISPATCHER
            .rend_chan
            .read()
            .unwrap()
            .read(&mut self.reader_id)
        {
            match event {
                RenderEvent::ElementFiltered(node_type) => todo!(),
                RenderEvent::ElementShown(node_type) => todo!(),
                RenderEvent::Paused => self.paused = true,
                RenderEvent::Resumed => self.paused = false,
                RenderEvent::Zoomed(zoom) => {
                    let delta = MouseScrollDelta::PixelDelta(PhysicalPosition { x: 0.0, y: *zoom });
                    self.handle_scroll(delta);
                }
                RenderEvent::CenterGraph => self.center_graph(),
            }
        }
    }

    pub fn handle_key(&mut self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        match (code, is_pressed) {
            (KeyCode::Space, true) => {
                self.paused = !self.paused;
                self.window.request_redraw();
            }
            _ => {}
        }
    }
    //

    pub fn handle_mouse_key(&mut self, button: MouseButton, is_pressed: bool) {
        match (button, is_pressed) {
            (MouseButton::Left, true) => {
                // Mouse Down
                if let Some(pos) = self.cursor_position {
                    self.click_start_pos = self.cursor_position;

                    if !self.node_dragged && self.hovered_index == -1 {
                        self.pan_active = true;
                        self.last_pan_position = Some(pos);
                    }

                    if !self.pan_active {
                        self.node_dragged = true;
                        let pos_world = self.screen_to_world(pos);
                        EVENT_DISPATCHER
                            .sim_chan
                            .write()
                            .unwrap()
                            .single_write(SimulatorEvent::DragStart(pos_world));
                    }
                }
            }
            (MouseButton::Left, false) => {
                // Mouse Up: Handle Actions

                self.pan_active = false;
                self.last_pan_position = None;

                // Stop Dragging
                if self.node_dragged {
                    self.node_dragged = false;
                    EVENT_DISPATCHER
                        .sim_chan
                        .write()
                        .unwrap()
                        .single_write(SimulatorEvent::DragEnd);
                }

                if let (Some(start), Some(end)) = (self.click_start_pos, self.cursor_position) {
                    let distance = start.distance(end);

                    if distance < 3.0 {
                        // Case A: Menu is Active -> Check for selection or close
                        if self.radial_menu_state.active {
                            let world_pos = self.screen_to_world(end);
                            let dist = world_pos.distance(self.radial_menu_state.center_world);

                            let r_inner_world = self.radial_menu_state.radius_inner;
                            let r_outer_world = self.radial_menu_state.radius_outer;

                            if dist >= r_inner_world && dist <= r_outer_world {
                                // Clicked inside the ring
                                match self.radial_menu_state.hovered_segment {
                                    0 => self.freeze_node(self.radial_menu_state.target_node_index),
                                    1 => {
                                        self.subgraph_node(self.radial_menu_state.target_node_index)
                                    }
                                    _ => {}
                                }
                                self.radial_menu_state.active = false;
                            } else {
                                // Clicked outside or in the hole -> Close menu
                                self.radial_menu_state.active = false;
                            }

                            // Return early
                            return;
                        }

                        // Case B: Menu Not Active -> Check for Node Click
                        let hovered = self.update_hover();
                        if hovered != -1 {
                            // Open radial menu
                            let idx = hovered as usize;

                            // Determine size based on node shape
                            let (w_base, h_base) = match self.node_shapes[idx] {
                                NodeShape::Circle { r } => (r * 50.0, r * 50.0),
                                NodeShape::Rectangle { w, h } => (w * 42.5, h * 25.0),
                            };
                            let base_size = w_base.max(h_base);

                            self.radial_menu_state = RadialMenuState {
                                active: true,
                                target_node_index: idx,
                                center_world: Vec2::new(
                                    self.positions[idx][0],
                                    self.positions[idx][1],
                                ),
                                radius_inner: base_size + 15.0, // Slight gap
                                radius_outer: base_size + 65.0, // Ring width
                                hovered_segment: -1,
                                menu_buffers: self.radial_menu_state.menu_buffers.clone(),
                            };
                        } else {
                            // Clicked empty space
                            self.radial_menu_state.active = false;
                        }
                    }
                }
                self.click_start_pos = None;
            }
            _ => {}
        }
    }

    pub fn handle_cursor(&mut self, position: PhysicalPosition<f64>) {
        // (x,y) coords in pixels relative to the top-left corner of the window
        let pos_screen = Vec2::new(position.x as f32, position.y as f32);
        self.cursor_position = Some(pos_screen);

        if self.node_dragged {
            // Convert screen coordinates to world coordinates before sending to simulator
            let pos_world = self.screen_to_world(pos_screen);
            EVENT_DISPATCHER
                .sim_chan
                .write()
                .unwrap()
                .single_write(SimulatorEvent::Dragged(pos_world));
        } else if self.pan_active {
            if let Some(last_pos_screen) = self.last_pan_position {
                // 1. Get the world position of the cursor now.
                let world_pos_current = self.screen_to_world(pos_screen);

                // 2. Get the world position of the cursor at its last position.
                let world_pos_last = self.screen_to_world(last_pos_screen);

                // 3. The difference is the true world-space delta.
                let delta_world = world_pos_current - world_pos_last;

                // 4. Adjust the pan by this delta.
                self.pan -= delta_world;

                // 5. Update the last position.
                self.last_pan_position = Some(pos_screen);
            }
        }
    }

    pub fn handle_scroll(&mut self, delta: MouseScrollDelta) {
        let scroll_amount = match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(PhysicalPosition { y, .. }) => (y / 10.0) as f32, // Heuristic
        };
        if scroll_amount == 0.0 {
            return;
        }

        let cursor_pos_screen = match self.cursor_position {
            Some(pos) => pos,
            None => return,
        };

        // 1. Get world pos under cursor before zoom
        let world_pos_before = self.screen_to_world(cursor_pos_screen);

        // 2. Calculate new zoom
        let zoom_sensitivity = 0.05;
        self.zoom *= 1.0 - -scroll_amount * zoom_sensitivity; // scroll down = zoom in
        self.zoom = self.zoom.clamp(0.05, 4.0); // Min/max zoom levels

        // 3. We want the world_pos_before to stay at cursor_pos_screen.
        //    Find the new pan that makes this true.
        //    P_new = W_before - (S_cursor - C) / Z_new

        let screen_center = Vec2::new(
            self.config.width as f32 / 2.0,
            self.config.height as f32 / 2.0,
        );
        let screen_offset = cursor_pos_screen - screen_center;
        let world_rel_zoomed = Vec2::new(screen_offset.x, -screen_offset.y);

        self.pan = world_pos_before - (world_rel_zoomed / self.zoom);
    }

    fn center_graph(&mut self) {
        // Fetch boundary from QuadTree
        let (center, width, height) = {
            let quadtree = self.simulator.world.read_resource::<QuadTree>();
            (
                quadtree.boundary.center,
                quadtree.boundary.width,
                quadtree.boundary.height,
            )
        };

        // Prevent division by zero
        if width <= 0.1 || height <= 0.1 {
            return;
        }

        // Calculate Scale
        let padding_factor = 0.90;
        let screen_width = self.config.width as f32;
        let screen_height = self.config.height as f32;

        let zoom_x = screen_width / width;
        let zoom_y = screen_height / height;

        let new_zoom = zoom_x.min(zoom_y) * padding_factor;

        self.zoom = new_zoom;

        // Calculate Pan
        self.pan = center;

        self.window.request_redraw();
    }

    /// Update hovered node
    fn update_hover(&self) -> i32 {
        let cursor_pos = match self.cursor_position {
            Some(pos) => pos,
            None => return -1,
        };

        if self.node_dragged {
            return self.hovered_index;
        }

        // Convert screen pixel coordinates to world coordinates
        let world_pos = self.screen_to_world(cursor_pos);

        const BASE_RADIUS: f32 = 50.0;
        const BASE_WIDTH: f32 = 85.0;
        const BASE_HEIGHT: f32 = 50.0;

        let mut found_index = -1;

        // Iterate backwards to prioritize nodes drawn "on top"
        for (i, pos_array) in self.positions.iter().enumerate().rev() {
            // Skip nodes that aren't drawn
            if matches!(self.node_types[i], NodeType::NoDraw) {
                continue;
            }

            let pos = Vec2::new(pos_array[0], pos_array[1]);
            let delta = world_pos - pos;

            let is_hovered = match self.node_shapes[i] {
                NodeShape::Circle { r } => {
                    // Check distance against scaled radius
                    delta.length_squared() <= (BASE_RADIUS * r).powi(2)
                }
                NodeShape::Rectangle { w, h } => {
                    // Check Axis-Aligned Bounding Box (AABB)
                    let half_w = (BASE_WIDTH * w) / 2.0;
                    let half_h = (BASE_HEIGHT * h) / 2.0;

                    delta.x.abs() <= half_w && delta.y.abs() <= half_h
                }
            };

            if is_hovered {
                found_index = i as i32;
                break;
            }
        }

        found_index
    }

    fn add_menu_labels_to_areas(
        &mut self,
        areas: &mut Vec<TextArea>,
        font_system: &mut FontSystem,
    ) {
        let scale = self.window.scale_factor() as f32;
        let menu = &self.radial_menu_state;

        // Calculate text positions (World -> Screen)
        let radius_mid = (menu.radius_inner + menu.radius_outer) / 2.0;

        // Top Label (Freeze)
        let top_pos_world = menu.center_world + Vec2::new(0.0, radius_mid);
        let world_offset = radius_mid;

        let top_pos = self.world_to_screen(menu.center_world + Vec2::new(0.0, world_offset));
        let bot_pos = self.world_to_screen(menu.center_world - Vec2::new(0.0, world_offset));

        let mut create_buf = |text: &str, pos: Vec2, color: Color| {
            let mut buf = GlyphBuffer::new(font_system, Metrics::new(14.0 * scale, 14.0 * scale));
            buf.set_text(
                font_system,
                text,
                &Attrs::new()
                    .family(Family::SansSerif)
                    .weight(glyphon::Weight::BOLD),
                Shaping::Advanced,
            );
            buf.set_size(font_system, Some(100.0 * scale), None);
            buf.shape_until_scroll(font_system, false);

            // Center alignment logic
            let width = buf.layout_runs().map(|run| run.line_w).fold(0.0, f32::max);
            let left = pos.x * scale - width / 2.0;
            let top = pos.y * scale - (7.0 * scale); // approx half height

            TextArea {
                buffer: &buf,
                left,
                top,
                scale: 1.0,
                bounds: TextBounds {
                    left: 0,
                    top: 0,
                    right: i32::MAX,
                    bottom: i32::MAX,
                },
                default_color: color,
                custom_glyphs: &[],
            };
        };
    }

    fn freeze_node(&self, index: usize) {
        log::info!("Freeze Node: {}", self.labels[index]);
        // TODO: Implement freeze logic
    }

    fn subgraph_node(&self, index: usize) {
        log::info!("Create Subgraph from Node: {}", self.labels[index]);
        // TODO: Implement subgraph logic
    }
}
