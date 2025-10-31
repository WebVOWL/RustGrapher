mod node_shape;
mod node_types;
mod vertex_buffer;

use crate::web::{
    renderer::node_shape::NodeShape,
    renderer::node_types::NodeType,
    simulator::{Simulator, components::nodes::Position, ressources::events::SimulatorEvent},
};
use glam::Vec2;
use glyphon::{
    Attrs, Buffer as GlyphBuffer, BufferLine, Cache, Color, Family, FontSystem, Metrics,
    Resolution, Shaping, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};
use log::info;
use specs::shrev::EventChannel;
use specs::{Join, WorldExt};
use std::{cmp::min, sync::Arc};
use vertex_buffer::{NodeInstance, VERTICES, Vertex};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use wgpu::{Face, util::DeviceExt};
use winit::dpi::PhysicalPosition;
use winit::event::MouseButton;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::EventLoopExtWebSys;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

pub struct State {
    #[cfg(target_arch = "wasm32")]
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    num_vertices: u32,
    resolution_buffer: wgpu::Buffer,
    // bind group for group(0): binding 0 = resolution uniform only
    bind_group0: wgpu::BindGroup,
    // instance buffer with node positions (array<vec2<f32>>), bound as vertex buffer slot 1
    node_instance_buffer: wgpu::Buffer,
    // number of instances (length of node positions)
    num_instances: u32,
    edge_pipeline: wgpu::RenderPipeline,
    edge_instance_buffer: wgpu::Buffer,
    num_edge_instances: u32,

    // Node and edge coordinates in pixels
    positions: Vec<[f32; 2]>,
    labels: Vec<String>,
    edges: Vec<[usize; 3]>,
    node_types: Vec<NodeType>,
    node_shapes: Vec<NodeShape>,
    cardinalities: Vec<(u32, (String, Option<String>))>,
    frame_count: u64, // TODO: Remove after implementing simulator
    simulator: Simulator<'static, 'static>,
    paused: bool,

    // User input
    cursor_position: Option<Vec2>,
    node_dragged: bool,

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
}

impl State {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        // Check if we can use WebGPU (as if this writing it's only enabled in some browsers)
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
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Create resolution uniform buffer
        let resolution_data = [size.width as f32, size.height as f32, 0.0f32, 0.0f32];
        let resolution_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Resolution Buffer"),
            contents: bytemuck::cast_slice(&resolution_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // TODO: remove test code after implementing ontology loading
        let positions = [
            [50.0, 50.0],
            [250.0, 50.0],
            [450.0, 50.0],
            [250.0, 250.0],
            [650.0, 450.0],
            [700.0, 50.0],
            [850.0, 50.0],
            [1050.0, 50.0],
            [1050.0, 250.0],
            [450.0, 250.0],
            [650.0, 250.0],
            [850.0, 250.0],
            [850.0, 450.0],
            [1250.0, 250.0],
            [50.0, 500.0],
            [1250.0, 150.0],
            [1250.0, 350.0],
            [150.0, 150.0],
            [950.0, 100.0],
            [350.0, 50.0],
            [950.0, 25.0],
            [950.0, 50.0],
            [550.0, 450.0],
            [575.0, 50.0],
        ];
        let labels = vec![
            String::from("My class"),
            String::from("Rdfs class"),
            String::from("Rdfs resource"),
            String::from("Loooooooong class 1 2 3 4 5 6 7 8 9"),
            String::from("Thing"),
            String::from("Eq1\nEq2\nEq3"),
            String::from("Deprecated"),
            String::new(),
            String::from("Literal"),
            String::new(),
            String::from("DisjointUnion 1 2 3 4 5 6 7 8 9"),
            String::new(),
            String::new(),
            String::from("This Datatype is very long"),
            String::from("AllValues"),
            String::from("Property1"),
            String::from("Property2"),
            String::new(),
            String::new(),
            String::from("is a"),
            String::from("Deprecated"),
            String::from("External"),
            String::from("Symmetric"),
            String::from("Property\nInverseProperty"),
        ];

        let node_types = [
            NodeType::Class,
            NodeType::RdfsClass,
            NodeType::RdfsResource,
            NodeType::ExternalClass,
            NodeType::Thing,
            NodeType::EquivalentClass,
            NodeType::DeprecatedClass,
            NodeType::AnonymousClass,
            NodeType::Literal,
            NodeType::Complement,
            NodeType::DisjointUnion,
            NodeType::Intersection,
            NodeType::Union,
            NodeType::Datatype,
            NodeType::ValuesFrom,
            NodeType::DatatypeProperty,
            NodeType::DatatypeProperty,
            NodeType::SubclassOf,
            NodeType::DisjointWith,
            NodeType::RdfProperty,
            NodeType::DeprecatedProperty,
            NodeType::ExternalProperty,
            NodeType::ObjectProperty,
            NodeType::InverseProperty,
        ];

        let node_shapes = vec![
            NodeShape::Circle { r: 1.0 },
            NodeShape::Circle { r: 1.0 },
            NodeShape::Circle { r: 1.0 },
            NodeShape::Circle { r: 1.0 },
            NodeShape::Circle { r: 0.8 },
            NodeShape::Circle { r: 1.2 },
            NodeShape::Circle { r: 1.1 },
            NodeShape::Circle { r: 1.0 },
            NodeShape::Rectangle { w: 1.0, h: 1.0 },
            NodeShape::Circle { r: 1.0 },
            NodeShape::Circle { r: 1.0 },
            NodeShape::Circle { r: 1.0 },
            NodeShape::Circle { r: 1.0 },
            NodeShape::Rectangle { w: 1.0, h: 1.0 },
            NodeShape::Rectangle { w: 1.0, h: 1.0 },
            NodeShape::Rectangle { w: 1.0, h: 1.0 },
            NodeShape::Rectangle { w: 1.0, h: 1.0 },
            NodeShape::Rectangle { w: 1.0, h: 1.0 },
            NodeShape::Rectangle { w: 0.75, h: 1.0 },
            NodeShape::Rectangle { w: 0.6, h: 1.0 },
            NodeShape::Rectangle { w: 0.8, h: 1.0 },
            NodeShape::Rectangle { w: 0.8, h: 1.0 },
            NodeShape::Rectangle { w: 1.0, h: 1.0 },
            NodeShape::Rectangle { w: 1.0, h: 1.0 },
        ];

        let edges = [
            [0, 14, 1],
            [13, 15, 8],
            [8, 16, 13],
            [0, 17, 3],
            [6, 18, 7],
            [1, 19, 2],
            [10, 10, 11],
            [11, 11, 12],
            [6, 20, 7],
            [6, 21, 7],
            [4, 22, 4],
            [2, 23, 5],
        ];

        let cardinalities: Vec<(u32, (String, Option<String>))> = vec![
            (0, ("∀".to_string(), None)),
            (8, ("1".to_string(), None)),
            (1, ("1".to_string(), Some("10".to_string()))),
            (10, ("5".to_string(), Some("10".to_string()))),
        ];

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
            if let Some(NodeType::DisjointWith) = node_types.get(i) {
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
            let mut capped_width = 90.0;
            let mut max_lines = 0;
            match node_shapes.get_mut(i) {
                Some(NodeShape::Rectangle { w, .. }) => {
                    let new_width_pixels = text_width;
                    *w = f32::min(new_width_pixels / capped_width, 2.0);
                    if matches!(node_types[i], NodeType::InverseProperty) {
                        continue;
                    }
                    max_lines = 1;
                    capped_width = 180.0;
                }
                Some(NodeShape::Circle { .. }) => match node_types[i] {
                    NodeType::EquivalentClass => continue,
                    NodeType::Complement
                    | NodeType::DisjointUnion
                    | NodeType::Union
                    | NodeType::Intersection => {
                        max_lines = 1;
                        capped_width = 80.0;
                    }
                    _ => max_lines = 2,
                },
                None => {}
            }
            let mut current_text = label_text.clone();

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
            &node_types,
            &node_shapes,
        );
        let num_instances = positions.len() as u32;

        // Create bind group 0 with only the resolution uniform (binding 0)
        let bind_group0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &resolution_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &resolution_buffer,
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

        let edge_instance_buffer = vertex_buffer::create_edge_instance_buffer(
            &device,
            &edges,
            &positions,
            &node_shapes,
            &node_types,
        );
        let num_edge_instances = edges.len() as u32;

        let edge_shader =
            device.create_shader_module(wgpu::include_wgsl!("./renderer/edge_shader.wgsl"));

        let edge_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Edge Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &edge_shader,
                entry_point: Some("vs_edge_main"),
                buffers: &[Vertex::desc(), vertex_buffer::EdgeInstance::desc()],
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

        let mut sim_nodes = Vec::with_capacity(positions.len());
        for pos in positions {
            sim_nodes.push(Vec2::new(pos[0], pos[1]));
        }

        let mut sim_edges = Vec::with_capacity(edges.len());
        for edge in edges {
            sim_edges.push([edge[0].try_into().unwrap(), edge[1].try_into().unwrap()]);
        }
        let mut simulator = Simulator::builder().build(sim_nodes, sim_edges);

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

        let mut state = Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: surface_configured,
            render_pipeline,
            vertex_buffer,
            num_vertices,
            resolution_buffer,
            bind_group0,
            node_instance_buffer,
            num_instances,
            edge_pipeline,
            edge_instance_buffer,
            num_edge_instances,
            positions: positions.to_vec(),
            labels,
            edges: edges.to_vec(),
            node_types: node_types.to_vec(),
            node_shapes,
            cardinalities,
            frame_count: 0,
            simulator,
            paused: false,
            cursor_position: None,
            node_dragged: false,
            font_system,
            swash_cache,
            viewport,
            atlas,
            text_renderer,
            text_buffers,
            cardinality_text_buffers,
            window,
        };

        if surface_configured {
            state.init_glyphon();
        }

        Ok(state)
    }

    // Initialize glyphon resources and create one text buffer per node.
    fn init_glyphon(&mut self) {
        // TODO: Update handling of text overflow to use left alignment, and ellipses at end of string
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
                    let height = match self.node_types[i] {
                        NodeType::InverseProperty => 48.0,
                        _ => 12.0,
                    };
                    (w * 85.0 * scale, height * scale)
                }
                NodeShape::Circle { .. } => {
                    let height = match self.node_types[i] {
                        NodeType::ExternalClass
                        | NodeType::DeprecatedClass
                        | NodeType::EquivalentClass
                        | NodeType::DisjointWith
                        | NodeType::Union
                        | NodeType::DisjointUnion
                        | NodeType::Complement
                        | NodeType::Intersection => 36.0,
                        _ => 24.0,
                    };
                    let width = match self.node_types[i] {
                        NodeType::Union
                        | NodeType::DisjointUnion
                        | NodeType::Complement
                        | NodeType::Intersection => 75.0,
                        _ => 85.0,
                    };
                    (width * scale, height * scale)
                }
            };
            buf.set_size(&mut font_system, Some(label_width), Some(label_height));
            buf.set_wrap(&mut font_system, glyphon::Wrap::Word);
            // sample label using the NodeType
            let attrs = &Attrs::new().family(Family::SansSerif);
            let node_type_metrics = Metrics::new(font_px - 3.0, line_px);
            let mut all_owned_labels: Vec<String> = Vec::new();
            let spans = match self.node_types[i] {
                NodeType::EquivalentClass => {
                    // TODO: Update when handling equivalent classes from ontology
                    let mut labels: Vec<&str> = label.split('\n').collect();
                    let label1 = labels.get(0).map_or("", |v| *v);
                    let eq_labels = labels.split_off(1);
                    let (last_label, eq_labels) = eq_labels.split_last().unwrap();

                    let mut eq_labels_attributes: Vec<(&str, _)> = Vec::new();
                    for eq_label in eq_labels {
                        let mut extended_label = eq_label.to_string();
                        extended_label.push_str(", ");
                        all_owned_labels.push(extended_label);
                    }
                    all_owned_labels.push(last_label.to_string());

                    for extended_label in &all_owned_labels {
                        eq_labels_attributes.push((extended_label.as_str(), attrs.clone()));
                    }

                    let mut combined_labels = vec![(label1, attrs.clone()), ("\n", attrs.clone())];
                    combined_labels.append(&mut eq_labels_attributes);

                    combined_labels
                }
                NodeType::InverseProperty => {
                    let labels: Vec<&str> = label.split('\n').collect();
                    let mut label1 = labels.get(0).map_or("", |v| *v).to_string();
                    label1.push_str("\n\n\n");
                    let label2 = labels.get(1).map_or("", |v| *v);
                    all_owned_labels.push(label1);
                    all_owned_labels.push(label2.to_string());
                    let mut labels_attributes: Vec<(&str, _)> = Vec::new();
                    for extended_label in &all_owned_labels {
                        labels_attributes.push((extended_label.as_str(), attrs.clone()));
                    }
                    labels_attributes
                }
                NodeType::ExternalClass => vec![
                    (label.as_str(), attrs.clone()),
                    ("\n(external)", attrs.clone().metrics(node_type_metrics)),
                ],
                NodeType::DeprecatedClass => vec![
                    (label.as_str(), attrs.clone()),
                    ("\n(deprecated)", attrs.clone().metrics(node_type_metrics)),
                ],
                NodeType::DisjointWith => vec![
                    (label.as_str(), attrs.clone()),
                    ("\n\n(disjoint)", attrs.clone().metrics(node_type_metrics)),
                ],
                NodeType::Thing => vec![("Thing", attrs.clone())],
                NodeType::Complement => {
                    vec![(label.as_str(), attrs.clone()), ("\n\n¬", attrs.clone())]
                }
                NodeType::DisjointUnion => {
                    vec![(label.as_str(), attrs.clone()), ("\n\n1", attrs.clone())]
                }
                NodeType::Intersection => {
                    vec![(label.as_str(), attrs.clone()), ("\n\n∩", attrs.clone())]
                }
                NodeType::Union => vec![(label.as_str(), attrs.clone()), ("\n\n∪", attrs.clone())],
                NodeType::SubclassOf => vec![("Subclass of", attrs.clone())],
                _ => vec![(label.as_str(), attrs.clone())],
            };
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

            // update GPU resolution uniform
            let data = [width as f32, height as f32, 0.0f32, 0.0f32];
            self.queue
                .write_buffer(&self.resolution_buffer, 0, bytemuck::cast_slice(&data));

            self.simulator.send_event(SimulatorEvent::WindowResized {
                width: width,
                height: height,
            });
        }
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.window.request_redraw();

        if !self.is_surface_configured {
            return Ok(());
        }

        // If glyphon isn't initialized yet, skip text rendering for now.
        if let (
            Some(font_system),
            Some(swash_cache),
            Some(viewport),
            Some(atlas),
            Some(text_renderer),
            Some(text_buffers),
        ) = (
            self.font_system.as_mut(),
            self.swash_cache.as_mut(),
            self.viewport.as_mut(),
            self.atlas.as_mut(),
            self.text_renderer.as_mut(),
            self.text_buffers.as_ref(),
        ) {
            let scale = self.window.scale_factor() as f32;
            // physical viewport height in pixels:
            let vp_h_px = self.config.height as f32 * scale as f32;
            let vp_w_px = self.config.width as f32 * scale as f32;

            let mut areas: Vec<TextArea> = Vec::new();
            for (i, buf) in text_buffers.iter().enumerate() {
                // node logical coords
                let node_logical = match self.node_types[i] {
                    NodeType::InverseProperty => {
                        let position_x = self.positions[i][0];
                        let position_y = self.positions[i][1] + 18.0;
                        [position_x, position_y]
                    }
                    NodeType::Complement
                    | NodeType::DisjointUnion
                    | NodeType::Union
                    | NodeType::Intersection => {
                        let position_x = self.positions[i][0];
                        let position_y = self.positions[i][1] + 24.0;
                        [position_x, position_y]
                    }
                    _ => self.positions[i],
                };

                // convert node coords to physical pixels
                let node_x_px = node_logical[0] * scale;
                let node_y_px = vp_h_px - node_logical[1] * scale;

                let (label_w_opt, label_h_opt) = buf.size();
                let label_w = label_w_opt.unwrap_or(96.0) as f32;
                let label_h = label_h_opt.unwrap_or(24.0) as f32;

                // center horizontally on node
                let left = node_x_px - label_w * 0.5;

                let line_height = 8.0;
                // top = distance from top in physical pixels
                let top = match self.node_types[i] {
                    NodeType::EquivalentClass => node_y_px - 2.0 * line_height,
                    _ => node_y_px - line_height,
                };

                areas.push(TextArea {
                    buffer: buf,
                    left,
                    top,
                    scale: 1.0,
                    bounds: TextBounds {
                        left: left as i32,
                        top: top as i32,
                        right: (left + label_w) as i32,
                        bottom: (top + label_h) as i32,
                    },
                    default_color: Color::rgb(0, 0, 0),
                    custom_glyphs: &[],
                });
            }

            // cardinalities
            if let Some(card_buffers) = self.cardinality_text_buffers.as_ref() {
                for (edge_idx, buf) in card_buffers.iter() {
                    // make sure edge index is valid
                    if *edge_idx >= self.edges.len() {
                        continue;
                    }
                    let edge = self.edges[*edge_idx];
                    let center_idx = edge[1];
                    let end_idx = edge[2];

                    // node logical coords
                    let center_log = self.positions[center_idx];
                    let end_log = self.positions[end_idx];

                    // convert to physical pixels
                    let center_x_px = center_log[0] * scale;
                    let center_y_px = vp_h_px - center_log[1] * scale;
                    let end_x_px = end_log[0] * scale;
                    let end_y_px = vp_h_px - end_log[1] * scale;

                    // direction vector (center - start) in physical pixel space
                    let dir_x = center_x_px - end_x_px;
                    let dir_y = center_y_px - end_y_px;
                    let radius_pix = 50.0;

                    let (offset_px_x, offset_px_y) = match self.node_shapes[end_idx] {
                        NodeShape::Circle { r } => (
                            (radius_pix + 15.0) * r * scale,
                            (radius_pix + 15.0) * r * scale,
                        ),
                        NodeShape::Rectangle { w, h } => {
                            let half_w_px = (w * 0.9 / 2.0) * radius_pix;
                            let half_h_px = (h * 0.25 / 2.0) * radius_pix;

                            let len = (dir_x * dir_x + dir_y * dir_y).sqrt().max(1.0);
                            let nx = dir_x / len;
                            let ny = dir_y / len;

                            // Compute intersection with rectangle perimeter in direction (nx, ny)
                            let tx = if nx.abs() > 1e-6 {
                                half_w_px / nx.abs()
                            } else {
                                f32::INFINITY
                            };
                            let ty = if ny.abs() > 1e-6 {
                                half_h_px / ny.abs()
                            } else {
                                f32::INFINITY
                            };

                            // Intersection distance is the smaller of the two
                            let t = tx.min(ty);

                            // Add padding to place the label away from the edge
                            let padding = 45.0;
                            let dist = t + padding;

                            // Return the final scalar distance for both axes
                            (dist, dist)
                        }
                    };

                    // normalized direction (guard against zero-length)
                    let len = (dir_x * dir_x + dir_y * dir_y).sqrt().max(1.0);
                    let nx: f32 = dir_x / len;
                    let ny = dir_y / len;

                    // place label at end node plus offset along the center-->end angle
                    let card_x_px = end_x_px + nx * offset_px_x;
                    let card_y_px = end_y_px + ny * offset_px_y;

                    // compute bounds from buffer size and center the label
                    let (label_w_opt, label_h_opt) = buf.size();
                    let label_w = label_w_opt.unwrap_or(48.0) as f32;
                    let label_h = label_h_opt.unwrap_or(24.0) as f32;

                    let left = card_x_px - label_w * 0.5;
                    let top = card_y_px - label_h * 0.5;

                    areas.push(TextArea {
                        buffer: buf,
                        left,
                        top,
                        scale: 1.0,
                        bounds: TextBounds {
                            left: left as i32,
                            top: top as i32,
                            right: (left + label_w) as i32,
                            bottom: (top + label_h) as i32,
                        },
                        default_color: Color::rgb(0, 0, 0),
                        custom_glyphs: &[],
                    });
                }
            }

            viewport.update(
                &self.queue,
                Resolution {
                    width: vp_w_px as u32,
                    height: vp_h_px as u32,
                },
            );
            // Prepare glyphon for rendering
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
                            // Background color
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

            // Draw edges
            render_pass.set_pipeline(&self.edge_pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..)); // quad
            render_pass.set_vertex_buffer(1, self.edge_instance_buffer.slice(..));
            render_pass.set_bind_group(0, &self.bind_group0, &[]);
            render_pass.draw(0..self.num_vertices, 0..self.num_edge_instances);

            // Draw nodes
            render_pass.set_pipeline(&self.render_pipeline);
            // set vertex buffers: slot 0 = quad vertices, slot 1 = per-instance positions
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.node_instance_buffer.slice(..));
            // bind group 0 contains resolution uniform
            render_pass.set_bind_group(0, &self.bind_group0, &[]);
            // draw one quad per node position (instances)
            render_pass.draw(0..self.num_vertices, 0..self.num_instances);

            // Render glyphon text on top of nodes if initialized
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

        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    pub fn update(&mut self) {
        if !self.paused {
            self.simulator.tick();
        }

        let positions = self.simulator.world.read_storage::<Position>();
        let entities = self.simulator.world.entities();
        for (i, (_, position)) in (&entities, &positions).join().enumerate() {
            self.positions[i] = [position.0.x, position.0.y];
        }

        let node_instances = vertex_buffer::build_node_instances(
            &self.device,
            &self.positions,
            &self.node_types,
            &self.node_shapes,
        );

        let edge_instances = vertex_buffer::build_edge_instances(
            &self.edges,
            &self.positions,
            &self.node_shapes,
            &self.node_types,
        );

        self.queue.write_buffer(
            &self.edge_instance_buffer,
            0,
            bytemuck::cast_slice(&edge_instances),
        );

        self.queue.write_buffer(
            &self.node_instance_buffer,
            0,
            bytemuck::cast_slice(&node_instances),
        );
    }

    pub fn handle_key(&mut self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        match (code, is_pressed) {
            (KeyCode::Escape, true) => event_loop.exit(),
            (KeyCode::Space, true) => {
                self.paused = !self.paused;
                self.window.request_redraw();
            }
            _ => {}
        }
    }
    pub fn handle_mouse_key(&mut self, button: MouseButton, is_pressed: bool) {
        match (button, is_pressed) {
            (MouseButton::Left, true) => {
                // Begin node dragging on mouse click
                // if !self.paused {
                if let Some(pos) = self.cursor_position {
                    self.node_dragged = true;
                    self.simulator.send_event(SimulatorEvent::DragStart(pos));

                    // self.simulator.drag_start(pos);
                    // }
                }
            }
            (MouseButton::Left, false) => {
                // Stop node dragging on mouse release
                // if !self.paused {
                self.node_dragged = false;
                self.simulator.send_event(SimulatorEvent::DragEnd);

                // self.simulator.drag_end();
                // }
            }
            (MouseButton::Right, false) => {
                // Radial menu on mouse release
            }
            _ => {}
        }
    }
    pub fn handle_cursor(&mut self, position: PhysicalPosition<f64>) {
        // (x,y) coords in pixels relative to the top-left corner of the window
        self.cursor_position = Some(Vec2::new(position.x as f32, position.y as f32));
        if self.node_dragged {
            if let Some(pos) = self.cursor_position {
                self.simulator.send_event(SimulatorEvent::Dragged(pos));

                // self.simulator.dragging(pos);
            }
        }
    }
}
