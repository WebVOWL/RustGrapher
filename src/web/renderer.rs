mod node_shape;
mod node_types;
mod vertex_buffer;

use glam::Vec2;
use log::info;
use std::{cmp::min, sync::Arc};
use wgpu::{Face, util::DeviceExt};
#[cfg(target_arch = "wasm32")]
use winit::platform::web::EventLoopExtWebSys;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use vertex_buffer::{EdgeInstance, NodeInstance, VERTICES, Vertex};

use crate::web::{renderer::node_shape::NodeShape, simulator::Simulator};

use crate::web::renderer::node_types::NodeType;

use glyphon::{
    Attrs, Buffer as GlyphBuffer, BufferLine, Cache, Color, Family, FontSystem, Metrics,
    Resolution, Shaping, SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};

struct State {
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

    // Glyphon resources are initialized lazily when we have a non-zero surface.
    font_system: Option<FontSystem>,
    swash_cache: Option<SwashCache>,
    viewport: Option<Viewport>,
    atlas: Option<TextAtlas>,
    text_renderer: Option<TextRenderer>,
    // one glyphon buffer per node containing its text (created when glyphon is initialized)
    text_buffers: Option<Vec<GlyphBuffer>>,
    cardinality_text_buffers: Option<Vec<(usize, GlyphBuffer)>>,
    window: Arc<Window>,
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

        info!("Building render state");

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

        // TODO: remove test code after adding simulator
        let positions = [
            [50.0, 50.0],
            [250.0, 50.0],
            [450.0, 50.0],
            [250.0, 250.0],
            [650.0, 450.0],
            [650.0, 50.0],
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
            [550.0, 250.0],
            [350.0, 50.0],
            [750.0, 250.0],
            [850.0, 350.0],
            [750.0, 450.0],
        ];
        let labels = vec![
            String::from("My class"),
            String::from("Rdfs class"),
            String::from("Rdfs resource"),
            String::from("Loooooooong class"),
            String::from("Thing"),
            String::from("Eq1-Eq2-Eq3"),
            String::from("Deprecated"),
            String::new(),
            String::from("Literal"),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::from("Datatype"),
            String::from("AllValues"),
            String::from("Property1"),
            String::from("Property2"),
            String::new(),
            String::new(),
            String::from("is a"),
            String::from("Deprecated"),
            String::from("External"),
            String::from("Symmetric"),
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
            NodeShape::Rectangle { w: 1.5, h: 0.5 },
            NodeShape::Rectangle { w: 1.0, h: 1.0 },
            NodeShape::Rectangle { w: 1.0, h: 1.0 },
            NodeShape::Rectangle { w: 1.0, h: 1.0 },
            NodeShape::Rectangle { w: 1.0, h: 1.0 },
            NodeShape::Rectangle { w: 0.6, h: 2.0 },
            NodeShape::Rectangle { w: 0.6, h: 1.5 },
            NodeShape::Rectangle { w: 0.8, h: 1.0 },
            NodeShape::Rectangle { w: 0.8, h: 1.0 },
            NodeShape::Rectangle { w: 1.0, h: 1.0 },
        ];

        let edges = [
            [0, 14, 1],
            [13, 15, 8],
            [8, 16, 13],
            [0, 17, 3],
            [9, 18, 10],
            [1, 19, 2],
            [10, 20, 11],
            [11, 21, 12],
            [12, 22, 12],
        ];

        let cardinalities: Vec<(u32, (String, Option<String>))> = vec![
            (0, ("∀".to_string(), None)),
            (8, ("1".to_string(), None)),
            (1, ("1".to_string(), Some("10".to_string()))),
        ];

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

        let sim_nodes = vec![Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0)];
        let sim_edges = vec![Vec2::new(0.0, 1.0)];
        let mut simulator = Simulator::builder().build(sim_nodes, sim_edges);
        for _ in 0..3 {
            info!("Dispatch");
            simulator.dispatcher.dispatch(&simulator.world);
        }

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
            // TODO: update if we implement dynamic node size
            let label_width = 90.0 * scale;
            let label_height = match self.node_types[i] {
                NodeType::DisjointWith => 62.0 * scale,
                NodeType::ExternalClass | NodeType::DeprecatedClass | NodeType::EquivalentClass => {
                    48.0
                }
                _ => 24.0,
            } * scale;
            buf.set_size(&mut font_system, Some(label_width), Some(label_height));
            buf.set_wrap(&mut font_system, glyphon::Wrap::None);
            // sample label using the NodeType
            let attrs = &Attrs::new().family(Family::SansSerif);
            let node_type_metrics = Metrics::new(font_px - 3.0, line_px);
            let mut all_owned_eq_labels: Vec<String> = Vec::new();
            let spans = match self.node_types[i] {
                NodeType::EquivalentClass => {
                    // TODO: Update when handling equivalent classes from ontology
                    let mut labels: Vec<&str> = label.split('-').collect();
                    let label1 = labels.get(0).map_or("", |v| *v);
                    let eq_labels = labels.split_off(1);
                    let (last_label, eq_labels) = eq_labels.split_last().unwrap();

                    let mut eq_labels_attributes: Vec<(&str, _)> = Vec::new();
                    for eq_label in eq_labels {
                        let mut extended_label = eq_label.to_string();
                        extended_label.push_str(", ");
                        all_owned_eq_labels.push(extended_label);
                    }
                    all_owned_eq_labels.push(last_label.to_string());

                    for extended_label in &all_owned_eq_labels {
                        eq_labels_attributes.push((extended_label.as_str(), attrs.clone()));
                    }

                    let mut combined_labels = vec![(label1, attrs.clone()), ("\n", attrs.clone())];
                    combined_labels.append(&mut eq_labels_attributes);

                    combined_labels
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
                NodeType::Complement => vec![("¬", attrs.clone())],
                NodeType::DisjointUnion => vec![("1", attrs.clone())],
                NodeType::Intersection => vec![("∩", attrs.clone())],
                NodeType::Union => vec![("∪", attrs.clone())],
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
                let node_logical = self.positions[i];

                // convert node coords to physical pixels
                let node_x_px = node_logical[0] * scale;
                let node_y_px = vp_h_px - node_logical[1] * scale;

                let (label_w_opt, label_h_opt) = buf.size();
                let label_w = label_w_opt.unwrap_or(96.0) as f32;
                let label_h = label_h_opt.unwrap_or(24.0) as f32;

                // center horizontally on node
                let left = node_x_px - label_w * 0.5;

                let line_height = 8.0;
                // top = distance-from-top-in-physical-pixels
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

                    let offset_px = match self.node_shapes[end_idx] {
                        NodeShape::Circle { r } => 65.0 * r * scale,

                        NodeShape::Rectangle { w, h } => {
                            // TODO: update offset based on angle
                            // angle
                            // let angle = dir_y.atan2(dir_x);
                            60.0 * w * scale
                        }
                    };

                    // normalized direction (guard against zero-length)
                    let len = (dir_x * dir_x + dir_y * dir_y).sqrt().max(1.0);
                    let nx = dir_x / len;
                    let ny = dir_y / len;

                    // place label at end node plus offset along the center-->end angle
                    let card_x_px = end_x_px + nx * (offset_px + 10.0);
                    let card_y_px = end_y_px + ny * offset_px;

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

    fn update(&mut self) {
        self.frame_count += 1;
        let t = ((self.frame_count as f32) * 0.05).sin();

        self.positions[1] = [self.positions[1][0], self.positions[1][1] + t];
        self.positions[19] = [self.positions[19][0], self.positions[19][1] + t / 2.0];

        let x = ((self.frame_count as f32) * 0.025).sin() * 5.0;
        let y = ((self.frame_count as f32) * 0.025).cos() * 5.0;

        self.positions[8] = [self.positions[8][0] + x, self.positions[8][1] + y];

        self.positions[3] = [
            self.positions[3][0] + x / 2.0,
            self.positions[3][1] + x / 2.0,
        ];

        self.positions[17] = [
            self.positions[17][0] + x / 4.0,
            self.positions[17][1] + x / 4.0,
        ];

        self.positions[22] = [
            self.positions[22][0] + x / 2.0,
            self.positions[22][1] + y / 2.0,
        ];

        let node_instances = vertex_buffer::build_node_instances(
            &self.device,
            &self.positions,
            &self.node_types,
            &self.node_shapes,
        );

        let edge_instances = vertex_buffer::build_edge_instances(
            &self.device,
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

    fn handle_key(&self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        match (code, is_pressed) {
            (KeyCode::Escape, true) => event_loop.exit(),
            _ => {}
        }
    }
}

struct App {
    #[cfg(target_arch = "wasm32")]
    proxy: Option<winit::event_loop::EventLoopProxy<State>>,
    state: Option<State>,
}

impl App {
    pub fn new(#[cfg(target_arch = "wasm32")] event_loop: &EventLoop<State>) -> Self {
        #[cfg(target_arch = "wasm32")]
        let proxy = Some(event_loop.create_proxy());
        Self {
            state: None,
            #[cfg(target_arch = "wasm32")]
            proxy,
        }
    }
}

impl ApplicationHandler<State> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        #[allow(unused_mut)]
        let mut window_attributes = Window::default_attributes();

        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            use web_sys::wasm_bindgen::UnwrapThrowExt;
            use winit::platform::web::WindowAttributesExtWebSys;

            const CANVAS_ID: &str = "canvas";

            let window = web_sys::window().unwrap_throw();
            let document = window.document().unwrap_throw();
            let canvas = document.get_element_by_id(CANVAS_ID).unwrap_throw();
            let html_canvas_element = canvas.unchecked_into();
            window_attributes = window_attributes.with_canvas(Some(html_canvas_element));
        }

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        #[cfg(target_arch = "wasm32")]
        {
            // Run the future asynchronously and use the
            // proxy to send the results to the event loop
            if let Some(proxy) = self.proxy.take() {
                wasm_bindgen_futures::spawn_local(async move {
                    assert!(
                        proxy
                            .send_event(
                                State::new(window)
                                    .await
                                    .expect("Unable to create canvas!!!")
                            )
                            .is_ok()
                    )
                });
            }
        }
    }

    #[allow(unused_mut)]
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, mut event: State) {
        // This is where proxy.send_event() ends up
        #[cfg(target_arch = "wasm32")]
        {
            event.window.request_redraw();
            event.resize(
                event.window.inner_size().width,
                event.window.inner_size().height,
            );
        }
        self.state = Some(event);
    }
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(canvas) => canvas,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            // WindowEvent::ScaleFactorChanged {
            //     scale_factor,
            //     inner_size_writer,
            // } => {
            //     state.resize(width, height);
            // }
            WindowEvent::RedrawRequested => {
                state.update();
                match state.render() {
                    Ok(_) => {
                        // Update frame count
                        // self.dom.fps_counter.update();
                    }
                    // Reconfigure the surface if it's lost or outdated
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let size = state.window.inner_size();
                        state.resize(size.width, size.height);
                    }
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        log::error!("Out of memory while rendering!");
                        event_loop.exit();
                    }
                    Err(e) => {
                        log::error!("Unable to render {}", e);
                    }
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => state.handle_key(event_loop, code, key_state.is_pressed()),
            _ => {}
        }
    }
}

pub fn run() -> anyhow::Result<()> {
    let event_loop = EventLoop::with_user_event().build()?;
    let app = App::new(
        #[cfg(target_arch = "wasm32")]
        &event_loop,
    );

    #[cfg(target_arch = "wasm32")]
    event_loop.spawn_app(app);

    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = initRender)]
pub fn init_render() -> Result<(), wasm_bindgen::JsValue> {
    console_error_panic_hook::set_once();
    run().unwrap_throw();

    Ok(())
}
