mod node_types;
mod vertex_buffer;

use crate::web::{
    quadtree::{BoundingBox2D, QuadTree},
    renderer::node_types::NodeType,
    simulator::{Simulator, components::nodes::Position, ressources::events::SimulatorEvent},
};
use glam::Vec2;
use glyphon::{
    Attrs, Buffer as GlyphBuffer, Cache, Color, Family, FontSystem, Metrics, Resolution, Shaping,
    SwashCache, TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};
use log::info;
use specs::shrev::EventChannel;
use specs::{Join, WorldExt};
use std::sync::Arc;
use vertex_buffer::{NodeInstance, VERTICES, Vertex};
use wgpu::util::DeviceExt;
use winit::dpi::PhysicalPosition;
use winit::event::MouseButton;
use winit::{event_loop::ActiveEventLoop, keyboard::KeyCode, window::Window};

pub struct State {
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
    edges: Vec<[usize; 2]>,
    node_types: Vec<NodeType>,
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
    pub window: Arc<Window>,
}

impl State {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        info!("Building render state");

        let size = window.inner_size();

        // The instance is a handle to our GPU
        // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
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
                // we're building for the web we'll have to disable some.
                required_limits: if cfg!(target_arch = "wasm32") {
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
        let positions = [[0.0, 0.0], [250.0, 50.0], [50.0, 450.0]];
        let labels = vec![
            String::from("My class"),
            String::from("Loooooooong class"),
            String::from("Thing"),
        ];

        let node_types = [NodeType::Class, NodeType::ExternalClass, NodeType::Thing];

        // Combine positions and types into NodeInstance entries

        let node_instance_buffer =
            vertex_buffer::create_node_instance_buffer(&device, &positions, &node_types);
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
            label: Some("Render Pipeline"),
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

        // TODO: remove test edges after adding simulator
        let edges = [[0, 1]];
        let mut edge_positions: Vec<[[f32; 2]; 2]> = vec![];
        for edge in edges {
            edge_positions.push([positions[edge[0]], positions[edge[1]]]);
        }
        let edge_instance_buffer =
            vertex_buffer::create_edge_instance_buffer(&device, &edge_positions);
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
                front_face: wgpu::FrontFace::Cw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let sim_nodes = vec![
            Vec2::new(positions[0][0], positions[0][1]),
            Vec2::new(positions[1][0], positions[1][1]),
            Vec2::new(positions[2][0], positions[2][1]),
        ];
        let sim_edges = vec![[0.0 as u32, 1.0 as u32]];
        let mut simulator = Simulator::builder().build(sim_nodes, sim_edges);

        // Glyphon: do not create heavy glyphon resources unless we have a non-zero surface.
        // Initialize them lazily below (or on first resize).
        let font_system = None;
        let swash_cache = None;
        let viewport = None;
        let atlas = None;
        let text_renderer = None;
        let text_buffers = None;

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
            window,
        };

        if surface_configured {
            state.init_glyphon();
            let num_buffers = state.text_buffers.as_ref().map(|b| b.len()).unwrap_or(0);
            log::info!("Glyphon initialized: {} text buffers", num_buffers);
            if let Some(text_buffers) = state.text_buffers.as_ref() {
                for (i, buf) in text_buffers.iter().enumerate() {
                    log::info!(
                        "Buffer {} has {} glyphs",
                        i,
                        buf.layout_runs().map(|r| r.glyphs.len()).sum::<usize>()
                    );
                }
            }
        }

        Ok(state)
    }

    // Initialize glyphon resources and create one text buffer per node.
    fn init_glyphon(&mut self) {
        if self.font_system.is_some() {
            return; // already initialized
        }

        // Embed font bytes into the binary
        const DEFAULT_FONT_BYTES: &'static [u8] = include_bytes!("../../assets/DejaVuSans.ttf");
        // log::info!("Font size: {} bytes", DEFAULT_FONT_BYTES.len());

        let mut font_system = FontSystem::new_with_fonts(core::iter::once(
            glyphon::fontdb::Source::Binary(Arc::new(DEFAULT_FONT_BYTES.to_vec())),
        ));
        font_system.db_mut().set_sans_serif_family("DejaVu Sans");
        let swash_cache = SwashCache::new();

        let cache = Cache::new(&self.device);
        let viewport = Viewport::new(&self.device, &cache);

        let mut atlas = TextAtlas::new(
            &self.device,
            &self.queue,
            &cache,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        );
        let text_renderer = TextRenderer::new(
            &mut atlas,
            &self.device,
            wgpu::MultisampleState::default(),
            None,
        );
        let scale = self.window.scale_factor() as f32;
        let mut text_buffers: Vec<GlyphBuffer> = Vec::new();
        for label in self.labels.clone() {
            let font_px = 16.0 * scale; // font size in physical pixels
            let line_px = 28.0 * scale;
            let mut buf = GlyphBuffer::new(&mut font_system, Metrics::new(font_px, line_px));
            // per-label size (in physical pixels)
            // TODO: update if we implement dynamic node size
            let label_width = 96.0 * scale;
            let label_height = 24.0 * scale;
            buf.set_size(&mut font_system, Some(label_width), Some(label_height));
            // sample label using the NodeType
            let attrs = &Attrs::new().family(Family::SansSerif);
            buf.set_rich_text(
                &mut font_system,
                [(label.as_str(), attrs.clone())],
                &attrs,
                Shaping::Advanced,
                Some(glyphon::cosmic_text::Align::Center),
            );
            buf.shape_until_scroll(&mut font_system, false);
            text_buffers.push(buf);
        }

        self.font_system = Some(font_system);
        self.swash_cache = Some(swash_cache);
        self.viewport = Some(viewport);
        self.atlas = Some(atlas);
        self.text_renderer = Some(text_renderer);
        self.text_buffers = Some(text_buffers);
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
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
                let node_logical = self.positions[i];

                // convert node coords to physical pixels
                let node_x_px = node_logical[0] * scale;
                let node_y_px = vp_h_px - node_logical[1] * scale;

                let (label_w_opt, label_h_opt) = buf.size();
                let label_w = label_w_opt.unwrap_or(96.0) as f32;
                let label_h = label_h_opt.unwrap_or(24.0) as f32;

                // center horizontally on node
                let left = node_x_px - label_w * 0.5;

                // top = distance-from-top-in-physical-pixels
                let top = node_y_px - 16.0;

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
                            r: 0.84,
                            g: 0.87,
                            b: 0.88,
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
            self.simulator.dispatcher.dispatch(&self.simulator.world);
            self.simulator.world.maintain();
        }

        let positions = self.simulator.world.read_storage::<Position>();
        let entities = self.simulator.world.entities();

        for (i, (_, position)) in (&entities, &positions).join().enumerate() {
            self.positions[i] = [position.0.x, position.0.y];
        }
        // self.frame_count += 1;
        // let t = ((self.frame_count as f32) * 0.05).sin();

        // // Update node positions
        // self.positions[1] = [self.positions[1][0], self.positions[1][1] + t];

        let nodes: Vec<NodeInstance> = self
            .positions
            .iter()
            .zip(self.node_types.iter())
            .map(|(pos, ty)| NodeInstance {
                position: *pos,
                node_type: *ty as u32,
            })
            .collect();

        let mut edge_positions: Vec<[[f32; 2]; 2]> = Vec::with_capacity(self.edges.len());
        // Update edge endpoints from node positions
        for edge in &mut self.edges {
            edge_positions.push([self.positions[edge[0]], self.positions[edge[1]]]);
        }

        self.queue.write_buffer(
            &self.edge_instance_buffer,
            0,
            bytemuck::cast_slice(&edge_positions),
        );

        self.queue
            .write_buffer(&self.node_instance_buffer, 0, bytemuck::cast_slice(&nodes));
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
