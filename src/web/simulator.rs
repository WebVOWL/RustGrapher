pub mod components;
pub mod ressources;
mod systems;

use crate::web::{
    quadtree::{BoundingBox2D, QuadTree},
    simulator::{
        components::{
            edges::Connects,
            forces::NodeForces,
            nodes::{Dragged, Fixed, Mass, Position, Velocity},
        },
        ressources::{
            events::SimulatorEvent,
            simulator_vars::{
                CursorPosition, Damping, DeltaTime, FreezeThreshold, GravityForce, QuadTreeTheta,
                RepelForce, SpringNeutralLength, SpringStiffness, WorldSize,
            },
        },
        systems::{
            force_compute::{
                ApplyNodeForce, ComputeEdgeForces, ComputeGravityForce, ComputeNodeForce,
            },
            location_detect::PointIntersect,
            position_update::{DragEnd, DragStart, Dragging, UpdateNodePosition},
        },
    },
};
use glam::Vec2;
use log::info;
use rayon::prelude::*;
use specs::prelude::*;
use specs::shrev::EventChannel;
use specs::{
    Builder, Dispatcher, DispatcherBuilder, Entities, Join, LazyUpdate, ParJoin, Read, ReadExpect,
    ReadStorage, ReaderId, System, World, WorldExt, Write, WriteStorage,
};
use std::collections::HashMap;
use winit::dpi::PhysicalSize;

#[derive(Default)]
struct EventManager<'a> {
    pub reader: Option<ReaderId<SimulatorEvent>>,
    simulator: Option<Simulator<'a, 'a>>,
}

impl<'a> System<'a> for EventManager<'a> {
    type SystemData = (
        Read<'a, EventChannel<SimulatorEvent>>,
        Write<'a, RepelForce>,
        Write<'a, SpringStiffness>,
        Write<'a, SpringNeutralLength>,
        Write<'a, GravityForce>,
        Write<'a, DeltaTime>,
        Write<'a, Damping>,
        Write<'a, QuadTreeTheta>,
        Write<'a, FreezeThreshold>,
        Write<'a, WorldSize>,
        Write<'a, CursorPosition>,
    );

    fn run(
        &mut self,
        (
            events,
            mut repel_force,
            mut spring_stiffness,
            mut spring_length,
            mut gravity_force,
            mut deltatime,
            mut damping,
            mut quadtree_theta,
            mut freeze_threshold,
            mut world_size,
            mut cursor_position,
        ): Self::SystemData,
    ) {
        for event in events.read(&mut self.reader.as_mut().unwrap()) {
            match event {
                SimulatorEvent::RepelForceUpdated(value) => repel_force.0 = *value,
                SimulatorEvent::SpringStiffnessUpdated(value) => spring_stiffness.0 = *value,
                SimulatorEvent::SpringNeutralLengthUpdated(value) => spring_length.0 = *value,
                SimulatorEvent::GravityForceUpdated(value) => gravity_force.0 = *value,
                SimulatorEvent::DeltaTimeUpdated(value) => deltatime.0 = *value,
                SimulatorEvent::DampingUpdated(value) => damping.0 = *value,
                SimulatorEvent::SimulationAccuracyUpdated(value) => quadtree_theta.0 = *value,
                SimulatorEvent::FreezeThresholdUpdated(value) => freeze_threshold.0 = *value,
                SimulatorEvent::WindowResized { width, height } => {
                    world_size.width = *width;
                    world_size.height = *height
                }
                SimulatorEvent::DragStart(cursor_pos) => {
                    cursor_position = cursor_pos;
                    if let Some(simulator) = self.simulator {
                        PointIntersect::default().run_now(&self.simulator.world);
                        DragStart::default().run_now(&self.simulator.world);
                    }
                }
                SimulatorEvent::DragEnd => DragEnd::default().run_now(&self.simulator.world),
                SimulatorEvent::Dragged(cursor_pos) => {
                    cursor_position = cursor_pos;
                    Dragging::default().run_now(&self.simulator.world);
                }
            }
        }
    }

    fn setup(&mut self, world: &mut World) {
        Self::SystemData::setup(world);
        self.reader = Some(
            world
                .fetch_mut::<EventChannel<SimulatorEvent>>()
                .register_reader(),
        );
    }
}

struct QuadTreeConstructor;

impl<'a> System<'a> for QuadTreeConstructor {
    type SystemData = (
        Write<'a, QuadTree>,
        ReadStorage<'a, Position>,
        ReadStorage<'a, Mass>,
    );

    fn run(&mut self, (mut quadtree, positions, masses): Self::SystemData) {
        let mut min = Vec2::INFINITY;
        let mut max = Vec2::NEG_INFINITY;
        let mut count = 0;

        // Join with masses to get node positions (as edges do not have the Mass component)
        for (i, (position, _)) in (&positions, &masses).join().enumerate() {
            min = min.min(position.0);
            max = max.max(position.0);
            count = i;
        }

        let dir = max - min;
        let boundary = BoundingBox2D::new((dir / 2.0) + min, dir[0], dir[1]);
        let mut new_tree = QuadTree::with_capacity(boundary, count);

        for (position, mass) in (&positions, &masses).join() {
            new_tree.insert(position.0, mass.0);
        }
        *quadtree = new_tree;
    }
}

//#[warn(dead_code = false)]
pub struct Simulator<'a, 'b> {
    pub world: World,
    pub dispatcher: Dispatcher<'a, 'b>,
}

impl<'a, 'b> Simulator<'a, 'b> {
    pub fn builder() -> SimulatorBuilder {
        SimulatorBuilder::default()
    }

    pub fn send_event(&self, event: SimulatorEvent) {
        let mut sim_channel = self.world.fetch_mut::<EventChannel<SimulatorEvent>>();
        sim_channel.single_write(event);
    }
}

/// Builder for `Simulator`
pub struct SimulatorBuilder {
    spring_stiffness: f32,
    spring_neutral_length: f32,
    delta_time: f32,
    gravity_force: f32,
    repel_force: f32,
    damping: f32,
    quadtree_theta: f32,
    freeze_thresh: f32,
}

impl SimulatorBuilder {
    /// Get an instance of `SimulatorBuilder` with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// How strong the spring force should be.
    ///
    /// Default: `100.0`
    pub fn spring_stiffness(mut self, spring_stiffness: f32) -> Self {
        self.spring_stiffness = spring_stiffness;
        self
    }

    /// Length of a edge in neutral position.
    ///
    /// If edge is shorter it pushers apart.
    /// If edge is longer it pulls together.
    ///
    /// Set to `0` if edges should always pull apart.
    ///
    /// Default: `2.0`
    pub fn spring_neutral_length(mut self, neutral_length: f32) -> Self {
        self.spring_neutral_length = neutral_length;
        self
    }

    /// How strong the pull to the center should be.
    ///
    /// Default: `1.0`
    pub fn gravity_force(mut self, gravity_force: f32) -> Self {
        self.gravity_force = gravity_force;
        self
    }

    /// How strong nodes should push others away.
    ///
    /// Default: `100.0`
    pub fn repel_force(mut self, repel_force_const: f32) -> Self {
        self.repel_force = repel_force_const;
        self
    }

    /// Amount of damping that should be applied to the node's movement
    ///
    /// `1.0` -> No Damping
    ///
    /// `0.0` -> No Movement
    ///
    /// Default: `0.9`
    pub fn damping(mut self, damping: f32) -> Self {
        self.damping = damping;
        self
    }

    /// How accurate the force calculations should be.
    /// Higher numbers result in more approximations but faster calculations.
    ///
    /// Value should be between 0.0 and 1.0.
    ///
    /// `0.0` -> No approximation -> n^2 brute force
    ///
    /// Default: `0.75`
    pub fn simulation_accuracy(mut self, theta: f32) -> Self {
        self.quadtree_theta = theta;
        self
    }

    /// Freeze nodes when their velocity falls below `freeze_thresh`.
    /// Set to `-1` to disable
    ///
    /// Default: `1e-2`
    pub fn freeze_threshold(mut self, freeze_thresh: f32) -> Self {
        self.freeze_thresh = freeze_thresh;
        self
    }

    /// How much time a simulation step should simulate. (euler method)
    ///
    /// Bigger time steps result in faster simulations, but less accurate or even wrong simulations.
    ///
    /// `delta_time` is in seconds
    ///
    /// Panics when delta time is `0` or below
    ///
    /// Default: `0.005`
    pub fn delta_time(mut self, delta_time: f32) -> Self {
        if delta_time <= 0.0 {
            panic!("delta_time may not be 0 or below!");
        }
        self.delta_time = delta_time;
        self
    }

    /// Constructs a instance of `Simulator`
    pub fn build<'a, 'b>(self, nodes: Vec<Vec2>, edges: Vec<[u32; 2]>) -> Simulator<'a, 'b> {
        let mut world = World::new();
        let mut dispatcher = DispatcherBuilder::new()
            .with(
                EventManager {
                    simulator: self,
                    ..Default::default()
                },
                "event_manager",
                &[],
            )
            .with(QuadTreeConstructor, "quadtree_constructor", &[])
            .with(
                ComputeNodeForce,
                "compute_gravity_force",
                &["quadtree_constructor"],
            )
            .with(ComputeGravityForce, "compute_gravity_force", &[])
            .with(
                ComputeEdgeForces,
                "compute_edge_forces",
                &["compute_gravity_force"],
            )
            .with(
                ApplyNodeForce,
                "apply_node_force",
                &["compute_gravity_force", "compute_gravity_force"],
            )
            .with(
                UpdateNodePosition,
                "update_node_position",
                &["apply_node_force"],
            )
            .build();

        dispatcher.setup(&mut world);
        Self::create_entities(&mut world, nodes, edges);
        self.add_ressources(&mut world);

        Simulator {
            world: world,
            dispatcher: dispatcher,
        }
    }

    fn add_ressources(self: Self, world: &mut World) {
        world.insert(RepelForce(self.repel_force));
        world.insert(SpringStiffness(self.spring_stiffness));
        world.insert(SpringNeutralLength(self.spring_neutral_length));
        world.insert(GravityForce(self.gravity_force));
        world.insert(DeltaTime(self.delta_time));
        world.insert(Damping(self.damping));
        world.insert(QuadTreeTheta(self.quadtree_theta));
        world.insert(FreezeThreshold(self.freeze_thresh));
        world.insert(QuadTree::default());
        world.insert(WorldSize::default());
        world.insert(CursorPosition::default());
    }

    fn create_entities(world: &mut World, nodes: Vec<Vec2>, edges: Vec<[u32; 2]>) {
        let mut node_entities = Vec::with_capacity(nodes.len());

        // Create node entities
        for node in nodes {
            let node_entity = world
                .create_entity()
                .with(Position(node))
                .with(Velocity::default())
                .with(Mass::default())
                .with(NodeForces::default())
                .build();
            node_entities.push(node_entity);
        }

        // Create edge components between nodes
        let mut edge_components: HashMap<u32, Connects> = HashMap::new();
        for edge in edges.iter() {
            // x == edge[0], y == edge[1]
            if let Some(connections) = edge_components.get_mut(&edge[0]) {
                connections.targets.push(node_entities[edge[1] as usize])
            } else {
                let new_connects = Connects {
                    targets: vec![node_entities[edge[1] as usize]],
                };
                edge_components.insert(edge[0], new_connects);
            }
        }

        // Add edge components to node entities
        let updater = world.read_resource::<LazyUpdate>();
        for (src, targets) in edge_components {
            let node = node_entities[src as usize];
            updater.insert(node, targets);
        }
    }
}

impl Default for SimulatorBuilder {
    /// Get a Instance of `SimulatorBuilder` with default values
    fn default() -> Self {
        Self {
            repel_force: 100.0,
            spring_stiffness: 100.0,
            spring_neutral_length: 200.0,
            gravity_force: 10.0,
            delta_time: 0.005,
            damping: 0.9,
            quadtree_theta: 0.75,
            freeze_thresh: 2.0,
        }
    }
}
