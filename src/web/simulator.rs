pub mod components;
mod ressources;

use std::{collections::HashMap, ops::Deref};

use crate::web::{
    quadtree::{BoundingBox2D, QuadTree},
    simulator::{
        components::{
            edges::Connects,
            forces::NodeForces,
            nodes::{Dragged, Fixed, Mass, Position, Velocity},
        },
        ressources::{
            event_channels::{
                DampingChan, DeltaTimeChan, FreezeThresholdChan, GravityForceChan,
                QuadTreeThetaChan, RepelForceChan, SpringNeutralChan, SpringStiffnessChan,
            },
            simulator_vars::{
                Damping, DeltaTime, FreezeThreshold, GravityForce, QuadTreeTheta, RepelForce,
                SpringNeutralLength, SpringStiffness,
            },
        },
    },
};
use glam::Vec2;
use log::{debug, info};
use rayon::prelude::*;
use specs::{
    Builder, Dispatcher, DispatcherBuilder, Entities, Join, LazyUpdate, ParJoin, Read, ReadExpect,
    ReadStorage, ReaderId, System, World, WorldExt, Write, WriteStorage,
};
use specs::{prelude::*, world::LazyBuilder};
use web_sys::Event;
use winit::dpi::{PhysicalPosition, PhysicalSize};

#[derive(Default)]
struct EventManager {
    repel_force_reader: Option<ReaderId<f32>>,
    spring_stiffness_reader: Option<ReaderId<f32>>,
    spring_neutral_reader: Option<ReaderId<f32>>,
    gravity_force_reader: Option<ReaderId<f32>>,
    deltatime_reader: Option<ReaderId<f32>>,
    damping_reader: Option<ReaderId<f32>>,
    quadtree_theta_reader: Option<ReaderId<f32>>,
    freeze_threshold_reader: Option<ReaderId<f32>>,
}

impl<'a> System<'a> for EventManager {
    type SystemData = (
        Read<'a, RepelForceChan>,
        Read<'a, SpringStiffnessChan>,
        Read<'a, SpringNeutralChan>,
        Read<'a, GravityForceChan>,
        Read<'a, DeltaTimeChan>,
        Read<'a, DampingChan>,
        Read<'a, QuadTreeThetaChan>,
        Read<'a, FreezeThresholdChan>,
        Write<'a, RepelForce>,
        Write<'a, SpringStiffness>,
        Write<'a, SpringNeutralLength>,
        Write<'a, GravityForce>,
        Write<'a, DeltaTime>,
        Write<'a, Damping>,
        Write<'a, QuadTreeTheta>,
        Write<'a, FreezeThreshold>,
    );

    fn run(
        &mut self,
        (
            repel_events,
            spring_stiffness_events,
            spring_neutral_events,
            gravity_events,
            deltatime_events,
            damping_events,
            quadtree_theta_events,
            freeze_threshold_events,
            mut repel_force,
            mut spring_stiffness,
            mut spring_length,
            mut gravity_force,
            mut deltatime,
            mut damping,
            mut quadtree_theta,
            mut freeze_threshold,
        ): Self::SystemData,
    ) {
        if let Some(val) = repel_events
            .0
            .read(&mut self.repel_force_reader.as_mut().unwrap())
            .last()
        {
            repel_force.0 = *val;
        }
        if let Some(val) = spring_stiffness_events
            .0
            .read(&mut self.spring_stiffness_reader.as_mut().unwrap())
            .last()
        {
            spring_stiffness.0 = *val;
        }
        if let Some(val) = spring_neutral_events
            .0
            .read(&mut self.spring_neutral_reader.as_mut().unwrap())
            .last()
        {
            spring_length.0 = *val;
        }
        if let Some(val) = gravity_events
            .0
            .read(&mut self.gravity_force_reader.as_mut().unwrap())
            .last()
        {
            gravity_force.0 = *val;
        }
        if let Some(val) = deltatime_events
            .0
            .read(&mut self.deltatime_reader.as_mut().unwrap())
            .last()
        {
            deltatime.0 = *val;
        }
        if let Some(val) = damping_events
            .0
            .read(&mut self.damping_reader.as_mut().unwrap())
            .last()
        {
            damping.0 = *val;
        }
        if let Some(val) = quadtree_theta_events
            .0
            .read(&mut self.quadtree_theta_reader.as_mut().unwrap())
            .last()
        {
            quadtree_theta.0 = *val;
        }
        if let Some(val) = freeze_threshold_events
            .0
            .read(&mut self.freeze_threshold_reader.as_mut().unwrap())
            .last()
        {
            freeze_threshold.0 = *val;
        }
    }

    fn setup(&mut self, world: &mut World) {
        Self::SystemData::setup(world);
        self.repel_force_reader = Some(world.fetch_mut::<RepelForceChan>().0.register_reader());
        self.spring_stiffness_reader =
            Some(world.fetch_mut::<SpringStiffnessChan>().0.register_reader());
        self.spring_neutral_reader =
            Some(world.fetch_mut::<SpringNeutralChan>().0.register_reader());
        self.gravity_force_reader = Some(world.fetch_mut::<GravityForceChan>().0.register_reader());
        self.deltatime_reader = Some(world.fetch_mut::<DeltaTimeChan>().0.register_reader());
        self.damping_reader = Some(world.fetch_mut::<DampingChan>().0.register_reader());
        self.quadtree_theta_reader =
            Some(world.fetch_mut::<QuadTreeThetaChan>().0.register_reader());
        self.freeze_threshold_reader =
            Some(world.fetch_mut::<FreezeThresholdChan>().0.register_reader());
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

struct CalculateNodeForce;

impl CalculateNodeForce {
    /// Computes the repel force between two nodes.
    ///
    /// Usage in force calculations is as follows: Number 1 is the actual node,
    /// number 2 is the "fake", approximate node.
    fn repel_force(pos1: Vec2, pos2: Vec2, mass1: f32, mass2: f32, repel_force: f32) -> Vec2 {
        let dir_vec = pos2 - pos1;
        let length_sqr = dir_vec.length_squared();
        if length_sqr == 0.0 {
            return Vec2::ZERO;
        }

        let f = -repel_force * (mass1 * mass2).abs() / length_sqr;
        let dir_vec_normalized = dir_vec.normalize_or(Vec2::ZERO);
        let force = dir_vec_normalized * f;

        force.clamp(
            Vec2::new(-100000.0, -100000.0),
            Vec2::new(100000.0, 100000.0),
        )
    }
}

impl<'a> System<'a> for CalculateNodeForce {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, Position>,
        ReadStorage<'a, Mass>,
        ReadStorage<'a, Fixed>,
        ReadStorage<'a, Dragged>,
        WriteStorage<'a, NodeForces>,
        ReadExpect<'a, QuadTree>,
        Read<'a, QuadTreeTheta>,
        Read<'a, RepelForce>,
    );

    fn run(
        &mut self,
        (
            entities,
            positions,
            masses,
            fixed,
            dragged,
            mut node_forces,
            quadtree,
            theta,
            repel_force,
        ): Self::SystemData,
    ) {
        (
            &*entities,
            &positions,
            &masses,
            &mut node_forces,
            !&fixed,
            !&dragged,
        )
            .par_join()
            .for_each(|(entity, pos, mass, node_forces, _, _)| {
                let node_approximations = quadtree.stack(&pos.0, theta.0);

                for node_approximation in node_approximations {
                    node_forces.0 = Self::repel_force(
                        pos.0,
                        node_approximation.position(),
                        mass.0,
                        node_approximation.mass(),
                        repel_force.0,
                    );
                    info!(
                        "(CNF) [{0}] f: {1} | p: {2} | nap: {3} | m: {4} | nam: {5} | Rrf: {6}",
                        entity.id(),
                        node_forces.0,
                        pos.0,
                        node_approximation.position(),
                        mass.0,
                        node_approximation.mass(),
                        repel_force.0
                    );
                }
            });
    }
}

struct CalculateGravityForce;

impl<'a> System<'a> for CalculateGravityForce {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, Position>,
        ReadStorage<'a, Mass>,
        ReadStorage<'a, Fixed>,
        WriteStorage<'a, NodeForces>,
        Read<'a, GravityForce>,
    );

    // compute_center_gravity()
    fn run(
        &mut self,
        (entities, positions, masses, fixed, mut forces, gravity_force): Self::SystemData,
    ) {
        (&entities, &positions, &masses, !&fixed, &mut forces)
            .par_join()
            .for_each(|(entity, pos, mass, _, force)| {
                force.0 += -pos.0 * mass.0 * gravity_force.0;
                info!(
                    "(CGF) [{0}] f: {1} | p: {2} | m: {3} | g: {4}",
                    entity.id(),
                    force.0,
                    pos.0,
                    mass.0,
                    gravity_force.0
                );
            });
    }
}

struct ApplyNodeForce;

impl<'a> System<'a> for ApplyNodeForce {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, Fixed>,
        ReadStorage<'a, NodeForces>,
        WriteStorage<'a, Velocity>,
        ReadStorage<'a, Mass>,
        Read<'a, DeltaTime>,
    );

    fn run(
        &mut self,
        (entities, fixed, forces, mut velocities, masses, delta_time): Self::SystemData,
    ) {
        (&entities, !&fixed, &forces, &mut velocities, &masses)
            .par_join()
            .for_each(|(entity, _, force, velocity, mass)| {
                velocity.0 += force.0 / mass.0 * delta_time.0;
                info!(
                    "(ANF) [{0}] v: {1} | f: {2} | m: {3} | d: {4}",
                    entity.id(),
                    velocity.0,
                    force.0,
                    mass.0,
                    delta_time.0
                );
            });
    }
}

struct UpdateNodePosition;

impl<'a> System<'a> for UpdateNodePosition {
    type SystemData = (
        Entities<'a>,
        WriteStorage<'a, Position>,
        WriteStorage<'a, Velocity>,
        ReadStorage<'a, Fixed>,
        Read<'a, DeltaTime>,
        Read<'a, Damping>,
        Read<'a, FreezeThreshold>,
        Read<'a, LazyUpdate>,
    );

    fn run(
        &mut self,
        (
            entities,
            mut positions,
            mut velocities,
            fixed,
            delta_time,
            damping,
            freeze_threshold,
            updater,
        ): Self::SystemData,
    ) {
        // for (entity, velocity, _) in (&entities, &mut velocities, &fixeds).join() {
        //     if freeze_threshold.0 < velocity.0.abs().length() {
        //         // Update is only visible next dispatch
        //         updater.remove::<Fixed>(entity);
        //     }
        // }

        (&entities, &mut positions, &mut velocities, !&fixed)
            .par_join()
            .for_each(|(entity, pos, velocity, _)| {
                velocity.0 *= damping.0;

                pos.0 += velocity.0 * delta_time.0;

                info!(
                    "freeze_threshold.0, {0} > velocity.0.abs().length(), {1}",
                    freeze_threshold.0,
                    velocity.0.abs().length()
                );
                if freeze_threshold.0 > velocity.0.abs().length() {
                    // Update is only visible next dispatch
                    updater.insert(entity, Fixed);
                    velocity.0 = Vec2::ZERO;
                }
            });
    }
}

struct UpdateEdgeForces;

impl<'a> System<'a> for UpdateEdgeForces {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, Connects>,
        ReadStorage<'a, Fixed>,
        WriteStorage<'a, NodeForces>,
        ReadStorage<'a, Position>,
        Read<'a, SpringStiffness>,
        Read<'a, SpringNeutralLength>,
    );

    fn run(
        &mut self,
        (
            entities,
            connections,
            fixed,
            mut forces,
            positions,
            spring_stiffness,
            spring_neutral_length,
        ): Self::SystemData,
    ) {
        for (entity, position, connects, _) in
            (&*entities, &positions, &connections, !&fixed).join()
        {
            let rb1 = entity;
            for rb2 in &connects.targets {
                let direction_vec = positions.get(*rb2).unwrap().0 - position.0;

                let force_magnitude =
                    spring_stiffness.0 * (direction_vec.length() - spring_neutral_length.0);

                let spring_force = direction_vec.normalize_or(Vec2::ZERO) * -force_magnitude;

                let rb1_force = forces.get(rb1).unwrap().0;
                let rb2_force = forces.get(*rb2).unwrap().0;

                let _ = forces.insert(rb1, NodeForces(rb1_force.clone() - spring_force));
                let _ = forces.insert(*rb2, NodeForces(rb2_force.clone() + spring_force));
                info!(
                    "(UEF) S[{0}] f: {1} | T[{2}] f: {3} | dv: {4} | fm: {5} | sf: {6} | st: {7}",
                    entity.id(),
                    rb1_force,
                    rb2.id(),
                    rb2_force,
                    direction_vec,
                    force_magnitude,
                    spring_force,
                    spring_stiffness.0
                );
            }
        }
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

    /// Notify simulator that the user started dragging an element.
    pub fn drag_start(&self, entity_id: u32) {
        debug!("[{0}] Drag start", entity_id);
        let entity = self.world.entities().entity(entity_id);
        let updater = self.world.read_resource::<LazyUpdate>();
        updater.insert(entity, Dragged);
    }

    /// Notify simulator that the user stopped dragging an element.
    pub fn drag_end(&self, entity_id: u32) {
        debug!("[{0}] Drag end", entity_id);
        let entity = self.world.entities().entity(entity_id);
        let updater = self.world.read_resource::<LazyUpdate>();
        updater.remove::<Dragged>(entity);
    }

    /// Update simulator with cursor offset from last position update.
    pub fn dragged(&self, cursor_position: Vec2, window_size: PhysicalSize<u32>, entity_id: u32) {
        let normalized_width = cursor_position.x.clamp(0.0, window_size.width as f32);
        let normalized_height = ((cursor_position.y * -1.0) + window_size.height as f32)
            .clamp(0.0, window_size.height as f32);

        debug!("[{0}] Dragged position: {1}", entity_id, cursor_position);
        debug!("[{0}] Computed position: {1}", entity_id, cursor_position);

        let updater = self.world.read_resource::<LazyUpdate>();
        let entity = self.world.entities().entity(entity_id);
        updater.insert(
            entity,
            Position(Vec2::new(normalized_width, normalized_height)),
        );
    }

    // TODO: Implement with signals
    // pub fn find_closest_node_index(&self, loc: Vec3) -> Option<u32> {
    //     let rb_read = self.rigid_bodies.read().unwrap();
    //     let mut dist = f32::INFINITY;
    //     let mut index = 0;
    //     for (i, rb) in rb_read.iter().enumerate() {
    //         let new_dist = rb.position.distance(loc.xy());
    //         if new_dist < dist {
    //             dist = new_dist;
    //             index = i as u32;
    //         }
    //     }
    //     if dist.is_infinite() {
    //         None
    //     } else {
    //         Some(index)
    //     }
    // }

    // TODO: Implement with signals
    // pub fn set_node_location_by_index(&self, loc: Vec3, index: u32) {
    //     let mut rb_write = self.rigid_bodies.write().unwrap();
    //     rb_write[index as usize].position = loc.xy();
    // }
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
    /// Get a Instance of `SimulatorBuilder` with default values
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
            .with(EventManager::default(), "event_manager", &[])
            .with(QuadTreeConstructor, "quadtree_constructor", &[])
            .with(
                CalculateNodeForce,
                "calc_node_force",
                &["quadtree_constructor"],
            )
            .with(CalculateGravityForce, "calc_gravity_force", &[])
            .with(UpdateEdgeForces, "update_edge_forces", &["calc_node_force"])
            .with(
                ApplyNodeForce,
                "apply_node_force",
                &["calc_node_force", "calc_gravity_force"],
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
            gravity_force: 1.0,
            delta_time: 0.005,
            damping: 0.9,
            quadtree_theta: 0.75,
            freeze_thresh: 20.0,
        }
    }
}
