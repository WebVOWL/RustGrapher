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
                Damping, DeltaTime, FreezeThreshold, GravityForce, QuadTreeTheta, RepelForce,
                SpringNeutralLength, SpringStiffness, WorldSize,
            },
        },
        systems::position_compute::norm_pos_center,
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

pub struct ComputeNodeForce;

impl ComputeNodeForce {
    /// Computes the repel force between two nodes.
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

impl<'a> System<'a> for ComputeNodeForce {
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

                node_forces.0 = Vec2::ZERO;
                for node_approximation in node_approximations {
                    node_forces.0 += Self::repel_force(
                        pos.0,
                        node_approximation.position(),
                        mass.0,
                        node_approximation.mass(),
                        repel_force.0,
                    );
                    // info!(
                    //     "(CNF) [{0}] f: {1} | p: {2} | nap: {3} | m: {4} | nam: {5} | Rrf: {6}",
                    //     entity.id(),
                    //     node_forces.0,
                    //     pos.0,
                    //     node_approximation.position(),
                    //     mass.0,
                    //     node_approximation.mass(),
                    //     repel_force.0
                    // );
                }
            });
    }
}

/// Computes center gravity of the world.
/// All elements will gravitate towards this point.
pub struct ComputeGravityForce;

impl<'a> System<'a> for ComputeGravityForce {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, Position>,
        ReadStorage<'a, Mass>,
        ReadStorage<'a, Fixed>,
        ReadStorage<'a, Dragged>,
        WriteStorage<'a, NodeForces>,
        Read<'a, GravityForce>,
        Read<'a, WorldSize>,
    );
    fn run(
        &mut self,
        (entities, positions, masses, fixed, dragged, mut forces, gravity_force, world_size): Self::SystemData,
    ) {
        (
            &entities,
            &positions,
            &masses,
            &mut forces,
            !&fixed,
            !&dragged,
        )
            .par_join()
            .for_each(|(entity, pos, mass, force, _, _)| {
                let norm_pos = norm_pos_center(pos.0, [world_size.width, world_size.height]);
                force.0 += norm_pos * mass.0 * gravity_force.0;
                // info!(
                //     "(CGF) [{0}] f: {1} | p: {2} | m: {3} | g: {4} | np: {5}",
                //     entity.id(),
                //     force.0,
                //     pos.0,
                //     mass.0,
                //     gravity_force.0,
                //     // dist_to_center,
                //     norm_pos
                // );
            });
    }
}

pub struct ApplyNodeForce;

impl<'a> System<'a> for ApplyNodeForce {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, Fixed>,
        ReadStorage<'a, Dragged>,
        ReadStorage<'a, NodeForces>,
        WriteStorage<'a, Velocity>,
        ReadStorage<'a, Mass>,
        Read<'a, DeltaTime>,
    );

    fn run(
        &mut self,
        (entities, fixed, dragged, forces, mut velocities, masses, delta_time): Self::SystemData,
    ) {
        (
            &entities,
            &forces,
            &mut velocities,
            &masses,
            !&fixed,
            !&dragged,
        )
            .par_join()
            .for_each(|(entity, force, velocity, mass, _, _)| {
                velocity.0 += force.0 / mass.0 * delta_time.0;
                // info!(
                //     "(ANF) [{0}] v: {1} | f: {2} | m: {3} | d: {4}",
                //     entity.id(),
                //     velocity.0,
                //     force.0,
                //     mass.0,
                //     delta_time.0
                // );
            });
    }
}

pub struct ComputeEdgeForces;

impl<'a> System<'a> for ComputeEdgeForces {
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
                // info!(
                //     "(UEF) S[{0}] f: {1} | T[{2}] f: {3} | dv: {4} | fm: {5} | sf: {6} | st: {7}",
                //     entity.id(),
                //     rb1_force,
                //     rb2.id(),
                //     rb2_force,
                //     direction_vec,
                //     force_magnitude,
                //     spring_force,
                //     spring_stiffness.0
                // );
            }
        }
    }
}
