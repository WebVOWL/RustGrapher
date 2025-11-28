use crate::web::{
    quadtree::QuadTree,
    simulator::{
        components::{
            edges::Connects,
            forces::NodeForces,
            nodes::{Mass, NodeState, Position, Velocity},
        },
        ressources::simulator_vars::{
            DeltaTime, GravityForce, QuadTreeTheta, RepelForce, SpringNeutralLength,
            SpringStiffness, WorldSize,
        },
    },
};
use glam::Vec2;
use rayon::prelude::*;
use specs::{Entities, Join, ParJoin, Read, ReadExpect, ReadStorage, System, WriteStorage};

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
        ReadStorage<'a, NodeState>,
        WriteStorage<'a, NodeForces>,
        ReadExpect<'a, QuadTree>,
        Read<'a, QuadTreeTheta>,
        Read<'a, RepelForce>,
    );

    fn run(
        &mut self,
        (entities, positions, masses, node_states, mut node_forces, quadtree, theta, repel_force): Self::SystemData,
    ) {
        (
            &*entities,
            &positions,
            &masses,
            &mut node_forces,
            &node_states,
        )
            .par_join()
            .for_each(|(entity, pos, mass, node_forces, state)| {
                if state.is_static() {
                    node_forces.0 = Vec2::ZERO;
                    return;
                }

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
        ReadStorage<'a, NodeState>,
        WriteStorage<'a, NodeForces>,
        Read<'a, GravityForce>,
        Read<'a, WorldSize>,
    );
    fn run(
        &mut self,
        (entities, positions, masses, node_states, mut forces, gravity_force, world_size): Self::SystemData,
    ) {
        (&entities, &positions, &masses, &mut forces, &node_states)
            .par_join()
            .for_each(|(entity, pos, mass, force, state)| {
                if state.is_static() {
                    return;
                }

                force.0 += -pos.0 * mass.0 * gravity_force.0;
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
        ReadStorage<'a, NodeState>,
        ReadStorage<'a, NodeForces>,
        WriteStorage<'a, Velocity>,
        ReadStorage<'a, Mass>,
        Read<'a, DeltaTime>,
    );

    fn run(
        &mut self,
        (entities, node_states, forces, mut velocities, masses, delta_time): Self::SystemData,
    ) {
        (&entities, &forces, &mut velocities, &masses, &node_states)
            .par_join()
            .for_each(|(entity, force, velocity, mass, state)| {
                if state.is_static() {
                    return;
                }

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
        ReadStorage<'a, NodeState>,
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
            node_states,
            mut forces,
            positions,
            spring_stiffness,
            spring_neutral_length,
        ): Self::SystemData,
    ) {
        let positions_storage = &positions;

        let force_updates: Vec<(specs::Entity, Vec2)> =
            (&entities, &positions, &connections, &node_states)
                .par_join()
                .fold(
                    || Vec::new(),
                    |mut acc, (entity, pos, connects, state)| {
                        if state.is_static() {
                            return acc;
                        }

                        let rb1 = entity;
                        for rb2 in &connects.targets {
                            // Look up the neighbor's position
                            if let Some(pos2_comp) = positions_storage.get(*rb2) {
                                let pos2 = pos2_comp.0;
                                let direction_vec = pos2 - pos.0;

                                let force_magnitude = spring_stiffness.0
                                    * (direction_vec.length() - spring_neutral_length.0);

                                let spring_force =
                                    direction_vec.normalize_or(Vec2::ZERO) * -force_magnitude;

                                acc.push((rb1, -spring_force));
                                acc.push((*rb2, spring_force));
                            }
                        }
                        acc
                    },
                )
                .reduce(
                    || Vec::new(),
                    |mut a, b| {
                        a.extend(b);
                        a
                    },
                );

        for (entity, force_vec) in force_updates {
            if let Some(force_comp) = forces.get_mut(entity) {
                force_comp.0 += force_vec;
            }
        }
    }
}
