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
                CursorPosition, Damping, DeltaTime, FreezeThreshold, GravityForce,
                PointIntersection, QuadTreeTheta, RepelForce, SpringNeutralLength, SpringStiffness,
                WorldSize,
            },
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

pub struct UpdateNodePosition;

impl<'a> System<'a> for UpdateNodePosition {
    type SystemData = (
        Entities<'a>,
        WriteStorage<'a, Position>,
        WriteStorage<'a, Velocity>,
        ReadStorage<'a, Fixed>,
        ReadStorage<'a, Dragged>,
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
            dragged,
            delta_time,
            damping,
            freeze_threshold,
            updater,
        ): Self::SystemData,
    ) {
        for (entity, velocity, _) in (&entities, &mut velocities, &fixed).join() {
            if freeze_threshold.0 < velocity.0.abs().length() {
                // Update is only visible next dispatch
                updater.remove::<Fixed>(entity);
            }
        }

        (
            &entities,
            &mut positions,
            &mut velocities,
            !&fixed,
            !&dragged,
        )
            .par_join()
            .for_each(|(entity, pos, velocity, _, _)| {
                velocity.0 *= damping.0;

                pos.0 += velocity.0 * delta_time.0;

                // info!(
                //     "freeze_threshold.0, {0} > velocity.0.abs().length(), {1}",
                //     freeze_threshold.0,
                //     velocity.0.abs().length()
                // );
                if freeze_threshold.0 > velocity.0.abs().length() {
                    // Update is only visible next dispatch
                    updater.insert(entity, Fixed);
                    // velocity.0 = Vec2::ZERO;
                }
            });
    }
}

/// A node is being dragged.
#[derive(Default)]
pub struct DragStart;

impl<'a> System<'a> for DragStart {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, Fixed>,
        Read<'a, PointIntersection>,
        Read<'a, LazyUpdate>,
    );

    fn run(&mut self, (entities, fixed, dragged_entity, updater): Self::SystemData) {
        info!("[{0}] Drag start", dragged_entity.0.id());

        // Enable simulation when node is dragged
        (&entities, &fixed).par_join().for_each(|(entity, _)| {
            updater.remove::<Fixed>(entity);
        });

        // Except for the dragged node
        updater.insert(dragged_entity.0, Dragged);
    }
}

/// A node is no longer being dragged.
#[derive(Default)]
pub struct DragEnd;

impl<'a> System<'a> for DragEnd {
    type SystemData = (Entities<'a>, ReadStorage<'a, Dragged>, Read<'a, LazyUpdate>);

    fn run(&mut self, (entities, dragged, updater): Self::SystemData) {
        for (entity, _) in (&entities, &dragged).join() {
            info!("[{0}] Drag end", entity.id());
            updater.remove::<Dragged>(entity);
        }
    }
}

/// The position of the dragged node has changed.
#[derive(Default)]
pub struct Dragging;

impl<'a> System<'a> for Dragging {
    type SystemData = (
        ReadStorage<'a, Dragged>,
        WriteStorage<'a, Position>,
        Read<'a, CursorPosition>,
        Read<'a, WorldSize>,
    );

    fn run(&mut self, (dragged, mut position, cursor_position, world_size): Self::SystemData) {
        for (pos, _) in (&position, &dragged).join() {
            // Normalize position to wgpu's coordinate system
            let norm_width = cursor_position.0.x.clamp(0.0, world_size.width as f32);
            let norm_height = (-cursor_position.0.y + world_size.height as f32)
                .clamp(0.0, world_size.height as f32);
            pos.0 = Vec2::new(norm_width, norm_height);

            // info!("[{0}] Dragged position: {1}", entity_id, norm_pos);
        }
    }
}
