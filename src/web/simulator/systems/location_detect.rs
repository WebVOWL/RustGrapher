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
use specs::shred;
use specs::shrev::EventChannel;
use specs::{
    Builder, Dispatcher, DispatcherBuilder, Entities, Join, LazyUpdate, ParJoin, Read, ReadExpect,
    ReadStorage, ReaderId, System, World, WorldExt, Write, WriteStorage,
};
use std::collections::HashMap;
use winit::dpi::PhysicalSize;

/// TODO: Implement using quadtree to improve performance
#[derive(Default)]
pub struct PointIntersect;

impl<'a> System<'a> for PointIntersect {
    type SystemData = (
        Entities<'a>,
        ReadStorage<'a, Position>,
        Read<'a, CursorPosition>,
        Write<'a, PointIntersection>,
    );

    fn run(&mut self, (entities, positions, cursor_position, mut intersection): Self::SystemData) {
        for (entity, pos) in (&*entities, &positions).join() {
            const NODE_RADIUS: f32 = 48.0;
            if (cursor_position.0.x - pos.0.x).powi(2) + (cursor_position.0.y - pos.0.y).powi(2)
                < NODE_RADIUS.powi(2)
            {
                // This node contains the cursor's position.
                // It is the node being dragged.
                intersection.0 = entity.id();
                info!("Point {0} intersect [{1}]", cursor_position.0, entity.id());
            }
        }
    }
}

// #[derive(SystemData)]
// pub struct PointIntersectSystemData<'a> {
//     entities: Entities<'a>,
//     positions: ReadStorage<'a, Position>,
//     cursor_position: Read<'a, CursorPosition>,
//     intersection: Write<'a, PointIntersection>,
// }

// /// TODO: Implement using quadtree to improve performance
// pub fn point_intersect(mut data: PointIntersectSystemData) {
//     for (entity, pos) in (&*data.entities, &data.positions).join() {
//         const NODE_RADIUS: f32 = 48.0;
//         if (data.cursor_position.0.x - pos.0.x).powi(2)
//             + (data.cursor_position.0.y - pos.0.y).powi(2)
//             < NODE_RADIUS.powi(2)
//         {
//             // This node contains the cursor's position.
//             // It is the node being dragged.
//             data.intersection.0 = entity.id();

//             info!(
//                 "Point {0} intersect [{1}]",
//                 data.cursor_position.0,
//                 entity.id()
//             );
//         }
//     }
// }
