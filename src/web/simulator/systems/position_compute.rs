use crate::web::simulator::{
    components::nodes::{Mass, Position},
    ressources::simulator_vars::{CursorPosition, PointIntersection, WorldSize},
};
use glam::Vec2;
use log::info;
use specs::prelude::*;
use specs::shred;
use specs::{Entities, Join, Read, ReadStorage, Write};

#[derive(SystemData)]
pub struct DistanceSystemData<'a> {
    entities: Entities<'a>,
    positions: ReadStorage<'a, Position>,
    cursor_position: Read<'a, CursorPosition>,
    world_size: Read<'a, WorldSize>,
    intersection: Write<'a, PointIntersection>,
    masses: ReadStorage<'a, Mass>,
}

/// TODO: Implement using quadtree to improve performance
pub fn distance(mut data: DistanceSystemData) {
    for (entity, circle, mass) in (&*data.entities, &data.positions, &data.masses).join() {
        let node_radius: f32 = 48.0 * mass.0;

        let d = (data.cursor_position.0.x - circle.0.x).powi(2)
            + (data.cursor_position.0.y - circle.0.y).powi(2);

        if d < node_radius.powi(2) {
            // This node contains the cursor's position.
            // It is the node being dragged.
            data.intersection.0 = entity.id() as i64;

            // info!(
            //     "Point {0} intersect [{1}]",
            //     data.cursor_position.0,
            //     entity.id()
            // );
        }
    }
}
