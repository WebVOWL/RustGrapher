use crate::web::simulator::{
    components::nodes::{NodeState, Position, Velocity},
    ressources::simulator_vars::{
        CursorPosition, Damping, DeltaTime, FreezeThreshold, PointIntersection, WorldSize,
    },
};
use glam::Vec2;
use log::info;
use rayon::prelude::*;
use specs::prelude::*;
use specs::{Entities, Join, ParJoin, Read, ReadStorage, WriteStorage, shred};

pub struct UpdateNodePosition;

impl<'a> System<'a> for UpdateNodePosition {
    type SystemData = (
        Entities<'a>,
        WriteStorage<'a, Position>,
        WriteStorage<'a, Velocity>,
        WriteStorage<'a, NodeState>,
        Read<'a, DeltaTime>,
        Read<'a, Damping>,
        Read<'a, FreezeThreshold>,
    );

    fn run(
        &mut self,
        (
            entities,
            mut positions,
            mut velocities,
            mut node_states,
            delta_time,
            damping,
            freeze_threshold,
        ): Self::SystemData,
    ) {
        (&entities, &mut positions, &mut velocities, &mut node_states)
            .par_join()
            .for_each(|(entity, pos, velocity, state)| {
                // Check Freeze Threshold
                if !state.dragged {
                    // Automatically freeze/unfreeze based on velocity
                    if velocity.0.abs().length() < freeze_threshold.0 {
                        state.fixed = true;
                    } else {
                        state.fixed = false;
                    }
                }

                if !state.is_static() {
                    velocity.0 *= damping.0;
                    pos.0 += velocity.0 * delta_time.0;

                    // Safety bounds
                    if pos.0.distance(Vec2::new(0.0, 0.0)) > 10_000_000.0 {
                        pos.0 = Vec2::new(0.0, 0.0);
                    }
                }
            });
    }
}

// /// A node is being dragged.
// #[derive(Default)]
// pub struct DragStart;

// impl<'a> System<'a> for DragStart {
//     type SystemData = (
//         Entities<'a>,
//         ReadStorage<'a, Fixed>,
//         Read<'a, PointIntersection>,
//         Read<'a, LazyUpdate>,
//     );

//     fn run(&mut self, (entities, fixed, dragged_entity, updater): Self::SystemData) {
//         info!("[{0}] Drag start", dragged_entity.0.id());

//         // Enable simulation when node is dragged
//         (&entities, &fixed).par_join().for_each(|(entity, _)| {
//             updater.remove::<Fixed>(entity);
//         });

//         // Except for the dragged node
//         updater.insert(dragged_entity.0, Dragged);
//     }
// }

// /// A node is no longer being dragged.
// #[derive(Default)]
// pub struct DragEnd;

// impl<'a> System<'a> for DragEnd {
//     type SystemData = (Entities<'a>, ReadStorage<'a, Dragged>, Read<'a, LazyUpdate>);

//     fn run(&mut self, (entities, dragged, updater): Self::SystemData) {
//         for (entity, _) in (&entities, &dragged).join() {
//             info!("[{0}] Drag end", entity.id());
//             updater.remove::<Dragged>(entity);
//         }
//     }
// }

// /// The position of the dragged node has changed.
// #[derive(Default)]
// pub struct Dragging;

// impl<'a> System<'a> for Dragging {
//     type SystemData = (
//         ReadStorage<'a, Dragged>,
//         WriteStorage<'a, Position>,
//         Read<'a, CursorPosition>,
//         Read<'a, WorldSize>,
//     );

//     fn run(&mut self, (dragged, mut position, cursor_position, world_size): Self::SystemData) {
//         for (pos, _) in (&position, &dragged).join() {
//             // Normalize position to wgpu's coordinate system
//             let norm_width = cursor_position.0.x.clamp(0.0, world_size.width as f32);
//             let norm_height = (-cursor_position.0.y + world_size.height as f32)
//                 .clamp(0.0, world_size.height as f32);
//             pos.0 = Vec2::new(norm_width, norm_height);

//             // info!("[{0}] Dragged position: {1}", entity_id, norm_pos);
//         }
//     }
// }

#[derive(SystemData)]
pub struct DragStartSystemData<'a> {
    entities: Entities<'a>,
    node_states: WriteStorage<'a, NodeState>,
    dragged_id: Read<'a, PointIntersection>,
}

/// A node is being dragged.
pub fn sys_drag_start(mut data: DragStartSystemData) {
    if data.dragged_id.0 >= 0 {
        info!("[{0}] Drag start", data.dragged_id.0);

        // Unfix everything
        (&mut data.node_states).par_join().for_each(|state| {
            state.fixed = false;
        });

        // Set dragged state on specific node
        if let Some(state) = data
            .node_states
            .get_mut(data.entities.entity(data.dragged_id.0.try_into().unwrap()))
        {
            state.dragged = true;
        }
    }
}

#[derive(SystemData)]
pub struct DragEndSystemData<'a> {
    entities: Entities<'a>,
    node_states: WriteStorage<'a, NodeState>,
}

/// A node is no longer being dragged.
pub fn sys_drag_end(mut data: DragEndSystemData) {
    for (_, state) in (&data.entities, &mut data.node_states).join() {
        if state.dragged {
            state.dragged = false;
        }
    }
}

#[derive(specs::SystemData)]
pub struct DraggingSystemData<'a> {
    entities: Entities<'a>,
    node_states: ReadStorage<'a, NodeState>,
    positions: WriteStorage<'a, Position>,
    cursor_position: Read<'a, CursorPosition>,
}

/// The position of the dragged node has changed.
pub fn sys_dragging(mut data: DraggingSystemData) {
    for (_, pos, state) in (&data.entities, &mut data.positions, &data.node_states).join() {
        if state.dragged {
            pos.0 = data.cursor_position.0;
        }
    }
}
