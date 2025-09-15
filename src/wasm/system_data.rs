use specs::prelude::*;
// `shred` needs to be in scope for the `SystemData` derive.
use specs::shred;


/// The node data used in force calculations
#[derive(SystemData)]
pub struct NodeForce<'a> {
    positions: ReadStorage<'a, Position>,
    velocities: ReadStorage<'a, Velocity>,
    forces: ReadStorage<'a, Force>,

    delta: Read<'a, DeltaTime>,
    game_state: Write<'a, GameState>,
}