//! Components storing computed force values
//! These components are updated every time step (tick)

use glam::Vec2;
use specs::{Component, VecStorage};

#[derive(Component, Default)]
#[storage(VecStorage)]
pub struct NodeForces(pub Vec2);
