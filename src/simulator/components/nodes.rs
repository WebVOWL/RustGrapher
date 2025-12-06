//! Components which make up a node

use glam::Vec2;
use specs::{Component, VecStorage};

/// The position of a node.
#[derive(Component, Default)]
#[storage(VecStorage)]
pub struct Position(pub Vec2);

/// The velocity of a node.
#[derive(Component, Default)]
#[storage(VecStorage)]
pub struct Velocity(pub Vec2);

/// The mass of a node.
#[derive(Component)]
#[storage(VecStorage)]
pub struct Mass(pub f32);

impl Default for Mass {
    fn default() -> Self {
        Self(1.0)
    }
}

#[derive(Component, Debug, Default, Clone)]
#[storage(VecStorage)]
pub struct NodeState {
    pub fixed: bool,
    pub dragged: bool,
}

impl NodeState {
    pub fn is_static(&self) -> bool {
        self.fixed || self.dragged
    }
}
