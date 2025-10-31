//! Components which make up a node

use glam::Vec2;
use specs::{Component, NullStorage, VecStorage};

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

/// A fixed node does not compute movement.
#[derive(Component, Default)]
#[storage(NullStorage)]
pub struct Fixed;

/// A dragged node has extra force applied to it
#[derive(Component, Default)]
#[storage(NullStorage)]
pub struct Dragged;
