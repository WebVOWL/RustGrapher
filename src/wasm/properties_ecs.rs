//! The ECS components of the simulator. Nodes and edges are bundles of components.

use bevy::prelude::*;
use glam::Vec2;

//// Components which make up a node ////

/// The position of a node.
#[derive(Component, Default)]
pub struct Position(pub Vec2);

/// The velocity of a node.
#[derive(Component, Default)]
pub struct Velocity(pub Vec2);

impl Velocity {
    #[inline(always)]
    pub const fn new(velocity: Vec2) -> Self {
        Self(velocity)
    }
}

/// The mass of a node.
#[derive(Component)]
pub struct Mass(pub f32);

impl Default for Mass {
    fn default() -> Self {
        Self(1.0)
    }
}

/// A fixed node does not compute movement.
#[derive(Component, Default)]
pub struct Fixed {}

#[derive(Bundle, Default)]
pub struct Node {
    pub position: Position,
    pub velocity: Velocity,
    pub mass: Mass,
    pub fixed: Fixed,
}

//// Components which make up an edge ////

/// An edge connects exactly two nodes.
#[derive(Component, Default)]
pub struct Connects {
    pub src: usize,
    pub target: usize,
}

impl Connects {
    #[inline(always)]
    pub const fn new(src: usize, target: usize) -> Self {
        Self {
            src: src,
            target: target,
        }
    }
}

/// How strong the spring force of an edge should be.
#[derive(Component)]
pub struct SpringStiffness(pub f32);

impl Default for SpringStiffness {
    fn default() -> Self {
        Self(1.0)
    }
}

/// Length of an edge in neutral position.
///
/// If edge is shorter than neutral it pushers apart.
/// If edge is longer than neutral it pulls together.
#[derive(Component)]
pub struct SpringNeutralLength(pub f32);

impl Default for SpringNeutralLength {
    fn default() -> Self {
        Self(2.0)
    }
}

#[derive(Bundle, Default)]
pub struct Edge {
    pub connects: Connects,
    pub string_stiffness: SpringStiffness,
    pub spring_neutral_length: SpringNeutralLength,
}
