//! Ressources used by the graph simulator.

use glam::Vec2;

/// How strong nodes should push others away.
#[derive(Default)]
pub struct RepelForce(pub f32);

/// How strong the edge force should be.
#[derive(Default)]
pub struct SpringStiffness(pub f32);

/// Length of a edge in neutral position.
///
/// If edge is shorter it pushers apart.
/// If edge is longer it pulls together.
#[derive(Default)]
pub struct SpringNeutralLength(pub f32);

/// How strong the pull to the center should be.
#[derive(Default)]
pub struct GravityForce(pub f32);

/// How much time a simulation step should simulate, measured in seconds.
#[derive(Default)]
pub struct DeltaTime(pub f32);

/// Amount of damping that should be applied to the node's movement.
#[derive(Default)]
pub struct Damping(pub f32);

/// How accurate the force calculations should be.
#[derive(Default)]
pub struct QuadTreeTheta(pub f32);

/// Freeze nodes when their velocity falls below this number.
#[derive(Default)]
pub struct FreezeThreshold(pub f32);

/// Simulation world size.
pub struct WorldSize {
    pub width: u32,
    pub height: u32,
}

impl Default for WorldSize {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
        }
    }
}

/// The current location of the mouse cursor.
#[derive(Default)]
pub struct CursorPosition(pub Vec2);

/// The entity ID of the node where the cursor position
/// is within the circumference of said node.
#[derive(Default)]
pub struct PointIntersection(pub u32);
