//! Ressources used by the graph simulator.

use glam::Vec2;

/// How strong nodes should push others away.
pub struct RepelForce(pub f32);

/// How strong the edge force should be.
pub struct SpringStiffness(pub f32);

/// Length of a edge in neutral position.
///
/// If edge is shorter it pushers apart.
/// If edge is longer it pulls together.
pub struct SpringNeutralLength(pub f32);

/// How strong the pull to the center should be.
pub struct GravityForce(pub f32);

/// How much time a simulation step should simulate, measured in seconds.
pub struct DeltaTime(pub f32);

/// Amount of damping that should be applied to the node's movement.
pub struct Damping(pub f32);

/// How accurate the force calculations should be.
pub struct QuadTreeTheta(pub f32);

/// Freeze nodes when their velocity falls below this number.
pub struct FreezeThreshold(pub f32);

/// Simulation world size.
pub struct WorldSize {
    pub width: u32,
    pub height: u32,
}

/// The current location of the mouse cursor.
#[derive(Default)]
pub struct CursorPosition(pub Vec2);

/// The entity ID of the node where the cursor position
/// is within the circumference of said node.
pub struct PointIntersection(pub i64);

impl Default for RepelForce {
    fn default() -> Self {
        Self(10e7)
    }
}

impl Default for SpringStiffness {
    fn default() -> Self {
        Self(400.0)
    }
}

impl Default for SpringNeutralLength {
    fn default() -> Self {
        Self(70.0)
    }
}

impl Default for GravityForce {
    fn default() -> Self {
        Self(10.0)
    }
}

impl Default for DeltaTime {
    fn default() -> Self {
        Self(0.005)
    }
}

impl Default for Damping {
    fn default() -> Self {
        Self(0.8)
    }
}

impl Default for QuadTreeTheta {
    fn default() -> Self {
        Self(1.0)
    }
}

impl Default for FreezeThreshold {
    fn default() -> Self {
        Self(10.0)
    }
}

impl Default for WorldSize {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
        }
    }
}

impl Default for PointIntersection {
    fn default() -> Self {
        Self(-1)
    }
}
