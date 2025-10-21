//! Event channels for communicating with the simulator from the outside.

use glam::Vec2;

/// Describes an event received by a [`Simulator`].
#[derive(Clone, PartialEq)]
pub enum SimulatorEvent {
    /// The simulation's repel force has been updated.
    RepelForceUpdated(f32),

    /// The simulation's spring stiffness has been updated.
    SpringStiffnessUpdated(f32),

    /// The simulation's spring neutral length has been updated.
    SpringNeutralLengthUpdated(f32),

    /// The simulation's gravity force has been updated.
    GravityForceUpdated(f32),

    /// The simulation's delta time has been updated.
    DeltaTimeUpdated(f32),

    /// The simulation's velocity damping has been updated.
    DampingUpdated(f32),

    /// The simulation's accuracy has been updated.
    /// This represents the quadtree theta value.
    SimulationAccuracyUpdated(f32),

    /// The simulation's freeze threshold has been updated.
    /// A node with a velocity below this threshold should not be considered
    /// in future force computations.
    FreezeThresholdUpdated(f32),

    /// The dimensions of the simulated world has changed.
    WindowResized { width: u32, height: u32 },

    /// A node is being dragged.
    DragStart(Vec2),

    /// A node is no longer being dragged.
    DragEnd,

    /// The position of the dragged node has changed.
    Dragged(Vec2),
}
