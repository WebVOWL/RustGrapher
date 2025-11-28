//! Event channels for communicating with the renderer from the outside.

use super::NodeType;

/// Describes an event received by a render [`State`].
#[derive(Clone, PartialEq)]
pub enum RenderEvent {
    /// Hide a [`NodeType`] during rendering.
    ElementFiltered(NodeType),

    /// Show a [`NodeType`] during rendering.
    ElementShown(NodeType),

    /// Pause graph simulation.
    Paused,

    /// Resume graph simulation
    Resumed,

    /// Zoom the graph.
    /// Negative values zoom out, positive zoom in.
    Zoomed(f64),

    // Zoom to show all nodes
    CenterGraph,
}
