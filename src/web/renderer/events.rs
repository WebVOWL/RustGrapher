//! Event channels for communicating with the renderer from the outside.

use super::ElementType;

/// Describes an event received by a render [`State`].
#[derive(Clone, PartialEq)]
pub enum RenderEvent {
    /// Hide an [`ElementType`] during rendering.
    ElementFiltered(ElementType),

    /// Show an [`ElementType`] during rendering.
    ElementShown(ElementType),

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
