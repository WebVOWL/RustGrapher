//! Event channels for communicating with the renderer from the outside.

use super::PrefixType;

/// Describes an event received by a render [`State`].
#[derive(Clone, PartialEq)]
pub enum RenderEvent {
    /// Hide a [`NodeType`] during rendering.
    ElementFiltered(PrefixType),

    /// Show a [`NodeType`] during rendering.
    ElementShown(PrefixType),

    /// Pause graph simulation.
    Paused,

    /// Resume graph simulation
    Resumed,

    /// Zoom the graph.
    /// Negative values soom out, positive zoom in.
    Zoomed(f64),
}
