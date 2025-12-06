#![allow(dead_code)]
#![allow(unused)]

mod app;
mod event_dispatcher;
mod quadtree;
mod renderer;
mod simulator;

#[cfg(target_arch = "wasm32")]
pub use app::init_render;

#[cfg(not(target_arch = "wasm32"))]
pub use app::run;

/// Exports all the core types of the library.
pub mod prelude {
    use crate::event_dispatcher::EventDispatcher;
    pub use crate::renderer::events::RenderEvent;
    pub use crate::renderer::node_types::NodeType;
    pub use crate::simulator::ressources::events::SimulatorEvent;
    pub use crate::simulator::ressources::simulator_vars::{
        Damping, DeltaTime, FreezeThreshold, GravityForce, QuadTreeTheta, RepelForce,
        SpringNeutralLength, SpringStiffness,
    };
    use std::sync::LazyLock;

    /// The global event handler for RustGrapher.
    pub static EVENT_DISPATCHER: LazyLock<EventDispatcher> =
        LazyLock::new(|| EventDispatcher::new());
}
