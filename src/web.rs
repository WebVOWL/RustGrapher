mod app;
mod event_dispatcher;
mod quadtree;
mod renderer;
mod simulator;

#[cfg(target_arch = "wasm32")]
pub use app::init_render;

/// Exports all the core types of the library.
pub mod prelude {
    use crate::web::event_dispatcher::EventDispatcher;
    pub use crate::web::renderer::elements::{
        element_type::ElementType, generic::*, owl::*, rdf::*, rdfs::*,
    };
    pub use crate::web::renderer::events::RenderEvent;
    pub use crate::web::simulator::ressources::events::SimulatorEvent;
    pub use crate::web::simulator::ressources::simulator_vars::{
        Damping, DeltaTime, FreezeThreshold, GravityForce, QuadTreeTheta, RepelForce,
        SpringNeutralLength, SpringStiffness, WorldSize,
    };
    use std::sync::LazyLock;

    /// The global event handler for RustGrapher.
    pub static EVENT_DISPATCHER: LazyLock<EventDispatcher> =
        LazyLock::new(|| EventDispatcher::new());
}
