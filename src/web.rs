mod app;
mod event_dispatcher;
mod quadtree;
mod renderer;
mod simulator;

pub use app::init_render;

pub mod prelude {
    use crate::web::event_dispatcher::EventDispatcher;
    pub use crate::web::simulator::ressources::events::SimulatorEvent;
    use std::sync::LazyLock;

    pub static EVENT_DISPATCHER: LazyLock<EventDispatcher> =
        LazyLock::new(|| EventDispatcher::new());
}
