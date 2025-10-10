mod quadtree;
mod renderer;
pub mod simulator;

#[cfg(target_arch = "wasm32")]
pub use renderer::init_render;

pub use renderer::run;
