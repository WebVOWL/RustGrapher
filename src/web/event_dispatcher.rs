use crate::web::{prelude::SimulatorEvent, renderer::events::RenderEvent};
use specs::shrev::EventChannel;
use std::sync::{Arc, RwLock};

pub struct EventDispatcher {
    pub sim_chan: Arc<RwLock<EventChannel<SimulatorEvent>>>,
    pub rend_chan: Arc<RwLock<EventChannel<RenderEvent>>>,
}

impl EventDispatcher {
    pub fn new() -> Self {
        let chan1 = EventChannel::<SimulatorEvent>::new();
        let chan2 = EventChannel::<RenderEvent>::new();
        let lock1 = RwLock::new(chan1);
        let lock2 = RwLock::new(chan2);
        Self {
            sim_chan: Arc::new(lock1),
            rend_chan: Arc::new(lock2),
        }
    }
}
