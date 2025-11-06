use crate::web::prelude::SimulatorEvent;
use specs::shrev::EventChannel;
use std::sync::{Arc, RwLock};

pub struct EventDispatcher {
    pub sim_chan: Arc<RwLock<EventChannel<SimulatorEvent>>>,
}

impl EventDispatcher {
    pub fn new() -> Self {
        let chan1 = EventChannel::<SimulatorEvent>::new();
        let lock = RwLock::new(chan1);
        Self {
            sim_chan: Arc::new(lock),
        }
    }
}
