//! Event channels for communicating with the simulator from the outside.

use specs::shrev::EventChannel;

#[derive(Default)]
pub struct RepelForceChan(pub EventChannel<f32>);

#[derive(Default)]
pub struct SpringStiffnessChan(pub EventChannel<f32>);

#[derive(Default)]
pub struct SpringNeutralChan(pub EventChannel<f32>);

#[derive(Default)]
pub struct GravityForceChan(pub EventChannel<f32>);

#[derive(Default)]
pub struct DeltaTimeChan(pub EventChannel<f32>);

#[derive(Default)]
pub struct DampingChan(pub EventChannel<f32>);

#[derive(Default)]
pub struct QuadTreeThetaChan(pub EventChannel<f32>);

#[derive(Default)]
pub struct FreezeThresholdChan(pub EventChannel<f32>);
