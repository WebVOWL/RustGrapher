//! Ressources used by the graph simulator.

#[derive(Default)]
pub struct Repel(pub f32);

#[derive(Default)]
pub struct Spring(pub f32);

#[derive(Default)]
pub struct Gravity(pub f32);

#[derive(Default)]
pub struct RepelForce(pub f32);

#[derive(Default)]
pub struct SpringStiffness(pub f32);

#[derive(Default)]
pub struct SpringNeutralLength(pub f32);

#[derive(Default)]
pub struct GravityForce(pub f32);

#[derive(Default)]
pub struct DeltaTime(pub f32);

#[derive(Default)]
pub struct Damping(pub f32);

#[derive(Default)]
pub struct QuadTreeTheta(pub f32);

#[derive(Default)]
pub struct FreezeThreshold(pub f32);
