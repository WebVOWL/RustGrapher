//! Components which make up an edge

use specs::{Component, Entity, VecStorage};

/// Vector of edges from one source to (potentially) multiple targets.
/// Source is implicitly the node from which this component is accessed.
#[derive(Component)]
#[storage(VecStorage)]
pub struct Connects {
    pub targets: Vec<Entity>,
}

// /// How strong the spring force of an edge should be.
// #[derive(Component)]
// #[storage(VecStorage)]
// pub struct SpringStiffness(pub f32);

// impl Default for SpringStiffness {
//     fn default() -> Self {
//         Self(1.0)
//     }
// }

// /// Length of an edge in neutral position.
// ///
// /// If edge is shorter than neutral it pushers apart.
// /// If edge is longer than neutral it pulls together.
// #[derive(Component)]
// #[storage(VecStorage)]
// pub struct SpringNeutralLength(pub f32);

// impl Default for SpringNeutralLength {
//     fn default() -> Self {
//         Self(2.0)
//     }
// }
