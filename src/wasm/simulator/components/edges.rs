//! Components which make up an edge

use specs::{Component, Entity, VecStorage};

/// An edge connects exactly two nodes.
#[derive(Component)]
#[storage(VecStorage)]
pub struct Connects {
    pub src: Entity,
    pub target: Entity,
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
