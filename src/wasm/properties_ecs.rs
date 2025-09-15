//! The ECS components of the simulator. Nodes and edges are bundles of components.

use glam::Vec2;
use specs::{Component, NullStorage, VecStorage, prelude::*};

//// Components which make up a node ////

/// The position of a node.
#[derive(Component, Default)]
#[storage(VecStorage)]
pub struct Position(pub Vec2);

/// The velocity of a node.
#[derive(Component, Default)]
#[storage(VecStorage)]
struct Velocity(pub Vec2);

/// The mass of a node.
#[derive(Component)]
#[storage(VecStorage)]
pub struct Mass(pub f32);

impl Default for Mass {
    fn default() -> Self {
        Self(1.0)
    }
}

/// A fixed node does not compute movement.
#[derive(Component, Default)]
#[storage(NullStorage)]
pub struct Fixed {}

// #[derive(Bundle, Default)]
// pub struct Node {
//     pub position: Position,
//     pub velocity: Velocity,
//     pub mass: Mass,
//     pub fixed: Fixed,
// }

#[derive(Clone, Copy)]
struct BodyRef {
    entity: Entity,
    x: f32,
    y: f32,
    px: f32, // momentum x
    py: f32, // momentum y
    m: f32,
}

struct Node {
    // center of this node (cx, cy) and half-size (hs)
    cx: f32,
    cy: f32,
    hs: f32,
    mass: f32,
    comx: f32,
    comy: f32,
    // children: 0..3 quadrants: (0: SW, 1: SE, 2: NW, 3: NE)
    children: [Option<Box<Node>>; 4],
    // if leaf and contains a single body index
    body_idx: Option<usize>,
}

impl Node {
    fn new(cx: f32, cy: f32, hs: f32) -> Self {
        Node {
            cx,
            cy,
            hs,
            mass: 0.0,
            // Center of mass
            comx: 0.0,
            comy: 0.0,
            children: [None, None, None, None],
            body_idx: None,
        }
    }

    fn quadrant_of(&self, x: f32, y: f32) -> usize {
        let right = x >= self.cx;
        let top = y >= self.cy;
        match (right, top) {
            (false, false) => 0, // SW
            (true, false) => 1,  // SE
            (false, true) => 2,  // NW
            (true, true) => 3,   // NE
        }
    }

    fn child_center(&self, quad: usize) -> (f32, f32) {
        let quarter = self.hs * 0.5;
        match quad {
            0 => (self.cx - quarter, self.cy - quarter),
            1 => (self.cx + quarter, self.cy - quarter),
            2 => (self.cx - quarter, self.cy + quarter),
            _ => (self.cx + quarter, self.cy + quarter),
        }
    }

    fn insert(&mut self, bodies: &[BodyRef], idx: usize, min_hs: f32) {
        let b = &bodies[idx];
        // Update mass and center-of-mass cumulatively
        if self.mass == 0.0 {
            self.comx = b.x;
            self.comy = b.y;
            self.mass = b.m;
        } else {
            let total_m = self.mass + b.m;
            self.comx = (self.comx * self.mass + b.x * b.m) / total_m;
            self.comy = (self.comy * self.mass + b.y * b.m) / total_m;
            self.mass = total_m;
        }

        // If node is empty leaf, occupy it
        if self.body_idx.is_none() && self.children.iter().all(|c| c.is_none()) {
            self.body_idx = Some(idx);
            return;
        }

        // If node is leaf but already has a body, push that existing body down
        if !self.children.iter().any(|c| c.is_some()) && self.body_idx.is_some() {
            let existing_idx = self.body_idx.take().unwrap();
            // Subdivide and re-insert existing body and the new one
            let quad_existing = self.quadrant_of(bodies[existing_idx].x, bodies[existing_idx].y);
            if self.children[quad_existing].is_none() {
                let (ccx, ccy) = self.child_center(quad_existing);
                self.children[quad_existing] = Some(Box::new(Node::new(ccx, ccy, self.hs * 0.5)));
            }
            // push existing down
            {
                let child = self.children[quad_existing].as_mut().unwrap();
                child.insert(bodies, existing_idx, min_hs);
            }
        }

        // Insert into the correct child
        let quad = self.quadrant_of(b.x, b.y);
        if self.children[quad].is_none() {
            let (ccx, ccy) = self.child_center(quad);
            self.children[quad] = Some(Box::new(Node::new(ccx, ccy, self.hs * 0.5)));
        }
        // If the child is too small, just accumulate mass here instead of going deeper
        if self.children[quad].as_ref().unwrap().hs < min_hs {
            // accumulate mass and COM already handled at top
            return;
        }
        self.children[quad]
            .as_mut()
            .unwrap()
            .insert(bodies, idx, min_hs);
    }

    fn distance_to(&self, x: f32, y: f32) -> f32 {
        let dx = self.comx - x;
        let dy = self.comy - y;
        (dx * dx + dy * dy).sqrt()
    }

    fn force_on(&self, b: &BodyRef, theta: f32, g: f32, eps: f32) -> (f32, f32) {
        // If this node has no mass, no force
        if self.mass == 0.0 {
            return (0.0, 0.0);
        }
        // If this node is a leaf and refers to the same body, skip
        if let Some(_idx) = self.body_idx {
            // Single body leaf: if it's the same body, skip
            if self.mass == b.m
                && (self.comx - b.x).abs() < 1e-12
                && (self.comy - b.y).abs() < 1e-12
            {
                return (0.0, 0.0);
            }
            // direct force
            let dx = self.comx - b.x;
            let dy = self.comy - b.y;
            let dist2 = dx * dx + dy * dy + eps * eps;
            let dist = dist2.sqrt();
            let f = g * self.mass * b.m / dist2;
            return (f * dx / dist, f * dy / dist);
        }

        // Otherwise, check opening criterion
        let d = self.distance_to(b.x, b.y);
        if self.hs * 2.0 / d < theta {
            // approximate by this node's COM
            let dx = self.comx - b.x;
            let dy = self.comy - b.y;
            let dist2 = dx * dx + dy * dy + eps * eps;
            let dist = dist2.sqrt();
            let f = g * self.mass * b.m / dist2;
            return (f * dx / dist, f * dy / dist);
        }

        // Else, recurse into children
        let mut fx = 0.0;
        let mut fy = 0.0;
        for c in self.children.iter().filter_map(|c| c.as_ref()) {
            let (cx, cy) = c.force_on(b, theta, g, eps);
            fx += cx;
            fy += cy;
        }
        (fx, fy)
    }
}

//// Components which make up an edge ////

/// An edge connects exactly two nodes.
#[derive(Component, Default)]
#[storage(VecStorage)]
pub struct Connects {
    pub src: usize,
    pub target: usize,
}

impl Connects {
    #[inline(always)]
    pub const fn new(src: usize, target: usize) -> Self {
        Self {
            src: src,
            target: target,
        }
    }
}

/// How strong the spring force of an edge should be.
#[derive(Component)]
#[storage(VecStorage)]
pub struct SpringStiffness(pub f32);

impl Default for SpringStiffness {
    fn default() -> Self {
        Self(1.0)
    }
}

/// Length of an edge in neutral position.
///
/// If edge is shorter than neutral it pushers apart.
/// If edge is longer than neutral it pulls together.
#[derive(Component)]
#[storage(VecStorage)]
pub struct SpringNeutralLength(pub f32);

impl Default for SpringNeutralLength {
    fn default() -> Self {
        Self(2.0)
    }
}

// #[derive(Bundle, Default)]
// pub struct Edge {
//     pub connects: Connects,
//     pub string_stiffness: SpringStiffness,
//     pub spring_neutral_length: SpringNeutralLength,
// }
