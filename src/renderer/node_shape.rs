#[derive(Copy, Clone, Debug)]
pub enum NodeShape {
    Circle { r: f32 },
    Rectangle { w: f32, h: f32 },
}
