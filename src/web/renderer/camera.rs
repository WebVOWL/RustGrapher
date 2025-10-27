pub mod camera_buffer;
pub mod camera_controller;

use glam::{Mat4, Vec3};

pub struct Camera {
    eye: Vec3<f32>,
    target: Vec3<f32>,
    up: Vec3<f32>,
    aspect: f32,
    /// Fovy must be in radians
    fovy: f32,
    znear: f32,
    zfar: f32,
}

impl Camera {
    pub fn build_view_projection_matrix(&self) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye, self.target, self.up);
        let proj = Mat4::perspective_rh(self.fovy, self.aspect, self.znear, self.zfar);
        proj * view
    }
}
