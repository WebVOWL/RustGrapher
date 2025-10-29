use glam::Mat4;

use super::PanCamera;

#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    // We can't use glam with bytemuck directly, so we'll have
    // to convert the Matrix4 into a 4x4 f32 array
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_proj: Mat4::IDENTITY.into(),
        }
    }

    pub fn update_view_proj(&mut self, camera: &PanCamera) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}
