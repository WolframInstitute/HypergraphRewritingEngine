#include <camera/camera.hpp>
#include <algorithm>

namespace viz::camera {

PerspectiveCamera::PerspectiveCamera() {
    update_orbit_position();
}

mat4 PerspectiveCamera::get_view_matrix(uint32_t eye) const {
    if (view_dirty_) {
        if (mode_ == CameraMode::Orbit || mode_ == CameraMode::AutoFollow) {
            view_matrix_ = mat4::look_at(position_, target_, vec3(0, 1, 0));
        } else {
            // Free-flying: compute view from position and angles
            vec3 forward = get_forward();
            vec3 target = position_ + forward;
            view_matrix_ = mat4::look_at(position_, target, vec3(0, 1, 0));
        }
        view_dirty_ = false;
    }
    return view_matrix_;
}

mat4 PerspectiveCamera::get_projection_matrix(uint32_t eye) const {
    if (projection_dirty_) {
        projection_matrix_ = mat4::perspective(
            radians(fov_degrees_), aspect_ratio_, near_plane_, far_plane_);
        projection_dirty_ = false;
    }
    return projection_matrix_;
}

vec3 PerspectiveCamera::get_position() const {
    return position_;
}

vec3 PerspectiveCamera::get_forward() const {
    if (mode_ == CameraMode::Orbit || mode_ == CameraMode::AutoFollow) {
        return (target_ - position_).normalized();
    } else {
        // Free-flying: compute from yaw/pitch
        float cy = std::cos(free_yaw_);
        float sy = std::sin(free_yaw_);
        float cp = std::cos(free_pitch_);
        float sp = std::sin(free_pitch_);
        return vec3(sy * cp, -sp, -cy * cp);
    }
}

void PerspectiveCamera::set_perspective(float fov_degrees, float aspect_ratio,
                                         float near_plane, float far_plane) {
    fov_degrees_ = fov_degrees;
    aspect_ratio_ = aspect_ratio;
    near_plane_ = near_plane;
    far_plane_ = far_plane;
    projection_dirty_ = true;
}

void PerspectiveCamera::set_aspect_ratio(float aspect) {
    aspect_ratio_ = aspect;
    projection_dirty_ = true;
}

void PerspectiveCamera::set_target(const vec3& target) {
    target_ = target;
    if (mode_ == CameraMode::Orbit) {
        update_orbit_position();
    }
    view_dirty_ = true;
}

void PerspectiveCamera::set_distance(float distance) {
    distance_ = clamp(distance, min_distance_, max_distance_);
    if (mode_ == CameraMode::Orbit) {
        update_orbit_position();
    }
    view_dirty_ = true;
}

void PerspectiveCamera::orbit(float delta_yaw, float delta_pitch) {
    yaw_ += delta_yaw;
    pitch_ = clamp(pitch_ + delta_pitch, min_pitch_, max_pitch_);
    update_orbit_position();
    view_dirty_ = true;
}

void PerspectiveCamera::pan(float dx, float dy) {
    // Move target in screen-space (right/up relative to camera view)
    // Scale by distance so panning feels consistent at different zoom levels
    float scale = distance_ * 0.001f;

    // Compute right and up vectors from yaw/pitch
    float cy = std::cos(yaw_);
    float sy = std::sin(yaw_);
    float cp = std::cos(pitch_);
    float sp = std::sin(pitch_);

    // Right vector (perpendicular to view direction in XZ plane)
    vec3 right(cy, 0, -sy);

    // Up vector (perpendicular to both forward and right)
    // For orbit camera, we want up to be perpendicular to view but still "upward"
    vec3 forward(-sy * cp, -sp, -cy * cp);
    vec3 up = right.cross(forward).normalized();

    target_ += right * (-dx * scale) + up * (dy * scale);
    update_orbit_position();
    view_dirty_ = true;
}

void PerspectiveCamera::zoom(float delta) {
    distance_ = clamp(distance_ * (1.0f - delta), min_distance_, max_distance_);
    update_orbit_position();
    view_dirty_ = true;
}

void PerspectiveCamera::set_position_direct(const vec3& pos) {
    position_ = pos;
    view_dirty_ = true;
}

void PerspectiveCamera::set_rotation(float yaw, float pitch) {
    free_yaw_ = yaw;
    free_pitch_ = clamp(pitch, min_pitch_, max_pitch_);
    view_dirty_ = true;
}

void PerspectiveCamera::move(const vec3& delta) {
    if (mode_ == CameraMode::FreeFlying) {
        // Convert local delta to world space
        float cy = std::cos(free_yaw_);
        float sy = std::sin(free_yaw_);

        vec3 right(cy, 0, sy);
        vec3 forward(sy, 0, -cy);
        vec3 up(0, 1, 0);

        position_ += right * delta.x + up * delta.y + forward * delta.z;
        view_dirty_ = true;
    }
}

void PerspectiveCamera::look(float delta_yaw, float delta_pitch) {
    free_yaw_ += delta_yaw;
    free_pitch_ = clamp(free_pitch_ + delta_pitch, min_pitch_, max_pitch_);
    view_dirty_ = true;
}

void PerspectiveCamera::frame_bounds(const AABB& bounds, float padding) {
    if (!bounds.valid()) return;

    target_ = bounds.center();

    // Calculate distance to fit bounds in view
    vec3 size = bounds.size();
    float max_extent = std::max({size.x, size.y, size.z}) * padding;

    float half_fov = radians(fov_degrees_) * 0.5f;
    distance_ = max_extent / (2.0f * std::tan(half_fov));
    distance_ = clamp(distance_, min_distance_, max_distance_);

    update_orbit_position();
    view_dirty_ = true;
}

void PerspectiveCamera::set_pitch_limits(float min_degrees, float max_degrees) {
    min_pitch_ = radians(min_degrees);
    max_pitch_ = radians(max_degrees);
}

void PerspectiveCamera::set_distance_limits(float min_dist, float max_dist) {
    min_distance_ = min_dist;
    max_distance_ = max_dist;
}

void PerspectiveCamera::update_orbit_position() {
    // Spherical to Cartesian
    float cp = std::cos(pitch_);
    float sp = std::sin(pitch_);
    float cy = std::cos(yaw_);
    float sy = std::sin(yaw_);

    vec3 offset(
        distance_ * cp * sy,
        distance_ * sp,
        distance_ * cp * cy
    );

    position_ = target_ + offset;
}

void PerspectiveCamera::update_view_matrix() {
    view_dirty_ = true;
}

// CameraController implementation

CameraController::CameraController(PerspectiveCamera* camera)
    : camera_(camera) {}

void CameraController::on_mouse_move(float dx, float dy) {
    if (!mouse_captured_) return;

    if (camera_->get_mode() == CameraMode::Orbit) {
        if (shift_held_) {
            // Shift + drag = pan (move orbit target)
            camera_->pan(dx * pan_speed_, dy * pan_speed_);
        } else {
            // Normal drag = orbit
            camera_->orbit(-dx * orbit_speed_, -dy * orbit_speed_);
        }
    } else if (camera_->get_mode() == CameraMode::FreeFlying) {
        camera_->look(-dx * look_speed_, -dy * look_speed_);
    }
}

void CameraController::on_mouse_scroll(float delta) {
    if (camera_->get_mode() == CameraMode::Orbit) {
        camera_->zoom(delta * zoom_speed_);
    }
}

void CameraController::on_key(int key, bool pressed) {
    // Common key bindings: WASD + Space/Shift for movement
    switch (key) {
        case 'W': case 'w': key_forward_ = pressed; break;
        case 'S': case 's': key_back_ = pressed; break;
        case 'A': case 'a': key_left_ = pressed; break;
        case 'D': case 'd': key_right_ = pressed; break;
        case ' ': key_up_ = pressed; break;  // Space
        case 340: key_down_ = pressed; break;  // Shift (GLFW code, adjust as needed)
    }
}

void CameraController::update(float dt) {
    if (camera_->get_mode() != CameraMode::FreeFlying) return;

    vec3 move(0);
    if (key_right_) move.x += 1;
    if (key_left_) move.x -= 1;
    if (key_up_) move.y += 1;
    if (key_down_) move.y -= 1;
    if (key_forward_) move.z += 1;
    if (key_back_) move.z -= 1;

    if (move.length_sq() > 0) {
        camera_->move(move.normalized() * move_speed_ * dt);
    }
}

} // namespace viz::camera
