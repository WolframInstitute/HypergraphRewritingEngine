#pragma once

#include <math/types.hpp>

namespace viz::camera {

using namespace math;

// Camera mode
enum class CameraMode {
    Orbit,      // Rotate around a target point
    FreeFlying, // WASD + mouse look (FPS style)
    AutoFollow  // Automatically frame the scene
};

// Base camera interface (VR-ready)
class Camera {
public:
    virtual ~Camera() = default;

    // Get view matrix (eye index for stereo/VR, 0 for mono)
    virtual mat4 get_view_matrix(uint32_t eye = 0) const = 0;

    // Get projection matrix
    virtual mat4 get_projection_matrix(uint32_t eye = 0) const = 0;

    // Combined view-projection
    mat4 get_view_projection_matrix(uint32_t eye = 0) const {
        return get_projection_matrix(eye) * get_view_matrix(eye);
    }

    // Number of views (1 for mono, 2 for stereo VR)
    virtual uint32_t get_view_count() const { return 1; }

    // Camera position in world space
    virtual vec3 get_position() const = 0;

    // Camera forward direction
    virtual vec3 get_forward() const = 0;
};

// Perspective camera with orbit/free-flying controls
class PerspectiveCamera : public Camera {
public:
    PerspectiveCamera();

    // Camera interface
    mat4 get_view_matrix(uint32_t eye = 0) const override;
    mat4 get_projection_matrix(uint32_t eye = 0) const override;
    vec3 get_position() const override;
    vec3 get_forward() const override;

    // Projection settings
    void set_perspective(float fov_degrees, float aspect_ratio, float near_plane, float far_plane);
    void set_aspect_ratio(float aspect);

    float get_fov() const { return fov_degrees_; }
    float get_aspect() const { return aspect_ratio_; }
    float get_near() const { return near_plane_; }
    float get_far() const { return far_plane_; }

    // Mode
    void set_mode(CameraMode mode) { mode_ = mode; }
    CameraMode get_mode() const { return mode_; }

    // Orbit mode controls
    void set_target(const vec3& target);
    void set_distance(float distance);
    void orbit(float delta_yaw, float delta_pitch);
    void pan(float dx, float dy);  // Move target in screen-space (for orbit origin movement)
    void zoom(float delta);  // Adjust distance

    vec3 get_target() const { return target_; }
    float get_distance() const { return distance_; }
    float get_yaw() const { return yaw_; }
    float get_pitch() const { return pitch_; }

    // Free-flying mode
    void set_position_direct(const vec3& pos);
    void set_rotation(float yaw, float pitch);
    void move(const vec3& delta);  // Local space movement
    void look(float delta_yaw, float delta_pitch);

    // Auto-follow mode
    void frame_bounds(const AABB& bounds, float padding = 1.2f);

    // Clamp settings
    void set_pitch_limits(float min_degrees, float max_degrees);
    void set_distance_limits(float min_dist, float max_dist);

private:
    void update_orbit_position();
    void update_view_matrix();

    // Projection
    float fov_degrees_ = 60.0f;
    float aspect_ratio_ = 16.0f / 9.0f;
    float near_plane_ = 0.1f;
    float far_plane_ = 1000.0f;

    // Mode
    CameraMode mode_ = CameraMode::Orbit;

    // Orbit mode state
    vec3 target_ = {0, 0, 0};
    float distance_ = 10.0f;
    float yaw_ = 0.0f;      // Radians around Y axis
    float pitch_ = 0.3f;    // Radians around X axis (elevation)

    // Free-flying mode state
    vec3 position_ = {0, 0, 10};
    float free_yaw_ = 0.0f;
    float free_pitch_ = 0.0f;

    // Limits - prevent going over poles to avoid gimbal lock
    float min_pitch_ = -PI * 0.49f;  // Just under -90 degrees (looking up from below)
    float max_pitch_ = PI * 0.49f;   // Just under +90 degrees (looking down from above)
    float min_distance_ = 0.1f;
    float max_distance_ = 10000.0f;

    // Cached matrices
    mutable mat4 view_matrix_;
    mutable mat4 projection_matrix_;
    mutable bool view_dirty_ = true;
    mutable bool projection_dirty_ = true;
};

// Camera controller for input handling
class CameraController {
public:
    CameraController(PerspectiveCamera* camera);

    // Input handling
    void on_mouse_move(float dx, float dy);
    void on_mouse_scroll(float delta);
    void on_key(int key, bool pressed);

    // Update (call each frame with delta time)
    void update(float dt);

    // Settings
    void set_orbit_speed(float speed) { orbit_speed_ = speed; }
    void set_pan_speed(float speed) { pan_speed_ = speed; }
    void set_zoom_speed(float speed) { zoom_speed_ = speed; }
    void set_move_speed(float speed) { move_speed_ = speed; }
    void set_look_speed(float speed) { look_speed_ = speed; }

    // State
    void set_mouse_captured(bool captured) { mouse_captured_ = captured; }
    bool is_mouse_captured() const { return mouse_captured_; }

    // Modifier state (call from app when modifier keys change)
    void set_shift_held(bool held) { shift_held_ = held; }
    bool is_shift_held() const { return shift_held_; }

private:
    PerspectiveCamera* camera_;

    // Speeds
    float orbit_speed_ = 0.005f;
    float pan_speed_ = 1.0f;
    float zoom_speed_ = 0.1f;
    float move_speed_ = 5.0f;
    float look_speed_ = 0.003f;

    // Input state
    bool mouse_captured_ = false;
    bool shift_held_ = false;
    bool key_forward_ = false;
    bool key_back_ = false;
    bool key_left_ = false;
    bool key_right_ = false;
    bool key_up_ = false;
    bool key_down_ = false;
};

} // namespace viz::camera
