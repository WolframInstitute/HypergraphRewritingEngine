#pragma once

#include <cmath>
#include <cstdint>

namespace viz::math {

// 2D Vector
struct vec2 {
    float x, y;

    constexpr vec2() : x(0), y(0) {}
    constexpr vec2(float x_, float y_) : x(x_), y(y_) {}
    constexpr explicit vec2(float s) : x(s), y(s) {}

    vec2 operator+(const vec2& o) const { return {x + o.x, y + o.y}; }
    vec2 operator-(const vec2& o) const { return {x - o.x, y - o.y}; }
    vec2 operator*(float s) const { return {x * s, y * s}; }
    vec2 operator/(float s) const { return {x / s, y / s}; }
    vec2& operator+=(const vec2& o) { x += o.x; y += o.y; return *this; }
    vec2& operator-=(const vec2& o) { x -= o.x; y -= o.y; return *this; }
    vec2& operator*=(float s) { x *= s; y *= s; return *this; }

    float dot(const vec2& o) const { return x * o.x + y * o.y; }
    float length() const { return std::sqrt(x * x + y * y); }
    float length_sq() const { return x * x + y * y; }
    vec2 normalized() const { float l = length(); return l > 0 ? *this / l : vec2(0); }
};

// 3D Vector
struct vec3 {
    float x, y, z;

    constexpr vec3() : x(0), y(0), z(0) {}
    constexpr vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    constexpr explicit vec3(float s) : x(s), y(s), z(s) {}
    constexpr vec3(const vec2& v, float z_) : x(v.x), y(v.y), z(z_) {}

    vec3 operator+(const vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    vec3 operator-(const vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    vec3 operator/(float s) const { return {x / s, y / s, z / s}; }
    vec3 operator-() const { return {-x, -y, -z}; }
    vec3& operator+=(const vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    vec3& operator-=(const vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }

    float dot(const vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    vec3 cross(const vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }
    float length() const { return std::sqrt(x * x + y * y + z * z); }
    float length_sq() const { return x * x + y * y + z * z; }
    vec3 normalized() const { float l = length(); return l > 0 ? *this / l : vec3(0); }

    vec2 xy() const { return {x, y}; }
};

inline vec3 operator*(float s, const vec3& v) { return v * s; }

// Free functions for vec3
inline float length(const vec3& v) { return v.length(); }
inline float length_sq(const vec3& v) { return v.length_sq(); }
inline vec3 normalize(const vec3& v) { return v.normalized(); }
inline float dot(const vec3& a, const vec3& b) { return a.dot(b); }
inline vec3 cross(const vec3& a, const vec3& b) { return a.cross(b); }

// 4D Vector
struct vec4 {
    float x, y, z, w;

    constexpr vec4() : x(0), y(0), z(0), w(0) {}
    constexpr vec4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
    constexpr explicit vec4(float s) : x(s), y(s), z(s), w(s) {}
    constexpr vec4(const vec3& v, float w_) : x(v.x), y(v.y), z(v.z), w(w_) {}

    vec4 operator+(const vec4& o) const { return {x + o.x, y + o.y, z + o.z, w + o.w}; }
    vec4 operator-(const vec4& o) const { return {x - o.x, y - o.y, z - o.z, w - o.w}; }
    vec4 operator*(float s) const { return {x * s, y * s, z * s, w * s}; }
    vec4 operator/(float s) const { return {x / s, y / s, z / s, w / s}; }

    float dot(const vec4& o) const { return x * o.x + y * o.y + z * o.z + w * o.w; }

    vec3 xyz() const { return {x, y, z}; }
    vec2 xy() const { return {x, y}; }
};

// 4x4 Matrix (column-major for GPU compatibility)
struct mat4 {
    float m[16];  // Column-major: m[col * 4 + row]

    mat4() {
        for (int i = 0; i < 16; ++i) m[i] = 0;
    }

    static mat4 identity() {
        mat4 r;
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
        return r;
    }

    float& at(int row, int col) { return m[col * 4 + row]; }
    float at(int row, int col) const { return m[col * 4 + row]; }

    vec4 col(int c) const {
        return {m[c * 4], m[c * 4 + 1], m[c * 4 + 2], m[c * 4 + 3]};
    }

    mat4 operator*(const mat4& o) const {
        mat4 r;
        for (int c = 0; c < 4; ++c) {
            for (int row = 0; row < 4; ++row) {
                float sum = 0;
                for (int k = 0; k < 4; ++k) {
                    sum += at(row, k) * o.at(k, c);
                }
                r.at(row, c) = sum;
            }
        }
        return r;
    }

    vec4 operator*(const vec4& v) const {
        return {
            m[0] * v.x + m[4] * v.y + m[8] * v.z + m[12] * v.w,
            m[1] * v.x + m[5] * v.y + m[9] * v.z + m[13] * v.w,
            m[2] * v.x + m[6] * v.y + m[10] * v.z + m[14] * v.w,
            m[3] * v.x + m[7] * v.y + m[11] * v.z + m[15] * v.w
        };
    }

    static mat4 translate(const vec3& t) {
        mat4 r = identity();
        r.m[12] = t.x;
        r.m[13] = t.y;
        r.m[14] = t.z;
        return r;
    }

    static mat4 scale(const vec3& s) {
        mat4 r;
        r.m[0] = s.x;
        r.m[5] = s.y;
        r.m[10] = s.z;
        r.m[15] = 1.0f;
        return r;
    }

    static mat4 rotate_x(float radians) {
        mat4 r = identity();
        float c = std::cos(radians), s = std::sin(radians);
        r.m[5] = c;  r.m[9] = -s;
        r.m[6] = s;  r.m[10] = c;
        return r;
    }

    static mat4 rotate_y(float radians) {
        mat4 r = identity();
        float c = std::cos(radians), s = std::sin(radians);
        r.m[0] = c;  r.m[8] = s;
        r.m[2] = -s; r.m[10] = c;
        return r;
    }

    static mat4 rotate_z(float radians) {
        mat4 r = identity();
        float c = std::cos(radians), s = std::sin(radians);
        r.m[0] = c;  r.m[4] = -s;
        r.m[1] = s;  r.m[5] = c;
        return r;
    }

    static mat4 perspective(float fov_y_radians, float aspect, float z_near, float z_far) {
        mat4 r;
        float tan_half_fov = std::tan(fov_y_radians / 2.0f);
        r.m[0] = 1.0f / (aspect * tan_half_fov);
        r.m[5] = 1.0f / tan_half_fov;
        r.m[10] = -(z_far + z_near) / (z_far - z_near);
        r.m[11] = -1.0f;
        r.m[14] = -(2.0f * z_far * z_near) / (z_far - z_near);
        return r;
    }

    static mat4 look_at(const vec3& eye, const vec3& target, const vec3& up) {
        vec3 f = (target - eye).normalized();
        vec3 r = f.cross(up).normalized();
        vec3 u = r.cross(f);

        mat4 m = identity();
        m.m[0] = r.x;  m.m[4] = r.y;  m.m[8] = r.z;
        m.m[1] = u.x;  m.m[5] = u.y;  m.m[9] = u.z;
        m.m[2] = -f.x; m.m[6] = -f.y; m.m[10] = -f.z;
        m.m[12] = -r.dot(eye);
        m.m[13] = -u.dot(eye);
        m.m[14] = f.dot(eye);
        return m;
    }

    static mat4 ortho(float left, float right, float bottom, float top, float z_near, float z_far) {
        mat4 r;
        r.m[0] = 2.0f / (right - left);
        r.m[5] = 2.0f / (top - bottom);
        r.m[10] = -2.0f / (z_far - z_near);
        r.m[12] = -(right + left) / (right - left);
        r.m[13] = -(top + bottom) / (top - bottom);
        r.m[14] = -(z_far + z_near) / (z_far - z_near);
        r.m[15] = 1.0f;
        return r;
    }
};

// Quaternion for rotations
struct quat {
    float x, y, z, w;

    quat() : x(0), y(0), z(0), w(1) {}
    quat(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}

    static quat from_axis_angle(const vec3& axis, float radians) {
        float half = radians * 0.5f;
        float s = std::sin(half);
        vec3 a = axis.normalized();
        return {a.x * s, a.y * s, a.z * s, std::cos(half)};
    }

    quat operator*(const quat& o) const {
        return {
            w * o.x + x * o.w + y * o.z - z * o.y,
            w * o.y - x * o.z + y * o.w + z * o.x,
            w * o.z + x * o.y - y * o.x + z * o.w,
            w * o.w - x * o.x - y * o.y - z * o.z
        };
    }

    quat conjugate() const { return {-x, -y, -z, w}; }

    float length() const { return std::sqrt(x * x + y * y + z * z + w * w); }

    quat normalized() const {
        float l = length();
        return l > 0 ? quat{x / l, y / l, z / l, w / l} : quat{};
    }

    vec3 rotate(const vec3& v) const {
        quat p{v.x, v.y, v.z, 0};
        quat r = *this * p * conjugate();
        return {r.x, r.y, r.z};
    }

    mat4 to_mat4() const {
        mat4 m = mat4::identity();
        float xx = x * x, yy = y * y, zz = z * z;
        float xy = x * y, xz = x * z, yz = y * z;
        float wx = w * x, wy = w * y, wz = w * z;

        m.m[0] = 1 - 2 * (yy + zz);
        m.m[1] = 2 * (xy + wz);
        m.m[2] = 2 * (xz - wy);

        m.m[4] = 2 * (xy - wz);
        m.m[5] = 1 - 2 * (xx + zz);
        m.m[6] = 2 * (yz + wx);

        m.m[8] = 2 * (xz + wy);
        m.m[9] = 2 * (yz - wx);
        m.m[10] = 1 - 2 * (xx + yy);

        return m;
    }
};

// Axis-Aligned Bounding Box
struct AABB {
    vec3 min, max;

    AABB() : min(INFINITY), max(-INFINITY) {}
    AABB(const vec3& min_, const vec3& max_) : min(min_), max(max_) {}

    void expand(const vec3& p) {
        min.x = std::fmin(min.x, p.x);
        min.y = std::fmin(min.y, p.y);
        min.z = std::fmin(min.z, p.z);
        max.x = std::fmax(max.x, p.x);
        max.y = std::fmax(max.y, p.y);
        max.z = std::fmax(max.z, p.z);
    }

    void expand(const AABB& o) {
        expand(o.min);
        expand(o.max);
    }

    vec3 center() const { return (min + max) * 0.5f; }
    vec3 size() const { return max - min; }
    vec3 extents() const { return size() * 0.5f; }

    bool contains(const vec3& p) const {
        return p.x >= min.x && p.x <= max.x &&
               p.y >= min.y && p.y <= max.y &&
               p.z >= min.z && p.z <= max.z;
    }

    bool valid() const {
        return min.x <= max.x && min.y <= max.y && min.z <= max.z;
    }
};

// Constants
constexpr float PI = 3.14159265358979323846f;
constexpr float TAU = 2.0f * PI;
constexpr float DEG_TO_RAD = PI / 180.0f;
constexpr float RAD_TO_DEG = 180.0f / PI;

inline float radians(float degrees) { return degrees * DEG_TO_RAD; }
inline float degrees(float radians) { return radians * RAD_TO_DEG; }

template<typename T>
T clamp(T val, T min_val, T max_val) {
    return val < min_val ? min_val : (val > max_val ? max_val : val);
}

template<typename T>
T lerp(T a, T b, float t) {
    return a + (b - a) * t;
}

} // namespace viz::math
