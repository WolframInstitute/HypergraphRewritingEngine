#version 450

// Instanced cone vertex shader
// Unit cone mesh (tip at origin, base at -Z, radius 1) transformed by per-instance data

// Per-vertex attributes (unit cone mesh)
layout(location = 0) in vec3 in_position;  // Unit cone vertex position
layout(location = 1) in vec3 in_normal;    // Vertex normal (for future lighting)

// Per-instance attributes
layout(location = 2) in vec3 inst_tip;     // Cone tip position (world space)
layout(location = 3) in vec3 inst_direction; // Cone direction (from base to tip, normalized)
layout(location = 4) in vec2 inst_size;    // x = length, y = radius
layout(location = 5) in vec4 inst_color;   // RGBA color

layout(location = 0) out vec4 frag_color;

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
} pc;

void main() {
    // Build rotation matrix to align unit cone (+Z direction) with inst_direction
    // Unit cone: tip at origin, base at z = -1, so it points +Z (from base toward tip)

    vec3 up = inst_direction;  // Target direction (normalized)
    vec3 from = vec3(0.0, 0.0, 1.0);  // Unit cone direction (pointing +Z, base to tip)

    // Rodrigues rotation formula to rotate 'from' to 'up'
    vec3 axis = cross(from, up);
    float axis_len = length(axis);

    mat3 rotation;
    if (axis_len < 0.0001) {
        // Nearly parallel - either same direction or opposite
        if (dot(from, up) > 0.0) {
            // Same direction - identity rotation
            rotation = mat3(1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0);
        } else {
            // Opposite direction - 180 degree rotation around X axis
            rotation = mat3(1.0, 0.0, 0.0,
                            0.0, -1.0, 0.0,
                            0.0, 0.0, -1.0);
        }
    } else {
        axis = axis / axis_len;  // Normalize
        float c = dot(from, up);
        float s = axis_len;
        float one_minus_c = 1.0 - c;

        // Rodrigues rotation matrix (GLSL mat3 is column-major)
        // Column 0: R[0][0], R[1][0], R[2][0]
        // Column 1: R[0][1], R[1][1], R[2][1]
        // Column 2: R[0][2], R[1][2], R[2][2]
        rotation = mat3(
            c + axis.x * axis.x * one_minus_c,           // R[0][0]
            axis.y * axis.x * one_minus_c + axis.z * s,  // R[1][0]
            axis.z * axis.x * one_minus_c - axis.y * s,  // R[2][0]

            axis.x * axis.y * one_minus_c - axis.z * s,  // R[0][1]
            c + axis.y * axis.y * one_minus_c,           // R[1][1]
            axis.z * axis.y * one_minus_c + axis.x * s,  // R[2][1]

            axis.x * axis.z * one_minus_c + axis.y * s,  // R[0][2]
            axis.y * axis.z * one_minus_c - axis.x * s,  // R[1][2]
            c + axis.z * axis.z * one_minus_c            // R[2][2]
        );
    }

    // Scale unit cone: x,y by radius, z by length
    vec3 scaled_pos = in_position;
    scaled_pos.xy *= inst_size.y;  // radius
    scaled_pos.z *= inst_size.x;   // length

    // Rotate to align with direction
    vec3 rotated_pos = rotation * scaled_pos;

    // Translate to tip position
    vec3 world_pos = rotated_pos + inst_tip;

    gl_Position = pc.view_proj * vec4(world_pos, 1.0);
    frag_color = inst_color;
}
