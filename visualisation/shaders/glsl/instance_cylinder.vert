#version 450

// Instanced cylinder vertex shader
// Unit cylinder mesh (bottom at Y=0, top at Y=1, radius 1) transformed by per-instance data

// Per-vertex attributes (unit cylinder mesh)
layout(location = 0) in vec3 in_position;  // Unit cylinder vertex position
layout(location = 1) in vec3 in_normal;    // Vertex normal

// Per-instance attributes
layout(location = 2) in vec3 inst_start;       // Cylinder start position (world space)
layout(location = 3) in vec3 inst_end;         // Cylinder end position (world space)
layout(location = 4) in float inst_radius;     // Cylinder radius
layout(location = 5) in vec4 inst_start_color; // RGBA color at start
layout(location = 6) in vec4 inst_end_color;   // RGBA color at end

layout(location = 0) out vec4 frag_color;
layout(location = 1) out vec3 frag_normal;

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
} pc;

void main() {
    // Direction from start to end
    vec3 direction = inst_end - inst_start;
    float cyl_length = length(direction);

    if (cyl_length < 0.0001) {
        // Degenerate - zero length
        gl_Position = pc.view_proj * vec4(inst_start, 1.0);
        frag_color = inst_start_color;
        frag_normal = vec3(0, 1, 0);
        return;
    }

    vec3 up = direction / cyl_length;  // Normalized direction
    vec3 from = vec3(0.0, 1.0, 0.0);  // Unit cylinder goes along +Y

    // Build rotation matrix to align +Y with 'up'
    vec3 axis = cross(from, up);
    float axis_len = length(axis);

    mat3 rotation;
    if (axis_len < 0.0001) {
        // Nearly parallel
        if (dot(from, up) > 0.0) {
            // Same direction - identity
            rotation = mat3(1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0);
        } else {
            // Opposite direction - 180 degree rotation around X
            rotation = mat3(1.0, 0.0, 0.0,
                            0.0, -1.0, 0.0,
                            0.0, 0.0, -1.0);
        }
    } else {
        axis = axis / axis_len;  // Normalize
        float c = dot(from, up);
        float s = axis_len;
        float one_minus_c = 1.0 - c;

        rotation = mat3(
            c + axis.x * axis.x * one_minus_c,
            axis.y * axis.x * one_minus_c + axis.z * s,
            axis.z * axis.x * one_minus_c - axis.y * s,

            axis.x * axis.y * one_minus_c - axis.z * s,
            c + axis.y * axis.y * one_minus_c,
            axis.z * axis.y * one_minus_c + axis.x * s,

            axis.x * axis.z * one_minus_c + axis.y * s,
            axis.y * axis.z * one_minus_c - axis.x * s,
            c + axis.z * axis.z * one_minus_c
        );
    }

    // Scale unit cylinder: XZ by radius, Y by length
    vec3 scaled_pos = in_position;
    scaled_pos.xz *= inst_radius;
    scaled_pos.y *= cyl_length;

    // Rotate to align with direction
    vec3 rotated_pos = rotation * scaled_pos;

    // Translate to start position
    vec3 world_pos = rotated_pos + inst_start;

    gl_Position = pc.view_proj * vec4(world_pos, 1.0);

    // Interpolate color based on position along cylinder (Y goes 0 to 1)
    frag_color = mix(inst_start_color, inst_end_color, in_position.y);

    // Transform normal
    frag_normal = rotation * in_normal;
}
