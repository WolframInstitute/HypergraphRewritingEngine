#version 450

// Instanced sphere vertex shader
// Unit sphere mesh (centered at origin, radius 1) transformed by per-instance data

// Per-vertex attributes (unit sphere mesh)
layout(location = 0) in vec3 in_position;  // Unit sphere vertex position
layout(location = 1) in vec3 in_normal;    // Vertex normal (for future lighting)

// Per-instance attributes
layout(location = 2) in vec3 inst_center;  // Sphere center position (world space)
layout(location = 3) in float inst_radius; // Sphere radius
layout(location = 4) in vec4 inst_color;   // RGBA color

layout(location = 0) out vec4 frag_color;
layout(location = 1) out vec3 frag_normal;

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
} pc;

void main() {
    // Scale unit sphere by radius and translate to center
    vec3 world_pos = in_position * inst_radius + inst_center;

    gl_Position = pc.view_proj * vec4(world_pos, 1.0);
    frag_color = inst_color;
    frag_normal = in_normal;  // Normal doesn't change for uniform scale
}
