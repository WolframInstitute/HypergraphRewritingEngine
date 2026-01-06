#version 450

// Instanced sphere picking vertex shader
// Outputs vertex index instead of color for GPU picking

// Per-vertex attributes (unit sphere mesh)
layout(location = 0) in vec3 in_position;  // Unit sphere vertex position
layout(location = 1) in vec3 in_normal;    // Vertex normal (unused, but keeps layout compatible)

// Per-instance attributes
layout(location = 2) in vec3 inst_center;       // Sphere center position (world space)
layout(location = 3) in float inst_radius;      // Sphere radius
layout(location = 4) in uint inst_vertex_index; // Vertex index for picking (0-based)

// Output vertex index to fragment shader (flat = no interpolation)
layout(location = 0) flat out uint frag_vertex_index;

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
} pc;

void main() {
    // Scale unit sphere by radius and translate to center
    vec3 world_pos = in_position * inst_radius + inst_center;

    gl_Position = pc.view_proj * vec4(world_pos, 1.0);
    frag_vertex_index = inst_vertex_index;
}
