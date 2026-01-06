#version 450

// 2D vertex picking shader
// For picking vertices rendered as triangles in 2D mode

// Per-vertex attributes
layout(location = 0) in vec3 in_position;       // Vertex position
layout(location = 1) in uint in_vertex_index;   // Vertex index for picking

// Output vertex index to fragment shader
layout(location = 0) flat out uint frag_vertex_index;

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
} pc;

void main() {
    gl_Position = pc.view_proj * vec4(in_position, 1.0);
    frag_vertex_index = in_vertex_index;
}
