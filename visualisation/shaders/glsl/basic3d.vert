#version 450

// Basic 3D vertex shader with MVP transform
// Input: position (vec3), color (vec4)

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec4 in_color;

layout(location = 0) out vec4 frag_color;

layout(push_constant) uniform PushConstants {
    mat4 view_proj;  // Combined view-projection matrix
} pc;

void main() {
    gl_Position = pc.view_proj * vec4(in_position, 1.0);
    frag_color = in_color;
}
