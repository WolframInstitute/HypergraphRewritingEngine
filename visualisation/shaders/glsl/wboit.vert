#version 450

// WBOIT vertex shader - outputs position and linear depth for weighting

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec4 in_color;

layout(location = 0) out vec4 frag_color;
layout(location = 1) out float frag_depth;  // Linear depth for weighting

layout(push_constant) uniform PushConstants {
    mat4 view_proj;
} pc;

void main() {
    vec4 clip_pos = pc.view_proj * vec4(in_position, 1.0);
    gl_Position = clip_pos;
    frag_color = in_color;
    // Linear depth (0 at near, 1 at far approximately)
    frag_depth = clip_pos.z / clip_pos.w;
}
