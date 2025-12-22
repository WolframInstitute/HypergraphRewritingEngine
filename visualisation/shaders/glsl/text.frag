#version 450

layout(location = 0) in vec2 frag_uv;
layout(location = 1) in vec4 frag_color;

layout(set = 0, binding = 0) uniform sampler2D font_atlas;

layout(location = 0) out vec4 out_color;

void main() {
    // Sample font atlas - alpha is stored in red channel (R8 texture)
    float alpha = texture(font_atlas, frag_uv).r;

    // Multiply color by alpha for proper blending
    out_color = vec4(frag_color.rgb, frag_color.a * alpha);
}
