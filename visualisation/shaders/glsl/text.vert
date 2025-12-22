#version 450

// Per-vertex attributes (unit quad)
layout(location = 0) in vec2 in_position;  // 0-1 range quad corners
layout(location = 1) in vec2 in_uv;        // 0-1 range UV corners

// Per-instance attributes (glyph data)
layout(location = 2) in vec2 inst_pos;      // Screen position (pixels, top-left origin)
layout(location = 3) in vec4 inst_uv_rect;  // UV rect in atlas (min_u, min_v, max_u, max_v)
layout(location = 4) in vec4 inst_color;    // Text color RGBA

layout(location = 0) out vec2 frag_uv;
layout(location = 1) out vec4 frag_color;

layout(push_constant) uniform PushConstants {
    vec4 screen_size;  // width, height, char_width, char_height
} pc;

void main() {
    vec2 char_size = pc.screen_size.zw;
    vec2 screen = pc.screen_size.xy;

    // Scale quad by character size and translate to screen position
    vec2 pos = inst_pos + in_position * char_size;

    // Convert to NDC: [0, screen] -> [-1, 1]
    // In Vulkan NDC: Y=-1 is at TOP, Y=+1 is at BOTTOM (matches screen coords)
    // So no Y-flip needed for top-left origin with Y increasing downward
    vec2 ndc = pos / screen * 2.0 - 1.0;

    gl_Position = vec4(ndc, 0.0, 1.0);

    // Interpolate UV within the glyph's atlas rect
    frag_uv = mix(inst_uv_rect.xy, inst_uv_rect.zw, in_uv);
    frag_color = inst_color;
}
