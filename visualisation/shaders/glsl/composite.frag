#version 450

// WBOIT composite fragment shader
// Combines accumulated transparency with the opaque background

layout(location = 0) in vec2 frag_uv;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D accum_texture;   // RGBA16F accumulation
layout(set = 0, binding = 1) uniform sampler2D reveal_texture;  // R8 revealage

void main() {
    vec4 accum = texture(accum_texture, frag_uv);
    float reveal = texture(reveal_texture, frag_uv).r;

    // Avoid division by zero
    const float epsilon = 1e-5;

    // Reconstruct color from weighted accumulation
    // accum.rgb = sum(premult_color * weight), accum.a = sum(alpha * weight)
    vec3 avg_color = accum.rgb / max(accum.a, epsilon);

    // reveal contains product of (1 - alpha) for all fragments
    // If reveal is 1.0, no transparent fragments were rendered
    // If reveal is 0.0, fully opaque transparent coverage

    // Final transparency factor
    float transparency = reveal;  // How much background shows through

    // Output for blending: use ONE_MINUS_SRC_ALPHA, SRC_ALPHA blend
    // This means: final = src * src.a + dst * (1 - src.a)
    // We want: final = transparent_color * (1-reveal) + background * reveal
    // So: src.rgb = avg_color, src.a = (1 - reveal)
    out_color = vec4(avg_color, 1.0 - transparency);
}
