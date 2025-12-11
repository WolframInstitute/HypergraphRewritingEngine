#version 450

// WBOIT fragment shader - Weighted Blended Order-Independent Transparency
// Outputs weighted color to accumulation buffer and alpha to revealage buffer
// Reference: McGuire and Bavoil, "Weighted Blended Order-Independent Transparency"

layout(location = 0) in vec4 frag_color;
layout(location = 1) in float frag_depth;

// Two render targets
layout(location = 0) out vec4 out_accum;    // RGBA16F - weighted color accumulation
layout(location = 1) out float out_reveal;  // R8 - revealage (product of 1-alpha)

void main() {
    vec4 color = frag_color;

    // Discard fully transparent fragments
    if (color.a < 0.001) {
        discard;
    }

    // Clamp alpha to avoid issues
    color.a = clamp(color.a, 0.001, 1.0);

    // Weight function - combines depth and color/alpha
    // Depth weight: closer = higher weight, falloff with distance
    float depth_weight = clamp(0.03 / (1e-5 + pow(abs(frag_depth) / 200.0, 4.0)), 1e-2, 3e3);

    // Alpha/color weight: brighter and more opaque = higher weight
    float alpha_weight = max(max(color.r, color.g), max(color.b, color.a)) * 40.0 + 0.01;
    alpha_weight = min(1.0, alpha_weight);
    alpha_weight *= alpha_weight;

    float weight = alpha_weight * depth_weight;

    // Premultiply color by alpha
    vec3 premult_color = color.rgb * color.a;

    // Output weighted color (will be additively blended)
    out_accum = vec4(premult_color * weight, color.a * weight);

    // Output (1-alpha) for revealage (will be multiplied: product of (1-alpha))
    out_reveal = 1.0 - color.a;
}
