#version 450

// Fullscreen triangle vertex shader for WBOIT composite pass
// Generates a fullscreen triangle without any vertex input

layout(location = 0) out vec2 frag_uv;

void main() {
    // Generate fullscreen triangle vertices
    // Vertex 0: (-1, -1), Vertex 1: (3, -1), Vertex 2: (-1, 3)
    vec2 positions[3] = vec2[](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0)
    );

    vec2 pos = positions[gl_VertexIndex];
    gl_Position = vec4(pos, 0.0, 1.0);

    // UV coordinates (0,0) to (1,1) with Y flipped for Vulkan
    frag_uv = pos * 0.5 + 0.5;
}
