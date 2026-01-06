#version 450

// Picking fragment shader
// Outputs vertex index to R32_UINT render target

layout(location = 0) flat in uint frag_vertex_index;

// Output to R32_UINT texture
layout(location = 0) out uint out_vertex_index;

void main() {
    // Add 1 so that 0 means "no vertex" (background/miss)
    // This allows distinguishing between vertex 0 and empty space
    out_vertex_index = frag_vertex_index + 1u;
}
