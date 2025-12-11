// Centralized color palette for hypergraph visualization
// Edit these colors to customize the appearance of the visualization
//
// All colors are RGBA in range [0.0, 1.0]
// Colors can be changed at runtime via HypergraphRenderConfig

#pragma once

#include <math/types.hpp>

namespace viz::scene::colors {

// =============================================================================
// BACKGROUND
// =============================================================================

// Main window background color (very dark gray #1A1A1A)
constexpr math::vec4 BACKGROUND = {0.1f, 0.1f, 0.1f, 1.0f};

// =============================================================================
// DEBUG AXES (XYZ coordinate display)
// =============================================================================

constexpr math::vec4 DEBUG_AXIS_X = {1.0f, 0.0f, 0.0f, 1.0f};  // Red - X axis
constexpr math::vec4 DEBUG_AXIS_Y = {0.0f, 1.0f, 0.0f, 1.0f};  // Green - Y axis
constexpr math::vec4 DEBUG_AXIS_Z = {0.0f, 0.0f, 1.0f, 1.0f};  // Blue - Z axis

// =============================================================================
// INTERNAL HYPERGRAPH (vertices and edges inside state cubes)
// =============================================================================

// Vertex spheres inside hypergraph states
constexpr math::vec4 VERTEX_SPHERE = {1.0f, 1.0f, 1.0f, 1.0f};  // Cyan

// Hyperedge lines connecting vertices
constexpr math::vec4 HYPEREDGE_LINE = {1.0f, 1.0f, 1.0f, 1.0f};  // Yellow

// Arrowheads on directed hyperedges
constexpr math::vec4 HYPEREDGE_ARROW = {1.0f, 1.0f, 1.0f, 1.0f};  // Orange-yellow

// Translucent bubble around hyperedges (for multi-vertex edges)
constexpr math::vec4 HYPEREDGE_BUBBLE = {1.0f, 1.0f, 1.0f, 0.025f};  // Magenta (translucent)

// Self-loop edges (edges connecting a vertex to itself)
constexpr math::vec4 SELF_LOOP = {1.0f, 1.0f, 1.0f, 1.0f};  // Orange

// =============================================================================
// STATE CUBES (containers for hypergraph states)
// =============================================================================

// Normal state cube wireframe (transparent white, ~30% opacity for lines)
constexpr math::vec4 STATE_CUBE_NORMAL = {1.0f, 1.0f, 1.0f, 0.3f};

// Initial state cube wireframe (same as normal - transparent white)
constexpr math::vec4 STATE_CUBE_INITIAL = {1.0f, 1.0f, 1.0f, 0.3f};

// Canonical state cube wireframe (same as normal - transparent white)
constexpr math::vec4 STATE_CUBE_CANONICAL = {1.0f, 1.0f, 1.0f, 0.3f};

// State cube face alpha (translucent white fill, lower opacity ~15%)
constexpr float STATE_CUBE_FACE_ALPHA = 0.05f;

// =============================================================================
// EVOLUTION EDGES (connections between states)
// =============================================================================

// Event edges (rewrite rule applications, parent->child) - white with 0.75 opacity
constexpr math::vec4 EVENT_EDGE = {1.0f, 1.0f, 1.0f, 0.75f};

// Causal edges (producer->consumer dependency) - orange/amber
constexpr math::vec4 CAUSAL_EDGE = {1.0f, 0.6f, 0.2f, 0.75f};

// Branchial edges (co-occurring events at same generation) - purple
constexpr math::vec4 BRANCHIAL_EDGE = {0.7f, 0.3f, 0.9f, 0.75f};

// =============================================================================
// HELPER: Get default HypergraphRenderConfig with palette colors
// =============================================================================

// This is used to initialize HypergraphRenderConfig with the palette colors
// You can still override colors on a per-config basis if needed

} // namespace viz::scene::colors
