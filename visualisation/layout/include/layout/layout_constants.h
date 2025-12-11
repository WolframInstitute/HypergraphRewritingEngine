// Layout constants shared between CUDA and GLSL compute shaders
// This file uses C preprocessor macros for cross-platform compatibility

#ifndef LAYOUT_CONSTANTS_H
#define LAYOUT_CONSTANTS_H

// Barnes-Hut algorithm parameters
#define BH_THETA 0.5f                    // Multipole acceptance criterion (lower = more accurate, slower)
#define BH_THETA_SQ (BH_THETA * BH_THETA)

// Force parameters
#define SPRING_CONSTANT 1.0f             // Attractive force strength
#define REPULSION_CONSTANT 1.0f          // Repulsive force strength
#define GRAVITY_CONSTANT 0.0f            // Optional center gravity (0 = disabled)

// Integration parameters
#define DAMPING 0.9f                     // Velocity damping per iteration
#define MAX_DISPLACEMENT 1.0f            // Clamp max movement per iteration
#define EPSILON 0.0001f                  // Small value to prevent division by zero

// Tree parameters
#define MAX_TREE_DEPTH 32                // Maximum octree/quadtree depth
#define WARPSIZE 32                      // GPU warp size

// Convergence
#define CONVERGENCE_THRESHOLD 0.001f     // Stop when average displacement below this
#define MAX_ITERATIONS 1000              // Maximum iterations before forced stop

// Kernel launch configuration
#define THREADS_BBOX 512                 // Threads for bounding box kernel
#define THREADS_TREE 128                 // Threads for tree building
#define THREADS_SUMMARY 64               // Threads for summarization
#define THREADS_SORT 256                 // Threads for sorting
#define THREADS_FORCE 256                // Threads for force calculation
#define THREADS_INTEGRATE 256            // Threads for integration

// Block factors (for occupancy tuning)
#define FACTOR_BBOX 3
#define FACTOR_TREE 6
#define FACTOR_SUMMARY 6
#define FACTOR_SORT 5
#define FACTOR_FORCE 5
#define FACTOR_INTEGRATE 3

// Tree node markers
#define CELL_EMPTY (-1)                  // Empty cell (no body, no children)
#define CELL_LOCKED (-2)                 // Cell is being modified (atomic lock)

// Dimensionality (compile-time switch)
#ifndef LAYOUT_2D
#define LAYOUT_3D
#endif

#ifdef LAYOUT_3D
#define TREE_CHILDREN 8                  // Octree: 8 children per node
#else
#define TREE_CHILDREN 4                  // Quadtree: 4 children per node
#endif

#endif // LAYOUT_CONSTANTS_H
