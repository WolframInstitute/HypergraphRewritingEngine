// CUDA Barnes-Hut kernels for 3D graph layout
// Adapted from GPUGraphLayout (Martin Burtscher, Texas State University)
// Extended to 3D octree

#include "barnes_hut_kernels.cuh"
#include <cfloat>

namespace viz::layout::cuda {

// Device variables
__device__ int bottomd;
__device__ int maxdepthd;
__device__ float radiusd;
__device__ int errd;
__device__ float minxd, minyd, minzd;
__device__ float maxxd, maxyd, maxzd;

//-----------------------------------------------------------------------------
// BoundingBoxKernel: Compute bounding box using parallel reduction
//-----------------------------------------------------------------------------
__global__ __launch_bounds__(THREADS_BBOX, FACTOR_BBOX)
void BoundingBoxKernel(
    int nbodies,
    volatile float* __restrict__ body_pos_x,
    volatile float* __restrict__ body_pos_y,
    volatile float* __restrict__ body_pos_z,
    volatile float* __restrict__ node_pos_x,
    volatile float* __restrict__ node_pos_y,
    volatile float* __restrict__ node_pos_z)
{
    __shared__ volatile float smin_x[THREADS_BBOX];
    __shared__ volatile float smax_x[THREADS_BBOX];
    __shared__ volatile float smin_y[THREADS_BBOX];
    __shared__ volatile float smax_y[THREADS_BBOX];
    __shared__ volatile float smin_z[THREADS_BBOX];
    __shared__ volatile float smax_z[THREADS_BBOX];

    int tid = threadIdx.x;
    int i = tid + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // Initialize with extreme values
    float local_minx = FLT_MAX, local_maxx = -FLT_MAX;
    float local_miny = FLT_MAX, local_maxy = -FLT_MAX;
    float local_minz = FLT_MAX, local_maxz = -FLT_MAX;

    // Each thread processes multiple bodies
    while (i < nbodies) {
        float x = body_pos_x[i];
        float y = body_pos_y[i];
        float z = body_pos_z[i];

        local_minx = fminf(local_minx, x);
        local_maxx = fmaxf(local_maxx, x);
        local_miny = fminf(local_miny, y);
        local_maxy = fmaxf(local_maxy, y);
        local_minz = fminf(local_minz, z);
        local_maxz = fmaxf(local_maxz, z);

        i += stride;
    }

    // Store in shared memory
    smin_x[tid] = local_minx;
    smax_x[tid] = local_maxx;
    smin_y[tid] = local_miny;
    smax_y[tid] = local_maxy;
    smin_z[tid] = local_minz;
    smax_z[tid] = local_maxz;
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smin_x[tid] = fminf(smin_x[tid], smin_x[tid + s]);
            smax_x[tid] = fmaxf(smax_x[tid], smax_x[tid + s]);
            smin_y[tid] = fminf(smin_y[tid], smin_y[tid + s]);
            smax_y[tid] = fmaxf(smax_y[tid], smax_y[tid + s]);
            smin_z[tid] = fminf(smin_z[tid], smin_z[tid + s]);
            smax_z[tid] = fmaxf(smax_z[tid], smax_z[tid + s]);
        }
        __syncthreads();
    }

    // First thread of first block writes result
    if (tid == 0) {
        // Atomic min/max across blocks
        atomicMin((int*)&minxd, __float_as_int(smin_x[0]));
        atomicMax((int*)&maxxd, __float_as_int(smax_x[0]));
        atomicMin((int*)&minyd, __float_as_int(smin_y[0]));
        atomicMax((int*)&maxyd, __float_as_int(smax_y[0]));
        atomicMin((int*)&minzd, __float_as_int(smin_z[0]));
        atomicMax((int*)&maxzd, __float_as_int(smax_z[0]));

        __threadfence();

        // Last block computes final result
        int last_block = atomicAdd(&errd, 1);
        if (last_block == gridDim.x - 1) {
            // Compute root cell
            float diffx = maxxd - minxd;
            float diffy = maxyd - minyd;
            float diffz = maxzd - minzd;
            float radius = fmaxf(diffx, fmaxf(diffy, diffz)) * 0.5f;

            // Add small padding
            radius *= 1.001f;
            radiusd = radius;

            // Root position is center of bounding box
            int root = nbodies;  // Root node index
            node_pos_x[root] = (minxd + maxxd) * 0.5f;
            node_pos_y[root] = (minyd + maxyd) * 0.5f;
            node_pos_z[root] = (minzd + maxzd) * 0.5f;

            // Reset error flag for next phase
            errd = 0;
            __threadfence();
        }
    }
}

//-----------------------------------------------------------------------------
// ClearKernel1: Reset child pointers to CELL_EMPTY
//-----------------------------------------------------------------------------
__global__ __launch_bounds__(THREADS_TREE, FACTOR_TREE)
void ClearKernel1(
    int nnodes,
    int nbodies,
    volatile int* __restrict__ children)
{
    int stride = blockDim.x * gridDim.x;
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Clear all child pointers for internal nodes
    int total_children = nnodes * TREE_CHILDREN;
    while (i < total_children) {
        children[i] = CELL_EMPTY;
        i += stride;
    }
}

//-----------------------------------------------------------------------------
// TreeBuildingKernel: Build octree by inserting bodies
//-----------------------------------------------------------------------------
__global__ __launch_bounds__(THREADS_TREE, FACTOR_TREE)
void TreeBuildingKernel(
    int nnodes,
    int nbodies,
    volatile int* __restrict__ children,
    volatile float* __restrict__ body_pos_x,
    volatile float* __restrict__ body_pos_y,
    volatile float* __restrict__ body_pos_z,
    volatile float* __restrict__ node_pos_x,
    volatile float* __restrict__ node_pos_y,
    volatile float* __restrict__ node_pos_z)
{
    int i, j, depth, localmaxdepth, skip;
    float x, y, z, r;
    float px, py, pz;
    float dx, dy, dz;
    int ch, n, cell, locked, patch;
    float rootr, rootx, rooty, rootz;

    // Get root cell info
    rootx = node_pos_x[nnodes];
    rooty = node_pos_y[nnodes];
    rootz = node_pos_z[nnodes];
    rootr = radiusd;

    localmaxdepth = 1;
    skip = 1;
    int inc = blockDim.x * gridDim.x;
    i = threadIdx.x + blockIdx.x * blockDim.x;

    while (i < nbodies) {
        if (skip != 0) {
            // Load body position
            skip = 0;
            px = body_pos_x[i];
            py = body_pos_y[i];
            pz = body_pos_z[i];

            // Start at root
            n = nnodes;  // Root node
            depth = 1;
            r = rootr * 0.5f;

            // Determine octant
            dx = dy = dz = -r;
            j = 0;
            if (rootx < px) { j = 1; dx = r; }
            if (rooty < py) { j |= 2; dy = r; }
            if (rootz < pz) { j |= 4; dz = r; }

            x = rootx + dx;
            y = rooty + dy;
            z = rootz + dz;
        }

        // Walk down tree
        ch = children[n * TREE_CHILDREN + j];

        while (ch >= nbodies) {
            // ch is an internal node, descend
            n = ch;
            depth++;
            r *= 0.5f;

            dx = dy = dz = -r;
            j = 0;
            if (x < px) { j = 1; dx = r; }
            if (y < py) { j |= 2; dy = r; }
            if (z < pz) { j |= 4; dz = r; }

            x += dx;
            y += dy;
            z += dz;

            ch = children[n * TREE_CHILDREN + j];
        }

        if (ch != CELL_LOCKED) {
            locked = n * TREE_CHILDREN + j;

            if (ch == CELL_EMPTY) {
                // Empty cell, try to insert body
                if (atomicCAS((int*)&children[locked], CELL_EMPTY, i) == CELL_EMPTY) {
                    localmaxdepth = max(depth, localmaxdepth);
                    i += inc;
                    skip = 1;
                }
            } else {
                // Cell contains another body, need to subdivide
                if (atomicCAS((int*)&children[locked], ch, CELL_LOCKED) == ch) {
                    // Check for coincident bodies (would cause infinite loop)
                    if (body_pos_x[ch] == px && body_pos_y[ch] == py && body_pos_z[ch] == pz) {
                        // Jitter position slightly
                        body_pos_x[i] *= 0.99999f;
                        body_pos_y[i] *= 0.99999f;
                        body_pos_z[i] *= 0.99999f;
                        px = body_pos_x[i];
                        py = body_pos_y[i];
                        pz = body_pos_z[i];
                        skip = 0;
                        children[locked] = ch;  // Unlock
                        continue;
                    }

                    // Create new cells until bodies are separated
                    patch = -1;
                    do {
                        cell = atomicSub(&bottomd, 1) - 1;
                        if (cell <= nbodies) {
                            errd = 1;  // Out of nodes
                            break;
                        }

                        if (patch != -1) {
                            children[n * TREE_CHILDREN + j] = cell;
                        }
                        patch = max(patch, cell);

                        depth++;
                        n = cell;
                        r *= 0.5f;

                        // Store cell center
                        node_pos_x[cell] = x;
                        node_pos_y[cell] = y;
                        node_pos_z[cell] = z;

                        // Insert existing body (ch) into correct octant
                        j = 0;
                        if (x < body_pos_x[ch]) j = 1;
                        if (y < body_pos_y[ch]) j |= 2;
                        if (z < body_pos_z[ch]) j |= 4;
                        children[cell * TREE_CHILDREN + j] = ch;

                        // Compute octant for new body
                        j = 0;
                        dx = dy = dz = -r;
                        if (x < px) { j = 1; dx = r; }
                        if (y < py) { j |= 2; dy = r; }
                        if (z < pz) { j |= 4; dz = r; }
                        x += dx;
                        y += dy;
                        z += dz;

                        ch = children[n * TREE_CHILDREN + j];
                    } while (ch >= 0);

                    // Insert new body
                    children[n * TREE_CHILDREN + j] = i;
                    localmaxdepth = max(depth, localmaxdepth);
                    i += inc;
                    skip = 2;
                }
            }
        }
        __syncthreads();

        // Finalize patch pointer
        if (skip == 2) {
            children[locked] = patch;
        }
    }

    atomicMax(&maxdepthd, localmaxdepth);
}

//-----------------------------------------------------------------------------
// ClearKernel2: Reset mass and start arrays for summarization
//-----------------------------------------------------------------------------
__global__ __launch_bounds__(THREADS_SUMMARY, FACTOR_SUMMARY)
void ClearKernel2(
    int nnodes,
    volatile int* __restrict__ start,
    volatile float* __restrict__ node_mass)
{
    int stride = blockDim.x * gridDim.x;
    int i = threadIdx.x + blockIdx.x * blockDim.x + nnodes;  // Start at first node after bodies

    while (i <= nnodes) {
        node_mass[i] = -1.0f;  // Mark as not computed
        start[i] = -1;
        i += stride;
    }
}

//-----------------------------------------------------------------------------
// SummarizationKernel: Compute center of mass bottom-up
//-----------------------------------------------------------------------------
__global__ __launch_bounds__(THREADS_SUMMARY, FACTOR_SUMMARY)
void SummarizationKernel(
    int nnodes,
    int nbodies,
    volatile int* __restrict__ count,
    volatile int* __restrict__ children,
    volatile float* __restrict__ body_mass,
    volatile float* __restrict__ node_mass,
    volatile float* __restrict__ body_pos_x,
    volatile float* __restrict__ body_pos_y,
    volatile float* __restrict__ body_pos_z,
    volatile float* __restrict__ node_pos_x,
    volatile float* __restrict__ node_pos_y,
    volatile float* __restrict__ node_pos_z)
{
    int i, j, ch, cnt;
    float m, cm, cmx, cmy, cmz;
    int missing;

    int stride = blockDim.x * gridDim.x;
    int bottom = bottomd;

    // Process from bottom of tree to root
    i = threadIdx.x + blockIdx.x * blockDim.x + bottom;

    while (i <= nnodes) {
        if (node_mass[i] < 0.0f) {
            // Not yet computed, check if children are ready
            missing = 0;
            cm = 0.0f;
            cmx = cmy = cmz = 0.0f;
            cnt = 0;

            for (j = 0; j < TREE_CHILDREN; j++) {
                ch = children[i * TREE_CHILDREN + j];
                if (ch >= 0) {
                    if (ch < nbodies) {
                        // Child is a body
                        m = body_mass[ch];
                        cm += m;
                        cmx += body_pos_x[ch] * m;
                        cmy += body_pos_y[ch] * m;
                        cmz += body_pos_z[ch] * m;
                        cnt++;
                    } else {
                        // Child is a node
                        m = node_mass[ch];
                        if (m < 0.0f) {
                            missing++;
                        } else {
                            cm += m;
                            cmx += node_pos_x[ch] * m;
                            cmy += node_pos_y[ch] * m;
                            cmz += node_pos_z[ch] * m;
                            cnt += count[ch];
                        }
                    }
                }
            }

            if (missing == 0) {
                // All children ready, compute center of mass
                count[i] = cnt;
                float inv_cm = 1.0f / cm;
                node_pos_x[i] = cmx * inv_cm;
                node_pos_y[i] = cmy * inv_cm;
                node_pos_z[i] = cmz * inv_cm;
                __threadfence();
                node_mass[i] = cm;
            }
        }
        i += stride;
    }
}

//-----------------------------------------------------------------------------
// SortKernel: Sort bodies into depth-first order
//-----------------------------------------------------------------------------
__global__ __launch_bounds__(THREADS_SORT, FACTOR_SORT)
void SortKernel(
    int nnodes,
    int nbodies,
    volatile int* __restrict__ sorted,
    volatile int* __restrict__ count,
    volatile int* __restrict__ start,
    volatile int* __restrict__ children)
{
    int i, j, ch, st, cnt;
    int stride = blockDim.x * gridDim.x;
    int bottom = bottomd;

    // Initialize start for root
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        start[nnodes] = 0;
    }
    __syncthreads();

    // Process nodes from root to bottom
    i = threadIdx.x + blockIdx.x * blockDim.x + bottom;

    while (i <= nnodes) {
        st = start[i];
        if (st >= 0) {
            // Node is ready
            cnt = 0;
            for (j = 0; j < TREE_CHILDREN; j++) {
                ch = children[i * TREE_CHILDREN + j];
                if (ch >= 0) {
                    if (ch < nbodies) {
                        // Body: assign sort index
                        sorted[st + cnt] = ch;
                        cnt++;
                    } else {
                        // Node: set start for children
                        start[ch] = st + cnt;
                        cnt += count[ch];
                    }
                }
            }
        }
        i += stride;
    }
}

//-----------------------------------------------------------------------------
// ForceCalculationKernel: Compute repulsive forces using Barnes-Hut
//-----------------------------------------------------------------------------
__global__ __launch_bounds__(THREADS_FORCE, FACTOR_FORCE)
void ForceCalculationKernel(
    int nnodes,
    int nbodies,
    float theta_sq,
    float repulsion_k,
    volatile int* __restrict__ sorted,
    volatile int* __restrict__ children,
    volatile float* __restrict__ body_mass,
    volatile float* __restrict__ node_mass,
    volatile float* __restrict__ body_pos_x,
    volatile float* __restrict__ body_pos_y,
    volatile float* __restrict__ body_pos_z,
    volatile float* __restrict__ node_pos_x,
    volatile float* __restrict__ node_pos_y,
    volatile float* __restrict__ node_pos_z,
    volatile float* __restrict__ force_x,
    volatile float* __restrict__ force_y,
    volatile float* __restrict__ force_z)
{
    int i, j, k, n, depth, base, sbase;
    float px, py, pz, dx, dy, dz, dist_sq, dist, f;
    float fx, fy, fz;

    // Shared memory for traversal stack
    __shared__ volatile int pos[MAX_TREE_DEPTH * THREADS_FORCE / WARPSIZE];
    __shared__ volatile int node[MAX_TREE_DEPTH * THREADS_FORCE / WARPSIZE];
    __shared__ float dq[MAX_TREE_DEPTH * THREADS_FORCE / WARPSIZE];

    // Compute distance thresholds for each depth
    if (threadIdx.x == 0) {
        float tmp = radiusd * 2.0f;
        dq[0] = tmp * tmp * theta_sq;
        for (i = 1; i < maxdepthd; i++) {
            dq[i] = dq[i - 1] * 0.25f;  // Each level has half the side length
            dq[i - 1] += EPSILON;
        }
        dq[i - 1] += EPSILON;
    }
    __syncthreads();

    if (maxdepthd <= MAX_TREE_DEPTH) {
        // Warp-based organization
        base = threadIdx.x / WARPSIZE;
        sbase = base * WARPSIZE;
        j = base * MAX_TREE_DEPTH;

        // Copy distance thresholds to warp's portion
        if (threadIdx.x - sbase < MAX_TREE_DEPTH) {
            dq[threadIdx.x - sbase + j] = dq[threadIdx.x - sbase];
        }
        __syncthreads();

        // Each thread processes bodies in sorted order
        for (k = threadIdx.x + blockIdx.x * blockDim.x; k < nbodies; k += blockDim.x * gridDim.x) {
            i = sorted[k];

            px = body_pos_x[i];
            py = body_pos_y[i];
            pz = body_pos_z[i];

            fx = fy = fz = 0.0f;

            // Initialize stack
            depth = j;
            if (threadIdx.x == sbase) {
                pos[j] = 0;
                node[j] = nnodes * TREE_CHILDREN;  // Root's children
            }

            // Traverse tree
            do {
                int pd = pos[depth];
                int nd = node[depth];

                while (pd < TREE_CHILDREN) {
                    n = children[nd + pd];
                    pd++;

                    if (n >= 0) {
                        // Get position
                        if (n < nbodies) {
                            dx = px - body_pos_x[n];
                            dy = py - body_pos_y[n];
                            dz = pz - body_pos_z[n];
                        } else {
                            dx = px - node_pos_x[n];
                            dy = py - node_pos_y[n];
                            dz = pz - node_pos_z[n];
                        }

                        dist_sq = dx * dx + dy * dy + dz * dz + EPSILON;

                        if (n < nbodies) {
                            // Body: compute direct force
                            if (n != i) {
                                dist = rsqrtf(dist_sq);
                                f = repulsion_k * body_mass[i] * body_mass[n] * dist * dist * dist;
                                fx += f * dx;
                                fy += f * dy;
                                fz += f * dz;
                            }
                        } else if (__all_sync(__activemask(), dist_sq >= dq[depth - j])) {
                            // Node: use approximation if far enough
                            dist = rsqrtf(dist_sq);
                            f = repulsion_k * body_mass[i] * node_mass[n] * dist * dist * dist;
                            fx += f * dx;
                            fy += f * dy;
                            fz += f * dz;
                        } else {
                            // Node too close: descend
                            if (threadIdx.x == sbase) {
                                pos[depth] = pd;
                                node[depth] = nd;
                            }
                            depth++;
                            pd = 0;
                            nd = n * TREE_CHILDREN;
                        }
                    } else {
                        pd = TREE_CHILDREN;  // Empty cell, skip rest
                    }
                }
                depth--;
            } while (depth >= j);

            // Accumulate forces (atomic since multiple threads may update same body)
            atomicAdd((float*)&force_x[i], fx);
            atomicAdd((float*)&force_y[i], fy);
            atomicAdd((float*)&force_z[i], fz);
        }
    }
}

//-----------------------------------------------------------------------------
// AttractiveForceKernel: Compute spring forces from edges
//-----------------------------------------------------------------------------
__global__
void AttractiveForceKernel(
    int nedges,
    float spring_k,
    const uint32_t* __restrict__ edge_src,
    const uint32_t* __restrict__ edge_dst,
    const float* __restrict__ edge_rest_len,
    const float* __restrict__ edge_strength,
    volatile float* __restrict__ body_pos_x,
    volatile float* __restrict__ body_pos_y,
    volatile float* __restrict__ body_pos_z,
    volatile float* __restrict__ force_x,
    volatile float* __restrict__ force_y,
    volatile float* __restrict__ force_z)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (i < nedges) {
        uint32_t src = edge_src[i];
        uint32_t dst = edge_dst[i];

        float dx = body_pos_x[dst] - body_pos_x[src];
        float dy = body_pos_y[dst] - body_pos_y[src];
        float dz = body_pos_z[dst] - body_pos_z[src];

        float dist = sqrtf(dx * dx + dy * dy + dz * dz + EPSILON);
        float rest = edge_rest_len[i];
        if (rest <= 0.0f) rest = 1.0f;  // Default rest length

        float displacement = dist - rest;
        float k = spring_k * edge_strength[i];
        float f = k * displacement / dist;

        float fx = f * dx;
        float fy = f * dy;
        float fz = f * dz;

        // Apply to both endpoints (opposite directions)
        atomicAdd((float*)&force_x[src], fx);
        atomicAdd((float*)&force_y[src], fy);
        atomicAdd((float*)&force_z[src], fz);

        atomicAdd((float*)&force_x[dst], -fx);
        atomicAdd((float*)&force_y[dst], -fy);
        atomicAdd((float*)&force_z[dst], -fz);

        i += stride;
    }
}

//-----------------------------------------------------------------------------
// GravityKernel: Optional center gravity
//-----------------------------------------------------------------------------
__global__
void GravityKernel(
    int nbodies,
    float gravity_k,
    float center_x,
    float center_y,
    float center_z,
    volatile float* __restrict__ body_pos_x,
    volatile float* __restrict__ body_pos_y,
    volatile float* __restrict__ body_pos_z,
    volatile float* __restrict__ body_mass,
    volatile float* __restrict__ force_x,
    volatile float* __restrict__ force_y,
    volatile float* __restrict__ force_z)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (i < nbodies) {
        float dx = center_x - body_pos_x[i];
        float dy = center_y - body_pos_y[i];
        float dz = center_z - body_pos_z[i];

        float f = gravity_k * body_mass[i];

        atomicAdd((float*)&force_x[i], f * dx);
        atomicAdd((float*)&force_y[i], f * dy);
        atomicAdd((float*)&force_z[i], f * dz);

        i += stride;
    }
}

//-----------------------------------------------------------------------------
// IntegrationKernel: Update positions using velocity Verlet with damping
//-----------------------------------------------------------------------------
__global__ __launch_bounds__(THREADS_INTEGRATE, FACTOR_INTEGRATE)
void IntegrationKernel(
    int nbodies,
    float damping,
    float max_disp,
    const bool* __restrict__ pinned,
    volatile float* __restrict__ body_pos_x,
    volatile float* __restrict__ body_pos_y,
    volatile float* __restrict__ body_pos_z,
    volatile float* __restrict__ vel_x,
    volatile float* __restrict__ vel_y,
    volatile float* __restrict__ vel_z,
    volatile float* __restrict__ force_x,
    volatile float* __restrict__ force_y,
    volatile float* __restrict__ force_z,
    volatile float* __restrict__ displacement_out)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (i < nbodies) {
        if (!pinned[i]) {
            // Update velocity with damping
            float vx = (vel_x[i] + force_x[i]) * damping;
            float vy = (vel_y[i] + force_y[i]) * damping;
            float vz = (vel_z[i] + force_z[i]) * damping;

            // Clamp displacement
            float disp = sqrtf(vx * vx + vy * vy + vz * vz);
            if (disp > max_disp) {
                float scale = max_disp / disp;
                vx *= scale;
                vy *= scale;
                vz *= scale;
                disp = max_disp;
            }

            // Update position
            body_pos_x[i] += vx;
            body_pos_y[i] += vy;
            body_pos_z[i] += vz;

            // Store velocity for next iteration
            vel_x[i] = vx;
            vel_y[i] = vy;
            vel_z[i] = vz;

            // Output displacement for convergence check
            displacement_out[i] = disp;
        } else {
            displacement_out[i] = 0.0f;
        }

        // Clear forces for next iteration
        force_x[i] = 0.0f;
        force_y[i] = 0.0f;
        force_z[i] = 0.0f;

        i += stride;
    }
}

//-----------------------------------------------------------------------------
// Host helpers
//-----------------------------------------------------------------------------

void reset_device_variables(int nnodes, cudaStream_t stream) {
    // Reset device variables before BoundingBoxKernel
    int zero = 0;
    float fmax = FLT_MAX;
    float fmin = -FLT_MAX;

    cudaMemcpyToSymbolAsync(bottomd, &nnodes, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(maxdepthd, &zero, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(errd, &zero, sizeof(int), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(minxd, &fmax, sizeof(float), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(minyd, &fmax, sizeof(float), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(minzd, &fmax, sizeof(float), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(maxxd, &fmin, sizeof(float), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(maxyd, &fmin, sizeof(float), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(maxzd, &fmin, sizeof(float), 0, cudaMemcpyHostToDevice, stream);
}

bool check_kernel_errors(cudaStream_t stream) {
    cudaStreamSynchronize(stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return false;
    }

    int errd_host;
    cudaMemcpyFromSymbol(&errd_host, errd, sizeof(int));
    return errd_host == 0;
}

} // namespace viz::layout::cuda
