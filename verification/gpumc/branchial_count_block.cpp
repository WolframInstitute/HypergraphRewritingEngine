// GPUMC harness: the quotient replay's branchial count sums every instance's term once.
//
// Runs gpu/src/quotient.cu's k_qe_count_branchial itself -- the grid-strided terms, the warp's
// shuffle-down reduction, the per-warp shared sums behind the block barrier, and the block's one
// saturating add -- with the device's map, shuffles, barriers and atomics
// (verification/gpumc/device).
//
// THE SHAPE. One block, a warp of two lanes, over three replay instances of one class whose B(c)
// is 5: lane 0 takes instances 0 and 2, lane 1 instance 1. (The kernel's block is 128 threads, four
// warps; one warp writes warp_sum[0] and the others stay zero.)
//
// THE PROPERTY. The count is 3 x 5 = 15. No access races; no barrier divergence.
//
// CALIBRATION. -DCALIBRATE_TWO_BLOCKS runs a second block over the same instances with no
// instance of its own: the kernel's shared sums are one copy per block on the device and one per
// program here, so the two blocks' writes to warp_sum race, and the checker must report it.
// GPUMC-CELL: clean PASS
// GPUMC-CELL: two_blocks FAIL -DCALIBRATE_TWO_BLOCKS

#include "hg_gpumc.hpp"
#include "quotient.cu"

#include <cassert>
#include <cstdint>
#include <new>
#include <pthread.h>

namespace {

namespace g = HG_NAMESPACE::gpu;
using InstNode = g::LockFreeList<g::QeInstRef>::Node;

constexpr uint64_t kClass = 0x77;
constexpr uint32_t kSlots = 4;
#if defined(CALIBRATE_TWO_BLOCKS)
constexpr unsigned kBlocks = 2;
#else
constexpr unsigned kBlocks = 1;
#endif

g::DeviceState* g_ds;
g::QeView* g_qe;
InstNode g_nodes[3];
uint32_t g_nodes_n = 3;
g::MapSlot<uint64_t, uint32_t> g_rep[kSlots];
uint32_t g_class_nmatch[1];
unsigned long long g_class_pairs[1] = {5};
unsigned long long g_counts[2];

void* lane(void* arg) {
    const long id = reinterpret_cast<long>(arg);
    hg_gpumc::enter(static_cast<unsigned>(id / 2), static_cast<unsigned>(id % 2));
    g::k_qe_count_branchial(*g_ds, *g_qe, 0u);
    return nullptr;
}

}  // namespace

int main() {
    const unsigned lanes[2] = {2, 2};
    hg_gpumc::launch(kBlocks, lanes);
    g_ds = new (malloc(sizeof(g::DeviceState))) g::DeviceState{};
    g_qe = new (malloc(sizeof(g::QeView))) g::QeView{};
    g_qe->max_steps = 1;
    g_qe->by_key.pool = {g_nodes, &g_nodes_n, 3};
    for (uint32_t i = 0; i < 3; ++i) g_nodes[i].value = g::QeInstRef{kClass, i, 0};
    for (uint32_t s = 0; s < kSlots; ++s) g_rep[s] = {0, ~uint32_t{0}};
    g_qe->rep = {g_rep, kSlots, kSlots - 1, nullptr};
    // The class's representative is state 0, stored + 1.
    const auto r = g_qe->rep.insert_if_absent(g::qe_expansion_key(*g_ds, kClass, 0), 1u);
    assert(r.inserted);
    g_qe->class_nmatch = g_class_nmatch;
    g_qe->class_nmatch_cap = 1;
    g_qe->class_pairs = g_class_pairs;
    g_qe->qm_counts = g_counts;

    pthread_t t[4];
    for (long i = 0; i < 2 * static_cast<long>(kBlocks); ++i)
        pthread_create(&t[i], nullptr, lane, reinterpret_cast<void*>(i));
    for (unsigned i = 0; i < 2 * kBlocks; ++i) pthread_join(t[i], nullptr);
    assert(g_counts[1] == 15u && "an instance's term was lost or counted twice");
    return 0;
}
