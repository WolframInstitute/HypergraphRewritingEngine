// GPUMC harness: concurrent saturating adds to one device counter lose no update and stop at the
// saturation value.
//
// Runs gpu/include/hg_gpu/saturating.hpp's saturating_add itself (verification/gpumc/device): the
// quotient replay's multiplicity counts (quotient_expansion.hpp, quotient.cu) and the step pool's
// distribution totals (step_pool.cu) are updated with it, and read by the host after the kernel.
//
// THE PROTOCOL. A relaxed load of the counter, then compare-exchange of the saturated sum until
// one exchange finds the value it computed from.
//
// THE SHAPE. Three threads in three blocks add 2, 3 and 4 to a counter starting at `kStart`:
// -DSHAPE_SATURATE starts 6 below hgcommon::QM_SATURATED, so the sum saturates part way.
//
// THE PROPERTY. The counter ends at min(kStart + 9, QM_SATURATED). No access races.
//
// CALIBRATION. -DHG_GPUMC_CALIBRATE_SPLIT_RMW makes the compare-exchange a load and a separate
// store; the checker must report a lost update.
// GPUMC-CELL: below PASS
// GPUMC-CELL: saturate PASS -DSHAPE_SATURATE
// GPUMC-CELL: split_rmw FAIL -DHG_GPUMC_CALIBRATE_SPLIT_RMW

#include "hg_gpumc.hpp"
#include "hg_gpu/saturating.hpp"

#include <cassert>
#include <cstdint>
#include <pthread.h>

namespace {

#if defined(SHAPE_SATURATE)
constexpr uint64_t kStart = hgcommon::QM_SATURATED - 6;
#else
constexpr uint64_t kStart = 100;
#endif

unsigned long long g_counter = kStart;

void* thread_fn(void* arg) {
    const unsigned id = static_cast<unsigned>(reinterpret_cast<long>(arg));
    hg_gpumc::enter(id, 0);
    HG_NAMESPACE::gpu::saturating_add(&g_counter, id + 2u);
    return nullptr;
}

}  // namespace

int main() {
    const unsigned lanes[3] = {1, 1, 1};
    hg_gpumc::launch(3, lanes);
    pthread_t t[3];
    for (long i = 0; i < 3; ++i) pthread_create(&t[i], nullptr, thread_fn, reinterpret_cast<void*>(i));
    for (int i = 0; i < 3; ++i) pthread_join(t[i], nullptr);
    const uint64_t want = hgcommon::qm_sat_add(kStart, 9);
    assert(g_counter == want && "an add was lost or the sum passed the saturation value");
    return 0;
}
