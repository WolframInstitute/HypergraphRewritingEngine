// GenMC harness: a class's frame owner and its step are observed together or not at all.
//
// WHY THIS EXISTS, AND WHAT IT WOULD HAVE CAUGHT. The device published a class's frame as TWO
// map inserts under one flag: the winner of `frame` went on to write `frame_step`, and a thread
// that LOST the first insert returned immediately and read the second. It found that slot EMPTY
// -- not locked, so there was nothing to wait on -- and took its own depth as the class's step.
// Every event it replayed then signed with an instance depth instead of the class's, which makes
// the two signature sets disjoint (quotient_replay_core.hpp:141).
//
// Nothing caught it, and nothing could: the code is in gpu/include/hg_gpu/quotient_expansion.hpp,
// which is CUDA. GenMC's interpreter cannot execute device code, so the device layer had NO model
// checking of any kind -- its only cover was gpu_differential_tests, which is deterministic
// CPU/GPU agreement and passes every time an intermittent race does not fire.
//
// WHAT IS CHECKED, STATED NARROWLY. This harness runs the CALLER-SIDE PUBLICATION SHAPE -- one
// exchange carrying both halves, versus two exchanges with a fallback on the second -- over the
// REAL host ConcurrentMap. It does NOT check gpu/include/hg_gpu/hash_table.hpp: that map's own
// atomics are device code, GenMC cannot execute them, and nothing here compiles them. The claim
// is therefore about the SHAPE, which is what was wrong and which is identical on both sides,
// and not about the device map's implementation, which remains unchecked for the reason set out
// in README.md under "Why there is no harness for the DEVICE termination protocol".
//
// That section warns that a transcription reads as coverage while proving nothing about the code
// that ships, and the warning applies here too, bounded: the two-exchange pattern admits a torn
// read against ANY map providing insert-if-absent with no cross-key ordering, which the device
// map does provide. What a drift in the device map's own semantics would do to this result is
// not covered.
//
// THE PROPERTY. Two threads race to give one class its frame, offering DIFFERENT (step, sid)
// pairs. Whatever any thread subsequently observes for that class must be a pair that was
// actually OFFERED -- never a mixture, and never an owner without its step. That is what "publish
// complete or not at all" means, and it is exactly what the two-map form could not provide.
//
// CALIBRATED, which is what makes a passing run worth anything:
//
//     HG_HARNESS_DEFINES=-DHG_TWO_MAP_FRAME verification/genmc/run.sh frame_publication_is_atomic
//
// reinstates the shipped-then-fixed protocol -- `frame` and `frame_step` as separate maps, only
// the winner writing the second, the loser reading it and falling back -- and GenMC reports a
// safety violation at once (exit 42, 0 complete executions, witness g_seen[0] = 429496729623 =
// id_key(99, 22): the fallback step 99 carried alongside owner sid 22, so thread A lost the frame
// exchange, read B as the owner, and found B's step not yet published). Without the switch the
// single packed publication is used and the harness explores 32 complete executions clean.
//
// GENMC-ARGS: --disable-estimation
// GENMC-EXPECT: pass

#include <pthread.h>
#include <cassert>
#include <cstdint>

#include "genmc_support.hpp"
#include "hgcommon/core.hpp"
#include "hypergraph/concurrent_map.hpp"

namespace {

using Map = hypergraph::ConcurrentMap<uint64_t, uint64_t>;

// One class, contested. The two threads differ in BOTH fields, so a mixture of the two offers is
// distinguishable from either offer -- which is the whole point: a torn publication has to be
// detectable, not merely improbable.
constexpr uint64_t kClass = 0x9e3779b97f4a7c15ull;
constexpr uint32_t kStepA = 3, kSidA = 11;
constexpr uint32_t kStepB = 7, kSidB = 22;

Map* g_frame;
#ifdef HG_TWO_MAP_FRAME
hypergraph::ConcurrentMap<uint64_t, uint64_t>* g_frame_step;
#endif

// What each thread observed for the class after publishing: the packed pair, or 0 for "absent".
uint64_t g_seen[2];

// The fallback a caller supplies when the class holds no frame. Distinct from every offered step
// so that taking it is visible in the assertions rather than indistinguishable from a real read.
constexpr uint32_t kFallback = 99;

void publish_and_read(int idx, uint32_t step, uint32_t sid) {
#ifdef HG_TWO_MAP_FRAME
    // THE DEFECT, reinstated exactly: the owner is published first, and only the winner of that
    // exchange goes on to publish the step. A loser proceeds straight to the read.
    const bool won = g_frame->insert_if_absent(kClass, hgcommon::id_key(0u, sid)).second;
    if (won) g_frame_step->insert_if_absent(kClass, hgcommon::id_key(0u, step));

    const auto owner = g_frame->lookup(kClass);
    if (!owner) { g_seen[idx] = 0; return; }
    const auto st = g_frame_step->lookup(kClass);
    const uint32_t observed_step = st ? hgcommon::id_pair_from_key(*st).b : kFallback;
    g_seen[idx] = hgcommon::id_key(observed_step, hgcommon::id_pair_from_key(*owner).b);
#else
    // THE FIX: one exchange carries both halves, so there is no second publication for another
    // thread to arrive ahead of.
    g_frame->insert_if_absent(kClass, hgcommon::id_key(step, sid));
    const auto held = g_frame->lookup(kClass);
    g_seen[idx] = held ? *held : 0;
#endif
}

void* w_a(void*) { publish_and_read(0, kStepA, kSidA); return nullptr; }
void* w_b(void*) { publish_and_read(1, kStepB, kSidB); return nullptr; }

bool is_offered(uint64_t packed) {
    return packed == hgcommon::id_key(kStepA, kSidA)
        || packed == hgcommon::id_key(kStepB, kSidB);
}

}  // namespace

int main() {
    Map frame(/*initial_capacity=*/4);
    g_frame = &frame;
#ifdef HG_TWO_MAP_FRAME
    hypergraph::ConcurrentMap<uint64_t, uint64_t> frame_step(/*initial_capacity=*/4);
    g_frame_step = &frame_step;
#endif

    pthread_t t0, t1;
    pthread_create(&t0, nullptr, w_a, nullptr);
    pthread_create(&t1, nullptr, w_b, nullptr);
    pthread_join(t0, nullptr);
    pthread_join(t1, nullptr);

    // A1: whatever a thread saw, it is a pair somebody actually offered. A mixture -- an owner
    // from one publication with a step from neither -- is the defect, and it is what the two-map
    // arm produces when the loser reads before the winner's second insert.
    for (int i = 0; i < 2; ++i) {
        assert(g_seen[i] != 0);
        assert(is_offered(g_seen[i]));
    }

    // A2: one class has one frame, so both threads agree on which publication won.
    assert(g_seen[0] == g_seen[1]);
    return 0;
}
