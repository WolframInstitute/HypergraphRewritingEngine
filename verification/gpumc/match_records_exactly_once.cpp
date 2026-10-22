// GPUMC harness: the persistent kernel hands every match record to exactly one rewriting block,
// which reads it fully written.
//
// Runs gpu/src/persistent.cu's claim_next_record and claim_next_records (over readable_records)
// and gpu/include/hg_gpu/match.hpp's publish_match and await_match themselves, over Pool::claim,
// with the device's atomics (verification/gpumc/device).
//
// THE PROTOCOL. A matching block claims a record slot with the pool's add, writes the record and
// publishes it (release store of `published`). A rewriting block claims the next records below the
// readable count with one compare-exchange on the consume cursor (one record, or a batch), waits
// for each record's `published` with acquire, and reads it.
//
// THE SHAPE. Two matching blocks write one record each; one rewriting block claims one record and
// another claims a batch of up to two. Blocks are one thread.
//
// THE PROPERTY. The claimed records are disjoint and each reads as its producer wrote it. No
// access races.
//
// CALIBRATION. -DHG_GPUMC_CALIBRATE_SPLIT_RMW makes the cursor's compare-exchange a load and a
// separate store; the checker must report a record claimed twice.
// GPUMC-CELL: clean PASS -- --disable-function-inliner
// GPUMC-CELL: split_rmw FAIL -DHG_GPUMC_CALIBRATE_SPLIT_RMW -- --disable-function-inliner

#include "hg_gpumc.hpp"
#include "persistent.cu"

#include <cassert>
#include <cstdint>
#include <pthread.h>

namespace {

namespace g = HG_NAMESPACE::gpu;

constexpr uint32_t kCap = 4;

g::MatchRecord g_records[kCap];
uint32_t g_count, g_cursor;
uint32_t g_first[2] = {g::INVALID_ID, g::INVALID_ID};
uint32_t g_n[2];
bool g_payload_ok[2] = {true, true};

g::Pool<g::MatchRecord>::DeviceView found() { return {g_records, &g_count, kCap}; }

bool record_ok(const g::MatchRecord& r) {
    return r.rule_id == r.state_id + 7u && r.step == 1u && r.num_edges == 1u &&
           r.matched_edges[0] == r.state_id + 20u;
}

void* matcher(void* arg) {
    const unsigned id = static_cast<unsigned>(reinterpret_cast<long>(arg));
    hg_gpumc::enter(id, 0);
    const uint32_t k = found().claim();
    assert(k != g::Pool<g::MatchRecord>::kInvalid);
    g::MatchRecord& r = found().at(k);
    r.state_id = id;
    r.rule_id = id + 7u;
    r.step = 1u;
    r.num_edges = 1u;
    r.matched_edges[0] = id + 20u;
    g::publish_match(r);
    return nullptr;
}

void* rewriter(void* arg) {
    const unsigned c = static_cast<unsigned>(reinterpret_cast<long>(arg));
    hg_gpumc::enter(2 + c, 0);
    uint32_t base = g::INVALID_ID, n = 0;
    if (c == 0) {
        base = g::claim_next_record(&g_cursor, found());
        n = base == g::INVALID_ID ? 0u : 1u;
    } else {
        const auto two = [](uint32_t, uint32_t) { return 2u; };
        n = g::claim_next_records(&g_cursor, found(), two, base);
    }
    for (uint32_t i = 0; i < n; ++i) {
        const g::MatchRecord& r = found().at(base + i);
        g::await_match(r);
        if (!record_ok(r)) g_payload_ok[c] = false;
    }
    g_first[c] = base;
    g_n[c] = n;
    return nullptr;
}

}  // namespace

int main() {
    const unsigned lanes[4] = {1, 1, 1, 1};
    hg_gpumc::launch(4, lanes);
    pthread_t t[4];
    for (long i = 0; i < 2; ++i) pthread_create(&t[i], nullptr, matcher, reinterpret_cast<void*>(i));
    for (long i = 0; i < 2; ++i)
        pthread_create(&t[2 + i], nullptr, rewriter, reinterpret_cast<void*>(i));
    for (int i = 0; i < 4; ++i) pthread_join(t[i], nullptr);

    assert(g_payload_ok[0] && g_payload_ok[1] && "a claimed record was read before it was written");
    if (g_n[0] && g_n[1])
        assert((g_first[0] + g_n[0] <= g_first[1] || g_first[1] + g_n[1] <= g_first[0]) &&
               "two blocks claimed one record");
    assert(g_cursor == g_n[0] + g_n[1] && "the cursor passed an unclaimed record");
    return 0;
}
