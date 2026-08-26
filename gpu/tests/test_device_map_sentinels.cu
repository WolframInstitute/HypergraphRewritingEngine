// A key equal to a reserved sentinel must still be stored and found, and a rule the device cannot
// represent must be refused rather than truncated.
//
// SENTINELS. The device map marks a free slot with EMPTY and a slot mid-publication with LOCKED.
// A genuine key equal to either is not merely mis-hashed, it is INVISIBLE: inserting EMPTY leaves
// the slot reading as free so the entry is silently never stored, and inserting LOCKED leaves
// readers waiting on a publication that already happened. Nothing reports it -- the run just
// behaves as though that state, causal triple or branchial pair did not exist.
//
// The keys are 64-bit hashes (canonical state hashes, hash_causal_triple), so both values are
// reachable rather than merely representable, and this class already cost four correctness bugs
// on the host before its map began rejecting them. Device code cannot throw, so the map folds the
// two values onto neighbours instead -- and folds them INSIDE, because a normalisation that
// insert applies and lookup forgets stores the entry where nothing will look for it.
//
// RULE DIMENSIONS. DeviceRule holds lhs[] and rhs[] at kMaxPatternEdges with uint8_t counts, so an
// oversized rule truncated on the cast and was then written past the end of the array -- a
// host-side buffer overflow reached from caller data, before any kernel launched. A variable
// index at or above MAX_VARS shifted a 32-bit mask by its own width, which is undefined. These
// are programmer errors in the caller's rule, so they throw; the partial-work-plus-warning
// contract covers a run that outgrows its pools, not a rule that cannot be represented at all.

#include <gtest/gtest.h>

#include "hg_gpu/evolve.hpp"
#include "hg_gpu/hash_table.hpp"
#include "hg_gpu/match.hpp"
#include "hg_gpu/exploration.hpp"
#include "hg_gpu/cuda_check.hpp"

#include <vector>

namespace {

using Map = hg_gpu::ConcurrentMap<uint64_t, uint32_t>;

__global__ void k_insert(Map::DeviceView m, const uint64_t* keys, uint32_t n,
                         uint32_t* inserted_flags) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    inserted_flags[i] = m.insert_if_absent(keys[i], i + 1u).inserted ? 1u : 0u;
}

// Deduplication with a canonical hash of 0. 0 is not a hash -- it is what the per-state hash
// array holds for "not computed yet" -- so the state must be KEPT and the run must report it,
// rather than every such state sharing one dedup slot and all but the first vanishing.
__global__ void k_dedup_zero_hash(hg_gpu::DeviceState ds, Map::DeviceView m, uint32_t n,
                                  uint32_t* survived) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    survived[i] = hg_gpu::state_survives_dedup(ds, i, /*hash=*/0ull, m, /*dedup=*/true,
                                               UINT32_MAX, 0ull, 0u) ? 1u : 0u;
}

// The value a caller asks to store, so a test can offer one the map reserves. k_insert stores
// i+1 and therefore can never offer zero, which is the one value that has to be checked.
__global__ void k_insert_value(Map::DeviceView m, const uint64_t* keys, const uint32_t* vals,
                               uint32_t n, uint32_t* inserted_flags) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    inserted_flags[i] = m.insert_if_absent(keys[i], vals[i]).inserted ? 1u : 0u;
}

// found and value REPORTED SEPARATELY. k_lookup answers `found ? value : 0`, which cannot tell
// a missing key from a stored zero -- and a stored zero is the subject here.
__global__ void k_lookup_reporting_found(Map::DeviceView m, const uint64_t* keys, uint32_t n,
                                         uint32_t* found, uint32_t* values) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const auto r = m.lookup(keys[i]);
    found[i]  = r.found ? 1u : 0u;
    values[i] = r.value;
}

// Every thread offers the SAME key, each with its own tid as the value -- so thread 0 offers 0.
__global__ void k_race_one_key(Map::DeviceView m, uint64_t key, uint32_t n,
                               uint32_t* inserted_flags, uint32_t* observed) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const auto r = m.insert_if_absent(key, i);
    inserted_flags[i] = r.inserted ? 1u : 0u;
    observed[i] = r.value;
}

__global__ void k_lookup(Map::DeviceView m, const uint64_t* keys, uint32_t n,
                         uint32_t* found_values) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const auto r = m.lookup(keys[i]);
    found_values[i] = r.found ? r.value : 0u;
}

}  // namespace

// Both sentinels, plus ordinary neighbours, inserted then read back.
TEST(DeviceMapSentinels, ReservedKeysAreStoredAndFound) {
    const std::vector<uint64_t> keys = {
        0ull,                       // EMPTY
        ~0ull,                      // LOCKED
        1ull,                       // the neighbour EMPTY folds onto
        ~0ull - 1ull,               // the neighbour LOCKED folds onto
        0x0123456789ABCDEFull,      // an ordinary key, as a control
    };
    const uint32_t n = static_cast<uint32_t>(keys.size());

    Map map(1024);
    uint64_t* d_keys = nullptr;
    uint32_t* d_ins = nullptr;
    uint32_t* d_found = nullptr;
    ASSERT_EQ(cudaMalloc(&d_keys, sizeof(uint64_t) * n), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_ins, sizeof(uint32_t) * n), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_found, sizeof(uint32_t) * n), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_keys, keys.data(), sizeof(uint64_t) * n, cudaMemcpyHostToDevice),
              cudaSuccess);

    // One thread per key so the inserts do not race each other; the question here is
    // representability, not concurrency.
    k_insert<<<1, n>>>(map.view(), d_keys, n, d_ins);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    k_lookup<<<1, n>>>(map.view(), d_keys, n, d_found);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uint32_t> ins(n), found(n);
    ASSERT_EQ(cudaMemcpy(ins.data(), d_ins, sizeof(uint32_t) * n, cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(found.data(), d_found, sizeof(uint32_t) * n, cudaMemcpyDeviceToHost),
              cudaSuccess);

    const char* names[] = {"EMPTY (0)", "LOCKED (~0)", "1", "~0-1", "ordinary"};
    for (uint32_t i = 0; i < n; ++i) {
        // Folding makes 0 and 1 the same key, and ~0 and ~0-1 the same key, so the SECOND of each
        // pair legitimately reports not-inserted. What must never happen is a key that is neither
        // stored nor findable.
        EXPECT_NE(found[i], 0u)
            << "key " << names[i] << " was inserted and then could not be found, so it is stored "
            << "where nothing will look for it -- or was never stored at all";
    }
    cudaFree(d_keys); cudaFree(d_ins); cudaFree(d_found);
}

// An oversized or unrepresentable rule must be refused at the boundary.
// GROUND-TRUTH for kUncomputedStateHash: the path is taken, every state survives, and the run
// carries a warning. Without this the kind is a branch nothing has ever executed.
TEST(DeviceMapSentinels, ZeroHashKeepsEveryStateAndWarns) {
    hg_gpu::EngineState engine(hg_gpu::EngineConfig{});
    Map map(1024);
    const uint32_t n = 8;
    uint32_t* d_surv = nullptr;
    HG_CUDA_CHECK(cudaMalloc(&d_surv, sizeof(uint32_t) * n), "surv alloc");

    k_dedup_zero_hash<<<1, n>>>(engine.device(), map.view(), n, d_surv);
    HG_CUDA_CHECK(cudaDeviceSynchronize(), "zero-hash dedup sync");

    std::vector<uint32_t> surv(n, 0);
    HG_CUDA_CHECK(cudaMemcpy(surv.data(), d_surv, sizeof(uint32_t) * n, cudaMemcpyDeviceToHost),
                  "surv copy");
    cudaFree(d_surv);

    for (uint32_t i = 0; i < n; ++i)
        EXPECT_EQ(surv[i], 1u) << "state " << i << " was deduplicated on a hash of 0, so it is "
                               << "a subtree the run will never explore";

    std::vector<hg_gpu::OverflowWarning> warnings;
    engine.collect_warnings_into(warnings, "zero-hash dedup");
    bool reported = false;
    for (const auto& w : warnings)
        if (w.kind == hg_gpu::ErrorKind::kUncomputedStateHash) { reported = true; break; }
    EXPECT_TRUE(reported) << "the run kept " << n << " un-deduplicated states and said nothing";
}

// ZERO IS AN ORDINARY VALUE, and it has to be, because the callers store ids raw: exploration
// inserts `sid`, event_identity inserts `eid`, persistent.cu inserts `sid`, and the first state
// and the first event are numbered 0. A map that reserves zero makes exactly those two entries
// read as unclaimed -- found by nobody, and re-claimed by every later thread.
TEST(DeviceMapSentinels, ZeroIsAnOrdinaryStoredValue) {
    const std::vector<uint64_t> keys = {0x11ull, 0x22ull, 0x33ull};
    const std::vector<uint32_t> vals = {0u, 1u, 7u};   // 0 first: the id of the first state
    const uint32_t n = static_cast<uint32_t>(keys.size());

    Map map(1024);
    uint64_t* d_keys = nullptr; uint32_t* d_vals = nullptr;
    uint32_t* d_ins = nullptr; uint32_t* d_found = nullptr; uint32_t* d_got = nullptr;
    ASSERT_EQ(cudaMalloc(&d_keys, sizeof(uint64_t) * n), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_vals, sizeof(uint32_t) * n), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_ins, sizeof(uint32_t) * n), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_found, sizeof(uint32_t) * n), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_got, sizeof(uint32_t) * n), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_keys, keys.data(), sizeof(uint64_t) * n, cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_vals, vals.data(), sizeof(uint32_t) * n, cudaMemcpyHostToDevice),
              cudaSuccess);

    k_insert_value<<<1, n>>>(map.view(), d_keys, d_vals, n, d_ins);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    k_lookup_reporting_found<<<1, n>>>(map.view(), d_keys, n, d_found, d_got);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uint32_t> ins(n), found(n), got(n);
    ASSERT_EQ(cudaMemcpy(ins.data(), d_ins, sizeof(uint32_t) * n, cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(found.data(), d_found, sizeof(uint32_t) * n, cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(got.data(), d_got, sizeof(uint32_t) * n, cudaMemcpyDeviceToHost),
              cudaSuccess);
    cudaFree(d_keys); cudaFree(d_vals); cudaFree(d_ins); cudaFree(d_found); cudaFree(d_got);

    for (uint32_t i = 0; i < n; ++i) {
        EXPECT_EQ(ins[i], 1u) << "key " << keys[i] << " was not inserted";
        EXPECT_EQ(found[i], 1u)
            << "value " << vals[i] << " was stored under key " << keys[i]
            << " and the slot then read as unclaimed, so nothing can ever find it";
        EXPECT_EQ(got[i], vals[i]) << "key " << keys[i] << " gave back the wrong value";
    }
}

// FIRST WRITER WINS EVEN WHEN THE FIRST WRITER STORES ZERO. Every thread offers one key with
// its own tid, so thread 0 offers the value 0; if that value reads as "not published", every
// other thread offers into the slot as well and several of them are told they won it.
TEST(DeviceMapSentinels, ConcurrentInsertOfOneKeyHasOneWinnerWhenTheValueIsZero) {
    constexpr uint32_t kThreads = 1024;
    const uint64_t key = 0xABCDEFull;

    Map map(4096);
    uint32_t* d_ins = nullptr; uint32_t* d_obs = nullptr;
    ASSERT_EQ(cudaMalloc(&d_ins, sizeof(uint32_t) * kThreads), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_obs, sizeof(uint32_t) * kThreads), cudaSuccess);

    k_race_one_key<<<4, 256>>>(map.view(), key, kThreads, d_ins, d_obs);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<uint32_t> ins(kThreads), obs(kThreads);
    ASSERT_EQ(cudaMemcpy(ins.data(), d_ins, sizeof(uint32_t) * kThreads, cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(obs.data(), d_obs, sizeof(uint32_t) * kThreads, cudaMemcpyDeviceToHost),
              cudaSuccess);
    cudaFree(d_ins); cudaFree(d_obs);

    uint32_t winners = 0, winner_tid = UINT32_MAX;
    for (uint32_t i = 0; i < kThreads; ++i)
        if (ins[i]) { ++winners; winner_tid = i; }
    EXPECT_EQ(winners, 1u) << "one key, " << kThreads << " threads, " << winners
                           << " of them told they inserted it";
    for (uint32_t i = 1; i < kThreads; ++i)
        EXPECT_EQ(obs[i], obs[0])
            << "thread " << i << " saw " << obs[i] << " where thread 0 saw " << obs[0]
            << "; every caller must answer from the one stored value";
    // THE STORED VALUE MUST BE THE WINNER'S, which the two checks above do not ask. They both
    // hold when the key exchange and the value exchange are won by DIFFERENT threads: one
    // winner, one agreed value, and the winner carrying a stranger's. A caller that reads both
    // fields -- event_identity does -- then makes the loser its own canonical event while the
    // count records the winner's, so one signature has two identities and the count says one.
    ASSERT_NE(winner_tid, UINT32_MAX);
    for (uint32_t i = 0; i < kThreads; ++i)
        EXPECT_EQ(obs[i], winner_tid)
            << "thread " << i << " answers " << obs[i] << " but thread " << winner_tid
            << " is the one told it inserted; the key and the value went to different threads";
}

TEST(DeviceRuleValidation, UnrepresentableRulesAreRefusedNotTruncated) {
    // Rule edges are lists of VARIABLE indices (uint8_t), not vertex ids.
    auto edge = [](uint8_t a, uint8_t b) { return std::vector<uint8_t>{a, b}; };

    {   // more LHS edges than the fixed array holds
        hg_gpu::RewriteRule r;
        for (int i = 0; i < static_cast<int>(hg_gpu::kMaxPatternEdges) + 4; ++i)
            r.lhs.push_back(edge(static_cast<uint8_t>(i % 8), static_cast<uint8_t>((i + 1) % 8)));
        r.rhs.push_back(edge(0, 1));
        r.num_lhs_vars = 8; r.num_rhs_vars = 8;
        EXPECT_THROW(hg_gpu::make_device_rule(r), std::runtime_error)
            << "an oversized LHS truncated on the uint8_t cast and was written past the array";
    }
    {   // a variable index at or above MAX_VARS, which would shift a 32-bit mask by its width
        hg_gpu::RewriteRule r;
        r.lhs.push_back(edge(0, 1));
        r.rhs.push_back(edge(0, static_cast<uint8_t>(hgcommon::MAX_VARS + 1)));
        r.num_lhs_vars = 2; r.num_rhs_vars = 40;
        EXPECT_THROW(hg_gpu::make_device_rule(r), std::runtime_error)
            << "a variable index at or above MAX_VARS is undefined in new_var_mask";
    }
    {   // an empty LHS matches everywhere and has no binding to apply
        hg_gpu::RewriteRule r;
        r.rhs.push_back(edge(0, 1));
        r.num_lhs_vars = 0; r.num_rhs_vars = 2;
        EXPECT_THROW(hg_gpu::make_device_rule(r), std::runtime_error);
    }
    {   // the control: an ordinary rule still builds
        hg_gpu::RewriteRule r;
        r.lhs.push_back(edge(0, 1));
        r.rhs.push_back(edge(0, 1));
        r.rhs.push_back(edge(1, 2));
        r.num_lhs_vars = 2; r.num_rhs_vars = 3;
        EXPECT_NO_THROW(hg_gpu::make_device_rule(r));
    }
}
