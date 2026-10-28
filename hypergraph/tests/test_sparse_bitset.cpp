// SparseBitset (hypergraph/bitset.hpp) against std::set, through chains of derived states that
// cross from the flat index to the radix index, and the engine's bytes per event on one history.
#include <gtest/gtest.h>

#include <memory>
#include <set>
#include <vector>

#include "hgcommon/sampling_core.hpp"
#include "hypergraph/arena.hpp"
#include "hypergraph/bitset.hpp"
#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"

using namespace hypergraph;

namespace {

struct Lcg {
    uint64_t x;
    uint32_t next(uint32_t m) {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<uint32_t>((x >> 33) % m);
    }
};

void expect_same(const SparseBitset& b, const std::set<uint32_t>& ref, uint32_t universe, int gen) {
    std::vector<uint32_t> got;
    b.for_each([&](uint32_t e) { got.push_back(e); });
    ASSERT_EQ(got, std::vector<uint32_t>(ref.begin(), ref.end())) << "generation " << gen;
    ASSERT_EQ(b.count(), ref.size()) << "generation " << gen;
    for (uint32_t e = 0; e < universe; e += 7) ASSERT_EQ(b.contains(e), ref.count(e) == 1) << e;
}

}  // namespace

// Every generation of a chain keeps its own contents while later generations clear and set
// edges: the chunks and nodes a derived set shares are copied before they are written. Live
// sets range from a few edges in one chunk to thousands over hundreds of chunks.
TEST(SparseBitset, DerivedSetsMatchAReferenceAndLeaveTheirParentsUnchanged) {
    for (const uint32_t spread : {600u, 20000u, 400000u}) {
        ConcurrentHeterogeneousArena arena;
        Lcg r{spread};
        std::vector<SparseBitset> gens;
        std::vector<std::set<uint32_t>> refs;
        std::vector<uint32_t> init;
        for (int i = 0; i < 8; ++i) init.push_back(r.next(spread));
        gens.push_back(SparseBitset::from_edges(init.data(), init.size(), arena));
        refs.emplace_back(init.begin(), init.end());
        uint32_t next_id = spread;
        for (int g = 1; g < 3000; ++g) {
            const size_t from = r.next(static_cast<uint32_t>(gens.size()));
            std::set<uint32_t> ref = refs[from];
            std::vector<uint32_t> consumed, produced;
            for (int k = 0; k < 2 && !ref.empty(); ++k) {
                auto it = ref.begin();
                std::advance(it, r.next(static_cast<uint32_t>(ref.size())));
                consumed.push_back(*it);
                ref.erase(it);
            }
            for (int k = 0; k < 3; ++k) {
                const uint32_t e = k == 0 ? r.next(next_id) : next_id++;
                produced.push_back(e);
                ref.insert(e);
            }
            gens.push_back(SparseBitset::derive(gens[from], consumed.data(), consumed.size(),
                                                produced.data(), produced.size(), arena));
            refs.push_back(std::move(ref));
        }
        for (size_t g = 0; g < gens.size(); g += 13) expect_same(gens[g], refs[g], next_id, int(g));
        expect_same(gens.back(), refs.back(), next_id, int(gens.size() - 1));
    }
}

// One history of a rule whose live edge set grows by one per event, edge ids spread over all
// events (OldestEdge order): the engine's arena bytes per event at 40,000 events are within 1.25
// of those at 10,000.
TEST(SparseBitset, OneHistoryCostsTheSameBytesPerEventAtEveryLength) {
    auto bytes_per_event = [](uint32_t events) {
        const size_t before = arena_block_bytes_live();
        auto hg_heap = std::make_unique<Hypergraph>();
        Hypergraph& hg = *hg_heap;
        hg.set_state_canonicalization_mode(StateCanonicalizationMode::None);
        auto e_heap = std::make_unique<ParallelEvolutionEngine>(&hg, 1);
        ParallelEvolutionEngine& e = *e_heap;
        e.add_rule(make_rule(0).lhs({1, 2, 3}).lhs({3, 4, 5}).rhs({2, 6, 2}).rhs({3, 2, 4}).rhs({5, 6, 1}).build());
        e.set_event_ordering(hgcommon::kEventOrderOldestEdge | (hgcommon::kEventOrderRuleOrdering << 4) |
                             (hgcommon::kEventOrderRuleIndex << 8));
        e.evolve({{0, 0, 0}, {0, 0, 0}}, events);
        EXPECT_EQ(hg.num_events(), events);
        return double(arena_block_bytes_live() - before) / events;
    };
    const double short_run = bytes_per_event(10000), long_run = bytes_per_event(40000);
    EXPECT_LT(long_run, 1.25 * short_run) << short_run << " B/event at 10,000 events, " << long_run
                                          << " at 40,000";
}
