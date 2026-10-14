#pragma once
#include "hgcommon/namespace.hpp"
//
// The CPU side of a session: a Hypergraph and the engine driving it, owned together.
//
// WHY THEY ARE OWNED TOGETHER. ParallelEvolutionEngine holds a POINTER to its Hypergraph, so the
// two have one lifetime and the pair cannot be returned by value. Heap-allocating the holder is
// what makes the graph's address stable, which is also exactly what a session needs: the objects
// outlive the call that built them.
//
// EXTENDING IS NOT RE-EVOLVING. `extend` calls ParallelEvolutionEngine::evolve_more, which
// carries the SAME run further from the frontier where the budget stopped it and keeps the
// states, events and relations already built. Re-running evolve() with a larger step count would
// recompute the whole graph and mint different raw ids, which a caller holding earlier results
// would have no way to notice.
//
// CONTINUABLE IS NOT FREE, which is why it is a constructor argument rather than always on. The
// frontier a continuation resumes from costs 12.5 MB across the oracle corpus and about 3.9% of
// the arena, and a one-shot job never continues. So a session pays for it and a plain `Evolve`
// job does not. evolve_more throws without it -- deliberately, because returning an unchanged
// graph would be a wrong answer that looks like a converged one.

#include "session.hpp"
#include "state_statistics.hpp"

#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <map>
#include <thread>
#include <unordered_set>
#include <vector>

namespace HG_NAMESPACE {
namespace ffi {

class CpuEngineHolder : public EngineHolder {
public:
    // `continuable` records the frontier so extend() has something to resume from. Pass false
    // for a one-shot job, which is what every caller that names no session is.
    explicit CpuEngineHolder(bool continuable,
                             unsigned threads = std::thread::hardware_concurrency());

    hypergraph::Hypergraph& hypergraph();
    hypergraph::ParallelEvolutionEngine& engine();
    const hypergraph::ParallelEvolutionEngine& engine() const;

    void extend(int steps, const std::vector<hgcommon::StateId>& only_from) override;

    std::vector<hgcommon::StateId> frontier() const override;

private:
    hypergraph::Hypergraph hg_;
    hypergraph::ParallelEvolutionEngine engine_;
};

// "StepStatistics" under full capture (ARCHITECTURE.md, "StepStatistics in the job system").
// Each step's input is gathered, and its record's jobs started (stats::StepStatisticsJobs), by a
// job on the engine's workers: during the run for each step the engine's depth join closes
// (attach), and at the reply for the rest (finish). A step's states are the hypergraph's
// per-step list when the run kept one, and otherwise one pass over every state at the reply;
// its branchial pairs are the event pairs recorded since the previous close, read through a
// cursor over the per-worker lists.
class StepStatisticsCollector {
public:
    StepStatisticsCollector(hypergraph::Hypergraph& hg, hypergraph::ParallelEvolutionEngine& engine,
                            uint32_t which, bool by_class, int steps);
    ~StepStatisticsCollector();
    StepStatisticsCollector(const StepStatisticsCollector&) = delete;
    StepStatisticsCollector& operator=(const StepStatisticsCollector&) = delete;

    // Before the evolution: close each step when no state can still arrive at it. With
    // "MaxStatesPerStep" (`step_cap`) step d's states are created by step d's selection, which
    // runs on depth d + 1, so depth d's report closes step d - 1. A run whose classes are
    // weighted by the multiplicity count (Hypergraph::quotient_multiplicity) closes nothing
    // during the run.
    void attach(bool step_cap);
    // After the evolution: closes every remaining step, waits for the jobs, returns the reply.
    // With `weighted`, each step's classes and events are its entry there, and only its
    // branchial input comes from the states.
    wxf::WXFValue finish(std::map<uint32_t, stats::StepInput>* weighted = nullptr);
    // Steps closed by finish rather than during the run.
    uint32_t closed_at_finish() const { return closed_at_finish_; }

private:
    void close_through(uint32_t last);
    void take_pairs();
    std::vector<uint32_t> states_at(uint32_t step) const;
    stats::StepInput gather(uint32_t step, std::vector<std::pair<int64_t, int64_t>> pairs);

    hypergraph::Hypergraph& hg_;
    hypergraph::ParallelEvolutionEngine& engine_;
    uint32_t which_;
    int steps_;
    stats::StepStatisticsJobs jobs_;
    bool attached_ = false;
    bool indexed_ = false;
    bool inactive_ = false;
    std::map<uint32_t, stats::StepInput>* weighted_ = nullptr;
    std::vector<std::vector<uint32_t>> by_step_;
    hypergraph::CausalGraph::BranchialCursor cursor_;
    std::map<uint32_t, std::vector<std::pair<int64_t, int64_t>>> pairs_;
    uint32_t next_ = 0;
    uint32_t closed_at_finish_ = 0;
    std::exception_ptr failed_;
};

}  // namespace ffi
}  // namespace HG_NAMESPACE