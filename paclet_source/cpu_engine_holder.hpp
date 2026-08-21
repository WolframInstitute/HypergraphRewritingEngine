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

#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"

#include <cstddef>
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

}  // namespace ffi
}  // namespace HG_NAMESPACE