// GenMC harness: the drain of a state's match join reads the minimum own rank and the spawned
// mark of every task that completed before it.
//
// WHAT IS BEING PROVED. Under sampling (transition rate below 1) each own-found match folds its
// seeded rank into MatchJoin::own_min_key with a relaxed min-CAS loop, and a match whose draw
// passes sets own_spawned (transition_survives_spined, parallel_evolution.cpp). Every match task
// then books its completion (note_match_task_done). The task whose completion balances the
// pushed and completed counters is the drainer, and spine_at_drain reads both fields: it forces
// the minimum-rank transition through when no own draw passed. The drain is correct only if those
// reads see the fold and the mark of every task, and nothing orders them except the join's
// counters. Property, under RC11:
//   ONE DRAIN:  exactly one completion balances the counters, per join.
//   MINIMUM:    on a join where no draw passed, the drainer reads the minimum of the folded ranks.
//   SPAWNED:    on a join where a draw passed, the drainer reads the mark and forces nothing.
//
// This runs the engine's own MatchJoin member functions (hypergraph/match_join.hpp):
// note_pushed, fold_own_rank, mark_own_spawned, note_completed, drains_at and spine_rank_at_drain,
// in the order the engine calls them.
//
// WHAT IS BOUNDED. One state's matching as a tree of three tasks: main books and starts the root
// task, the root books and starts two child tasks and then does its own work, so the root's
// completion races its children's and any of the three can drain. Two joins, A and B, run by the
// same three tasks with ranks 5, 3 and 7: on A no draw passes; on B the rank-7 task's draw passes.
//
// CALIBRATION. -DHG_CALIBRATE_MATCH_JOIN_RELAXED_COMPLETION books completions relaxed: the drainer
// is then not ordered after the other tasks' folds and can read a larger rank or no mark.
// -DHG_CALIBRATE_SPINE_FOLD_BY_STORE folds by load-compare-store: two folds that both read the
// initial value can leave the larger rank as the minimum.
//
// GENMC-ARGS: --disable-estimation
// GENMC-EXPECT: pass
// GENMC-CALIBRATE: -DHG_CALIBRATE_MATCH_JOIN_RELAXED_COMPLETION
// GENMC-CALIBRATE: -DHG_CALIBRATE_SPINE_FOLD_BY_STORE
//
// Build/run: verification/genmc/run.sh spine_min_rank

#include <pthread.h>
#include <cassert>
#include <cstdint>
#include <atomic>

#include "genmc_support.hpp"
#include "hypergraph/match_join.hpp"

namespace {

using hg::engine::MatchJoin;

MatchJoin* g_a;   // no draw passes
MatchJoin* g_b;   // the rank-7 task's draw passes
std::atomic<int> g_drains_a{0}, g_drains_b{0};
std::atomic<uint64_t> g_want_a{0}, g_want_b{0};
pthread_t g_kids[2];

struct Task { uint64_t rank; bool passes_b; };
Task g_tasks[3] = {{5, false}, {3, false}, {7, true}};

// One match task's spine work and completion on one join: transition_survives_spined, then
// note_match_task_done and, if this task drains, spine_at_drain's read.
void run_on(MatchJoin* j, uint64_t rank, bool passes, std::atomic<int>& drains,
            std::atomic<uint64_t>& want) {
    j->fold_own_rank(rank);
    if (passes) j->mark_own_spawned();
    const size_t done = j->note_completed();
    if (j->drains_at(done)) {
        drains.fetch_add(1, std::memory_order_relaxed);
        want.store(j->spine_rank_at_drain(), std::memory_order_relaxed);
    }
}

void run_task(const Task& t) {
    run_on(g_a, t.rank, false, g_drains_a, g_want_a);
    run_on(g_b, t.rank, t.passes_b, g_drains_b, g_want_b);
}

void* child(void* arg) {
    run_task(g_tasks[reinterpret_cast<uintptr_t>(arg)]);
    return nullptr;
}

// The root task books both children on both joins before they can run, then does its own work.
void* root(void*) {
    g_a->note_pushed(); g_a->note_pushed();
    g_b->note_pushed(); g_b->note_pushed();
    pthread_create(&g_kids[0], nullptr, child, reinterpret_cast<void*>(uintptr_t{1}));
    pthread_create(&g_kids[1], nullptr, child, reinterpret_cast<void*>(uintptr_t{2}));
    run_task(g_tasks[0]);
    return nullptr;
}

}  // namespace

int main() {
    MatchJoin a, b;
    g_a = &a;
    g_b = &b;
    a.note_pushed();
    b.note_pushed();

    pthread_t t;
    pthread_create(&t, nullptr, root, nullptr);
    pthread_join(t, nullptr);
    pthread_join(g_kids[0], nullptr);
    pthread_join(g_kids[1], nullptr);

    assert(g_drains_a.load(std::memory_order_relaxed) == 1 && "join A drained other than once");
    assert(g_drains_b.load(std::memory_order_relaxed) == 1 && "join B drained other than once");
    assert(g_want_a.load(std::memory_order_relaxed) == 3 && "the drain missed the minimum rank");
    assert(g_want_b.load(std::memory_order_relaxed) == MatchJoin::kNoSpine &&
           "the drain missed a passed draw");
    return 0;
}
