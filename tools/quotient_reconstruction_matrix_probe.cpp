// Broad-coverage gate for the quotient raw reconstruction.
//
// Every earlier gate had a blind spot: depth 4, one thread, or one seed each independently hid the
// slot reference-frame defect. This one crosses the axes that matter:
//   rule type      productive / idempotent / reductive / duplicating / self-loop / arity-3
//   rule count     single and multi-rule combinations (incl. mixed types in one set)
//   initial state  single edge, path, cycle, disconnected, duplicate edges, self-loop, arity-3
//   depth          3..6 (the defect is invisible below 5)
//   threads        1, 2, 8
//   seed           two seeds (rule order changes which raw state represents a class)
//
// Two properties are checked per configuration:
//   EQUIVALENCE  reconstructed counts == full-capture counts (events / causal pairs / branchial)
//   DETERMINISM  the reconstructed causal edge multiset is identical across threads and seeds
// Counts alone are invariant under a wrong slot permutation, so the edge-identity fingerprint is
// what actually gates correctness here.
#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"
#include <cstdio>
#include <set>
#include <vector>
#include <algorithm>
using namespace hypergraph;
using Rules = std::vector<RewriteRule>;
using Init  = std::vector<std::vector<VertexId>>;

// ---- rule types ----
static Rules r_productive(){ return { make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build() }; }
static Rules r_wolfram()   { return { make_rule(0).lhs({0,1}).lhs({0,2}).rhs({0,1}).rhs({0,3}).rhs({1,3}).rhs({2,3}).build() }; }
static Rules r_idempotent(){ return { make_rule(0).lhs({0,1}).rhs({1,0}).build() }; }
static Rules r_shift()     { return { make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,2}).rhs({2,1}).build() }; }
static Rules r_reductive() { return { make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,2}).build() }; }
static Rules r_dedup()     { return { make_rule(0).lhs({0,1}).lhs({0,1}).rhs({0,1}).build() }; }
static Rules r_duplicate() { return { make_rule(0).lhs({0,1}).rhs({0,1}).rhs({0,1}).build() }; }
static Rules r_selfloop()  { return { make_rule(0).lhs({0,0}).rhs({0,0}).rhs({0,0}).build() }; }
static Rules r_arity3()    { return { make_rule(0).lhs({0,1}).rhs({0,1,2}).rhs({2,0}).build() }; }
// ---- combinations ----
static Rules c_prod_red()  { return { make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build(),
                                      make_rule(1).lhs({0,1}).lhs({1,2}).rhs({0,2}).build() }; }
static Rules c_idem_prod() { return { make_rule(0).lhs({0,1}).rhs({1,0}).build(),
                                      make_rule(1).lhs({0,1}).rhs({0,2}).rhs({2,1}).build() }; }
static Rules c_all_three() { return { make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build(),
                                      make_rule(1).lhs({0,1}).rhs({1,0}).build(),
                                      make_rule(2).lhs({0,1}).lhs({1,2}).rhs({0,2}).build() }; }
static Rules c_dup_dedup() { return { make_rule(0).lhs({0,1}).rhs({0,1}).rhs({0,1}).build(),
                                      make_rule(1).lhs({0,1}).lhs({0,1}).rhs({0,1}).build() }; }
static Rules c_wolf_red()  { return { make_rule(0).lhs({0,1}).lhs({0,2}).rhs({0,1}).rhs({0,3}).rhs({1,3}).rhs({2,3}).build(),
                                      make_rule(1).lhs({0,1}).lhs({1,2}).rhs({0,2}).build() }; }

struct RuleSet { const char* name; const char* kind; Rules (*fn)(); };
static const RuleSet RULESETS[] = {
    {"productive","single",r_productive}, {"wolfram","single",r_wolfram},
    {"idempotent","single",r_idempotent}, {"shift","single",r_shift},
    {"reductive","single",r_reductive},   {"dedup","single",r_dedup},
    {"duplicate","single",r_duplicate},   {"selfloop","single",r_selfloop},
    {"arity3","single",r_arity3},
    {"prod+red","combo",c_prod_red},      {"idem+prod","combo",c_idem_prod},
    {"all-three","combo",c_all_three},    {"dup+dedup","combo",c_dup_dedup},
    {"wolf+red","combo",c_wolf_red},
};
struct InitState { const char* name; Init init; };
static const InitState INITS[] = {
    {"edge",        {{0,1}}},
    {"path",        {{0,1},{1,2}}},
    {"cycle",       {{0,1},{1,2},{2,0}}},
    {"disconnected",{{0,1},{2,3}}},
    {"duplicate",   {{0,1},{0,1}}},
    {"selfloop",    {{0,0}}},
    {"fan",         {{0,1},{0,2}}},
    {"arity3",      {{0,1,2}}},
};

struct Counts { size_t events, pairs, branchial; uint64_t fp; };
static uint64_t fnv(uint64_t h, uint64_t x){ h^=x; h*=1099511628211ULL; return h; }

// An event's isomorphism-invariant identity, built exactly as Hypergraph::qc_apply builds
// qc_event_sig_: fnv over (input state canonical hash, output state canonical hash, rule).
// Raw event ids cannot be compared across the two arms -- full capture and the quotient
// reconstruction mint them independently -- so this is what makes an EDGE-IDENTITY comparison
// possible at all, rather than only a comparison of cardinalities.
static uint64_t event_sig(const Hypergraph& hg, EventId eid){
    const Event& ev = hg.get_event(eid);
    uint64_t s = 1469598103934665603ULL;
    s = fnv(s ^ 0, hg.get_state(ev.input_state).canonical_hash);
    s ^= hg.get_state(ev.output_state).canonical_hash; s *= 1099511628211ULL;
    s ^= ev.rule_index;                                s *= 1099511628211ULL;
    return s;
}

// Fold a set of (producer, consumer) signature pairs into one order-independent value.
static uint64_t fold_pairs(std::vector<uint64_t>& v){
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    uint64_t f = 1469598103934665603ULL;
    for(uint64_t x : v) f = fnv(f, x);
    return f;
}

static Counts full_capture(const Rules& rules, const Init& init, int steps){
    Hypergraph hg; hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&hg,1); e.set_transitive_reduction(false);
    e.set_explore_from_canonical_states_only(false);
    for(const auto& r:rules) e.add_rule(r);
    Init in=init; e.evolve(in,steps);
    // The reference arm gets the SAME fingerprint the reconstruction arm computes. Without
    // this the EQ check compared only counts, so two different causal relations of equal
    // cardinality passed as equivalent.
    std::vector<uint64_t> v;
    for(const auto& ce : hg.causal_graph().get_causal_edges()){
        if(ce.producer == INVALID_ID || ce.consumer == INVALID_ID) continue;
        v.push_back(fnv(fnv(0, event_sig(hg, ce.producer)), event_sig(hg, ce.consumer)));
    }
    return { hg.num_events(), hg.causal_graph().num_causal_event_pairs(),
             hg.causal_graph().num_branchial_edges(), fold_pairs(v) };
}
static Counts reconstructed(const Rules& rules, const Init& init, int steps, int threads, uint64_t seed){
    Hypergraph hg; hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&hg,threads); e.set_transitive_reduction(false);
    e.set_explore_from_canonical_states_only(true); e.set_random_seed(seed);
    hg.set_quotient_reconstruction(true);
    for(const auto& r:rules) e.add_rule(r);
    Init in=init; e.evolve(in,steps);
    std::vector<uint64_t> v;
    hg.for_each_reconstructed_causal(false,[&](uint64_t p,uint64_t c){ v.push_back(fnv(fnv(0,p),c)); });
    return { hg.num_reconstructed_events(), hg.num_reconstructed_causal_pairs(false),
             hg.num_reconstructed_branchial(), fold_pairs(v) };
}

int main(int argc, char** argv){
    const int max_depth = argc>1 ? atoi(argv[1]) : 6;
    size_t configs=0, eq_fail=0, det_fail=0;
    printf("quotient reconstruction matrix: rule types x combinations x initial states x depth x threads x seeds\n");
    printf("(EQ = counts AND causal edge identity match full-capture; DET = identical across threads and seeds)\n\n");
    for(const auto& rs : RULESETS){
        for(const auto& is : INITS){
            for(int steps=3; steps<=max_depth; ++steps){
                Counts f = full_capture(rs.fn(), is.init, steps);
                if(f.events == 0) continue;                 // rule does not fire here
                std::set<uint64_t> fps; std::set<size_t> ev,pr,br;
                for(int th : {1,2,8}) for(uint64_t seed : {0xABCDEFull, 0x1234ull}){
                    Counts r = reconstructed(rs.fn(), is.init, steps, th, seed);
                    fps.insert(r.fp); ev.insert(r.events); pr.insert(r.pairs); br.insert(r.branchial);
                }
                ++configs;
                const bool eq  = ev.size()==1 && *ev.begin()==f.events
                              && pr.size()==1 && *pr.begin()==f.pairs
                              && br.size()==1 && *br.begin()==f.branchial
                              && fps.size()==1 && *fps.begin()==f.fp;
                const bool det = fps.size()==1;
                if(!eq)  ++eq_fail;
                if(!det) ++det_fail;
                if(!eq || !det)
                    printf("  %-10s %-9s %-13s d=%d | full ev=%-6zu pr=%-6zu br=%-6zu | recon ev=%zu pr=%zu br=%zu"
                           " | EQ %s  DET %s (%zu fps)\n",
                           rs.name, rs.kind, is.name, steps, f.events, f.pairs, f.branchial,
                           *ev.begin(), *pr.begin(), *br.begin(), eq?"ok":"FAIL", det?"ok":"FAIL", fps.size());
            }
        }
    }
    printf("\n%zu configurations | equivalence failures: %zu | determinism failures: %zu\n",
           configs, eq_fail, det_fail);
    return (eq_fail || det_fail) ? 1 : 0;
}
