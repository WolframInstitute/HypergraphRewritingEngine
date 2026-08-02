#include "hypergraph/parallel_evolution.hpp"
#include <cstdio>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <set>
#include <functional>
using namespace hypergraph;
struct WL { const char* name; RewriteRule (*rule)(); std::vector<std::vector<VertexId>> init; int steps; };
static RewriteRule rA(){ return make_rule(0).lhs({0,1}).lhs({0,2}).rhs({0,1}).rhs({0,3}).rhs({1,3}).rhs({2,3}).build(); }
static RewriteRule rB(){ return make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build(); }
static RewriteRule rC(){ return make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,2}).rhs({2,1}).rhs({1,0}).build(); }

static std::set<std::pair<uint32_t,uint32_t>> pairs_of(const std::vector<CausalEdge>& e){
    std::set<std::pair<uint32_t,uint32_t>> p; for(auto&c:e) p.insert({c.producer,c.consumer}); return p; }

// offline minimal TR (unique pairs) of a raw pair set
static std::set<std::pair<uint32_t,uint32_t>> offline_tr(const std::set<std::pair<uint32_t,uint32_t>>& pairs){
    std::unordered_map<uint32_t,std::unordered_set<uint32_t>> succ;
    for(auto&pc:pairs) succ[pc.first].insert(pc.second);
    std::unordered_map<uint32_t,std::unordered_set<uint32_t>> reach; std::unordered_set<uint32_t> done;
    std::function<std::unordered_set<uint32_t>&(uint32_t)> R=[&](uint32_t u)->std::unordered_set<uint32_t>&{
        auto& s=reach[u]; if(done.count(u)) return s; done.insert(u);
        auto it=succ.find(u); if(it!=succ.end()) for(uint32_t w:it->second){ s.insert(w); auto& rw=R(w); s.insert(rw.begin(),rw.end()); }
        return s; };
    std::set<std::pair<uint32_t,uint32_t>> kept;
    for(auto&pc:pairs){ bool red=false; for(uint32_t w:succ[pc.first]){ if(w==pc.second) continue; if(R(w).count(pc.second)){red=true;break;} } if(!red) kept.insert(pc); }
    return kept;
}
static std::set<std::pair<uint32_t,uint32_t>> run(const WL& w, bool tr, int th,
                                                  bool automatic_identity=false){
    Hypergraph hg; hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    // EVENT_SIG_AUTOMATIC makes configure_identity_and_quotient turn the quotient
    // reconstruction on even under full-capture exploration.
    if (automatic_identity) hg.set_event_signature_keys(hgcommon::EVENT_SIG_AUTOMATIC);
    ParallelEvolutionEngine e(&hg, th); e.set_transitive_reduction(tr);
    e.set_explore_from_canonical_states_only(false);
    e.add_rule(w.rule()); auto in=w.init; e.evolve(in,w.steps);
    return pairs_of(hg.causal_graph().get_causal_edges());
}

// The pairs in the order the engine emitted them, first occurrence wins.
static std::vector<std::pair<uint32_t,uint32_t>> ordered_pairs(const WL& w, int th,
                                                               bool automatic_identity){
    Hypergraph hg; hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    if (automatic_identity) hg.set_event_signature_keys(hgcommon::EVENT_SIG_AUTOMATIC);
    ParallelEvolutionEngine e(&hg, th); e.set_transitive_reduction(false);
    e.set_explore_from_canonical_states_only(false);
    e.add_rule(w.rule()); auto in=w.init; e.evolve(in,w.steps);
    std::set<std::pair<uint32_t,uint32_t>> seen;
    std::vector<std::pair<uint32_t,uint32_t>> out;
    for (const auto& c : hg.causal_graph().get_causal_edges()) {
        auto p = std::make_pair(c.producer, c.consumer);
        if (seen.insert(p).second) out.push_back(p);
    }
    return out;
}

// The engine's rule, replayed offline: keep a pair unless the ALREADY-KEPT set has a path.
static std::set<std::pair<uint32_t,uint32_t>> replay_online_tr(
        const std::vector<std::pair<uint32_t,uint32_t>>& order){
    std::unordered_map<uint32_t,std::unordered_set<uint32_t>> kept_succ;
    std::set<std::pair<uint32_t,uint32_t>> kept;
    for (auto& pc : order) {
        // Reachable over KEPT edges only, exactly what qc_preds_/preds_ holds.
        bool redundant = false;
        std::vector<uint32_t> stack{pc.first};
        std::unordered_set<uint32_t> seen{pc.first};
        while (!stack.empty() && !redundant) {
            uint32_t x = stack.back(); stack.pop_back();
            auto it = kept_succ.find(x);
            if (it == kept_succ.end()) continue;
            for (uint32_t y : it->second) {
                if (y == pc.second) { redundant = true; break; }
                if (seen.insert(y).second) stack.push_back(y);
            }
        }
        if (!redundant) { kept.insert(pc); kept_succ[pc.first].insert(pc.second); }
    }
    return kept;
}
int main(){
    WL wls[] = { {"wolfram5", rA, {{0u,1u},{0u,2u}}, 5},
                 {"chain6",   rB, {{0u,1u}}, 6},
                 {"tri4",     rC, {{0u,1u},{1u,2u}}, 4} };
    bool allok=true;

    // ARM 1: Automatic identity, single-threaded. This is the reconstruction serving the causal
    // graph WITH the reduction enabled -- reachable through set_event_signature_keys, because
    // guard_quotient_transitive_reduction only disables TR under quotient EXPLORATION.
    printf("== Automatic identity (quotient reconstruction) + TR, th=1 ==\n");
    bool auto_ok = true;
    for (auto& w : wls) {
        auto order = ordered_pairs(w, 1, /*automatic_identity=*/true);
        std::set<std::pair<uint32_t,uint32_t>> raw(order.begin(), order.end());
        auto want   = offline_tr(raw);
        auto got    = run(w, true, 1, /*automatic_identity=*/true);
        auto replay = replay_online_tr(order);
        const bool ok = (got.size() == want.size());
        auto_ok &= ok;
        printf("%-9s raw=%4zu  engineTR=%4zu  minimalTR=%4zu  replayOnlineTR=%4zu  %s\n",
               w.name, raw.size(), got.size(), want.size(), replay.size(),
               ok ? "EXACT" : "*** NOT MINIMAL ***");
        if (!ok) {
            // The replay models "online TR over the emission order, reachability across KEPT
            // edges". It is a MODEL, and it is only evidence about the engine when it agrees
            // with it. It currently keeps MORE than the engine does, so the engine is finding
            // redundancy the model does not -- the model is incomplete and names no cause.
            // Reported as three numbers, not as a verdict.
            printf("            replay=%zu vs engine=%zu: the replay model does NOT reproduce "
                   "the engine, so it identifies no cause here\n", replay.size(), got.size());
        }
    }
    printf("%s\n\n", auto_ok ? "Automatic arm EXACT"
                              : "*** Automatic arm NOT MINIMAL: reconstruction + TR is wrong ***");
    allok &= auto_ok;

    printf("== default identity (full-capture rendezvous) ==\n");
    for (auto& w : wls) {
        for (int th : {1,2,4,8,16}) {
            // Event ids differ between runs, so compare SIZES: the causal graphs are
            // isomorphic and TR size is an isomorphism invariant.
            auto raw  = run(w,false,th);
            auto want = offline_tr(raw);
            auto got  = run(w,true,th);
            bool ok = (got.size()==want.size());
            allok &= ok;
            printf("%-9s th=%2d  raw=%4zu  engineTR=%4zu  minimalTR=%4zu  %s\n",
                   w.name, th, raw.size(), got.size(), want.size(), ok?"EXACT":"*** NOT MINIMAL ***");
        }
    }
    printf("\n%s\n", allok? "ALL EXACT: no redundant edge slipped through, at any thread count"
                          : "SOME RUNS NOT MINIMAL");
    return allok?0:1;
}
