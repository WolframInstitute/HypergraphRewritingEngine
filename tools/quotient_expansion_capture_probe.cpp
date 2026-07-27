// S1 GATE: does the ENGINE's captured slot-indexed expansion reproduce full-capture?
//
// The verified reconstruction (tools/quotient_raw_causal_reconstruction_probe.cpp) computed the
// representative's expansion itself, outside the engine. This probe replays the SAME per-instance
// algorithm but drives it entirely from what the engine captured online
// (Hypergraph::for_each_expansion_match), so a mismatch localises to the capture rather than to the
// algorithm. Checks raw event count, causal-pair count and the projected reduced edge set against
// full-capture.
#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"
#include <algorithm>
#include <cstdio>
#include <map>
#include <set>
#include <vector>
using namespace hypergraph;
using Rules = std::vector<RewriteRule>;
using Init  = std::vector<std::vector<VertexId>>;
struct WL { const char* name; const char* kind; Rules (*rules)(); Init init; int steps; };

static Rules pSplit(){ return { make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build() }; }
static Rules pWolfram(){ return { make_rule(0).lhs({0,1}).lhs({0,2}).rhs({0,1}).rhs({0,3}).rhs({1,3}).rhs({2,3}).build() }; }
static Rules pDupe(){ return { make_rule(0).lhs({0,1}).rhs({0,1}).rhs({0,1}).build() }; }
static Rules selfLoop(){ return { make_rule(0).lhs({0,0}).rhs({0,0}).rhs({0,0}).build() }; }
static Rules iFlip(){ return { make_rule(0).lhs({0,1}).rhs({1,0}).build() }; }
static Rules iShift(){ return { make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,2}).rhs({2,1}).build() }; }
static Rules rMerge(){ return { make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,2}).build() }; }
static Rules mixed1(){ return { make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build(),
                                make_rule(1).lhs({0,1}).rhs({1,0}).build(),
                                make_rule(2).lhs({0,1}).lhs({1,2}).rhs({0,2}).build() }; }
static Rules mixed2(){ return { make_rule(0).lhs({0,1}).rhs({1,0}).build(),
                                make_rule(1).lhs({0,1}).rhs({0,2}).rhs({2,1}).build() }; }
static Rules TC6(){ return { make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,1}).rhs({1,2}).rhs({1,3}).build() }; }

static uint64_t mixh(uint64_t h,uint64_t v){ h^=v; h*=1099511628211ull; return h; }
static uint64_t te_of(uint64_t from,uint64_t to,uint32_t rule){
    uint64_t k=1469598103934665603ull; k=mixh(k,from); k=mixh(k,to); k=mixh(k,rule); return k; }

struct FullCap { size_t raw_events=0, tr_pairs=0; std::set<std::pair<uint64_t,uint64_t>> proj; };
static FullCap fullcapture(const WL& w){
    FullCap fc;
    Hypergraph hg; hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&hg,1);
    e.set_transitive_reduction(true);
    e.set_explore_from_canonical_states_only(false);
    for(auto&r:w.rules()) e.add_rule(r);
    auto in=w.init; e.evolve(in,w.steps);
    fc.raw_events = hg.num_events();
    fc.tr_pairs   = hg.causal_graph().num_causal_event_pairs();
    auto esig=[&](EventId ev){ const Event& x=hg.get_event(ev);
        return te_of(hg.get_or_compute_canonical_hash(x.input_state),
                     hg.get_or_compute_canonical_hash(x.output_state), x.rule_index); };
    for(const auto& c: hg.causal_graph().get_causal_edges()){
        if(c.producer==INVALID_ID||c.consumer==INVALID_ID) continue;
        fc.proj.insert({esig(c.producer), esig(c.consumer)}); }
    return fc;
}

static void check(const WL& w){
    FullCap fc = fullcapture(w);

    Hypergraph hg; hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&hg,8);
    e.set_transitive_reduction(false);
    e.set_explore_from_canonical_states_only(true);
    for(auto&r:w.rules()) e.add_rule(r);
    auto init=w.init; e.evolve(init,w.steps);

    // Per-instance replay, driven ONLY by the engine's captured expansion.
    const EventId INIT = UINT32_MAX;
    struct Inst { uint64_t hash; std::vector<EventId> prod; };
    const uint64_t h0 = hg.get_state(0).canonical_hash;
    const EdgeOrbitTable* o0 = hg.state_orbits(0);
    if(!o0){ printf("%-9s  NO ORBIT TABLE FOR INITIAL STATE\n", w.name); return; }

    std::vector<Inst> cur; cur.push_back({h0, std::vector<EventId>(o0->n, INIT)});
    std::vector<uint64_t> esig;
    std::set<std::pair<uint32_t,uint32_t>> cedges;
    bool bounds_ok = true;
    for(int k=0;k<w.steps;++k){
        std::vector<Inst> next;
        for(const Inst& I : cur){
            hg.for_each_expansion_match(I.hash, [&](const SlotMatch& m){
                if(m.from_slots != I.prod.size()) { bounds_ok = false; return; }
                uint32_t ev=(uint32_t)esig.size();
                esig.push_back(te_of(I.hash, m.to_hash, m.rule));
                for(uint32_t i=0;i<m.num_consumed;++i){
                    uint32_t s=m.consumed_slots[i];
                    if(s>=I.prod.size()){ bounds_ok=false; continue; }
                    EventId p=I.prod[s]; if(p!=INIT) cedges.insert({p,ev});
                }
                std::vector<EventId> cp(m.to_slots, INIT);
                for(uint32_t i=0;i<m.num_survivors;++i){
                    uint32_t a=m.surv_from_slot[i], b=m.surv_to_slot[i];
                    if(a<I.prod.size() && b<cp.size()) cp[b]=I.prod[a];
                }
                for(uint32_t i=0;i<m.num_produced;++i){
                    uint32_t s=m.produced_slots[i];
                    if(s<cp.size()) cp[s]=ev;
                }
                next.push_back({m.to_hash, std::move(cp)});
            });
        }
        cur.swap(next);
    }

    // TR over the reconstructed raw graph (ids are topological by construction).
    std::map<uint32_t,std::vector<uint32_t>> adj;
    for(auto&e2:cedges) adj[e2.first].push_back(e2.second);
    auto bypassed=[&](uint32_t u,uint32_t v)->bool{
        std::set<uint32_t> vis; std::vector<uint32_t> st;
        auto a=adj.find(u); if(a==adj.end()) return false;
        for(uint32_t x:a->second) if(x!=v && vis.insert(x).second) st.push_back(x);
        while(!st.empty()){ uint32_t x=st.back(); st.pop_back(); if(x==v) return true;
            auto b=adj.find(x); if(b==adj.end()) continue;
            for(uint32_t y:b->second) if(y<=v && vis.insert(y).second) st.push_back(y); }
        return false; };
    size_t tr_pairs=0; std::set<std::pair<uint64_t,uint64_t>> proj;
    for(auto&e2:cedges) if(!bypassed(e2.first,e2.second)){ tr_pairs++; proj.insert({esig[e2.first], esig[e2.second]}); }

    const bool ev_ok = (esig.size()==fc.raw_events);
    const bool tr_ok = (tr_pairs==fc.tr_pairs);
    const bool pj_ok = (proj==fc.proj);
    printf("%-9s %-10s s=%d | events %6zu/%-6zu %s | TRpairs %6zu/%-6zu %s | proj %s%s\n",
           w.name, w.kind, w.steps,
           fc.raw_events, esig.size(), ev_ok?"OK":"**",
           fc.tr_pairs,   tr_pairs,    tr_ok?"OK":"**",
           pj_ok?"OK":"**", bounds_ok?"":"  (SLOT BOUNDS VIOLATED)");
}

int main(){
    printf("S1 gate: full-capture / engine-captured-expansion replay\n\n");
    WL wls[] = {
        {"pSplit",   "productive", pSplit,   {{0,1}}, 4},
        {"pWolfram", "productive", pWolfram, {{0,1},{0,2}}, 4},
        {"pDupe",    "dup-edges",  pDupe,    {{0,1}}, 4},
        {"selfLoop", "high-Aut",   selfLoop, {{0,0}}, 4},
        {"iFlip",    "idempotent", iFlip,    {{0,1}}, 4},
        {"iShift",   "idempotent", iShift,   {{0,1},{1,2}}, 4},
        {"rMerge",   "reductive",  rMerge,   {{0,1},{1,2},{2,0}}, 3},
        {"mixed1",   "mixed",      mixed1,   {{0,1}}, 4},
        {"mixed2",   "mixed",      mixed2,   {{0,1}}, 4},
        {"TC6",      "productive", TC6,      {{1,2},{2,3}}, 4},
    };
    for(auto& w : wls) check(w);
    return 0;
}
