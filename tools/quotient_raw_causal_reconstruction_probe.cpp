// Per-instance RAW causal reconstruction from the quotient skeleton.
//
// Tests the DECIDED design (docs/MEMORY_ARCHITECTURE_DESIGN.md §2b): the quotient retains the
// representative's FULL expansion (every match with its consumed/produced canonical edge slots)
// plus multiplicity; so enumerate raw INSTANCES (paths) over the depth-unrolled skeleton, each
// carrying a per-instance producer assignment indexed by canonical edge SLOT. Each match in the
// representative's expansion becomes one raw event in each instance, whose causal parents are that
// instance's producers of the consumed slots. This reconstructs the RAW causal graph -- O(raw
// output) to emit, but with NO pattern matching and NO canonicalization per raw state (the
// expensive work stays O(canonical)). Then transitively reduce the reconstructed RAW graph and
// compare against full-capture's TR.
//
// Slots: edges sorted by (canonical content class, raw edge id); slot = position. Isomorphic states
// have the same class multiset, so slot i carries the same class in every instance of a class --
// which is what lets a child's slots be read in the child class representative's frame. Within a
// class, occurrences are interchangeable in content but not in producer; the alignment is arbitrary,
// but since ALL matches fire, the emitted causal edge set is invariant under that choice (the
// Aut-ambiguity objection). That invariance is exactly what this probe tests.
#include "hypergraph/hypergraph.hpp"
#include "hypergraph/parallel_evolution.hpp"
#include "hypergraph/ir_canonicalization.hpp"
#include <algorithm>
#include <cstdio>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>
using namespace hypergraph;
using Rules = std::vector<RewriteRule>;
using Init  = std::vector<std::vector<VertexId>>;
struct WL { const char* name; const char* kind; Rules (*rules)(); Init init; int steps; };

// --- workloads: productive / idempotent / reductive / mixed, plus HIGH-AUTOMORPHISM cases ---
static Rules pSplit(){ return { make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build() }; }
static Rules pWolfram(){ return { make_rule(0).lhs({0,1}).lhs({0,2}).rhs({0,1}).rhs({0,3}).rhs({1,3}).rhs({2,3}).build() }; }
static Rules iFlip(){ return { make_rule(0).lhs({0,1}).rhs({1,0}).build() }; }
static Rules iShift(){ return { make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,2}).rhs({2,1}).build() }; }
static Rules rMerge(){ return { make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,2}).build() }; }
static Rules pDupe(){ return { make_rule(0).lhs({0,1}).rhs({0,1}).rhs({0,1}).build() }; }   // duplicate edges (same content class)
static Rules selfLoop(){ return { make_rule(0).lhs({0,0}).rhs({0,0}).rhs({0,0}).build() }; } // high-Aut self-loops
static Rules mixed1(){ return { make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build(),
                                make_rule(1).lhs({0,1}).rhs({1,0}).build(),
                                make_rule(2).lhs({0,1}).lhs({1,2}).rhs({0,2}).build() }; }
static Rules mixed2(){ return { make_rule(0).lhs({0,1}).rhs({1,0}).build(),
                                make_rule(1).lhs({0,1}).rhs({0,2}).rhs({2,1}).build() }; }
static Rules TC6(){ return { make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,1}).rhs({1,2}).rhs({1,3}).build() }; }

static uint64_t mixh(uint64_t h,uint64_t v){ h^=v; h*=1099511628211ull; return h; }
static uint64_t te_of(uint64_t from,uint64_t to,uint32_t rule){
    uint64_t k=1469598103934665603ull; k=mixh(k,from); k=mixh(k,to); k=mixh(k,rule); return k; }

// ---------- full-capture reference: TR-on pair count + esig-projected reduced set ----------
struct FullCap { size_t tr_pairs=0, raw_states=0, raw_events=0; std::set<std::pair<uint64_t,uint64_t>> proj; };
static FullCap fullcapture(const WL& w){
    FullCap fc;
    Hypergraph hg; hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&hg,1);
    e.set_transitive_reduction(true);
    e.set_explore_from_canonical_states_only(false);
    for(auto&r:w.rules()) e.add_rule(r);
    auto in=w.init; e.evolve(in,w.steps);
    fc.tr_pairs = hg.causal_graph().num_causal_event_pairs();
    fc.raw_states = hg.num_states(); fc.raw_events = hg.num_events();
    auto esig=[&](EventId ev){ const Event& x=hg.get_event(ev);
        return te_of(hg.get_or_compute_canonical_hash(x.input_state),
                     hg.get_or_compute_canonical_hash(x.output_state), x.rule_index); };
    for(const auto& c: hg.causal_graph().get_causal_edges()){
        if(c.producer==INVALID_ID||c.consumer==INVALID_ID) continue;
        fc.proj.insert({esig(c.producer), esig(c.consumer)}); }
    return fc;
}

// ---------- slots ----------
struct Slots { uint64_t hash=0; std::unordered_map<uint32_t,uint32_t> slot_of; uint32_t n=0; };
static Slots slots_of(const Hypergraph& hg, IRCanonicalizer& ir, uint32_t sid){
    Slots s;
    std::vector<std::vector<VertexId>> es; std::vector<uint32_t> ids;
    hg.get_state(sid).edges.for_each([&](EdgeId eid){
        const auto& e=hg.get_edge(eid);
        std::vector<VertexId> v; for(uint8_t i=0;i<e.arity;++i) v.push_back(e.vertices[i]);
        es.push_back(std::move(v)); ids.push_back(eid); });
    std::vector<uint32_t> cls;
    s.hash = ir.compute_canonical_hash_with_edge_map(es, cls);
    std::vector<std::pair<std::pair<uint32_t,uint32_t>,uint32_t>> order;  // ((class, raw id), raw id)
    for(size_t i=0;i<ids.size();++i) order.push_back({{cls.empty()?0:cls[i], ids[i]}, ids[i]});
    std::sort(order.begin(), order.end());
    for(size_t i=0;i<order.size();++i) s.slot_of[order[i].second] = (uint32_t)i;
    s.n = (uint32_t)order.size();
    return s;
}

// ---------- the reconstruction ----------
struct Match { uint32_t rule; uint64_t to_hash; uint32_t child_slots;
               std::vector<uint32_t> consumed;                         // parent slots
               std::vector<uint32_t> produced;                         // child slots
               std::vector<std::pair<uint32_t,uint32_t>> survivors; }; // (parent slot, child slot)

struct Recon { size_t instances=0, events=0, pairs_before_tr=0, tr_pairs=0; bool skeleton_ok=true;
               std::set<std::pair<uint64_t,uint64_t>> proj; };

static Recon reconstruct(const WL& w, size_t& skel_states, size_t& skel_events){
    Recon R;
    Hypergraph hg; hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&hg,8);
    e.set_transitive_reduction(false);
    e.set_explore_from_canonical_states_only(true);
    for(auto&r:w.rules()) e.add_rule(r);
    auto init=w.init; e.evolve(init,w.steps);
    skel_states = hg.num_canonical_states(); skel_events = hg.num_events();

    IRCanonicalizer ir;
    std::unordered_map<uint32_t,Slots> S;
    for(uint32_t s=0;s<hg.num_states();++s) if(hg.get_state(s).id!=INVALID_ID) S[s]=slots_of(hg,ir,s);

    // expansion per canonical class: the matches of the ONE expanded representative
    std::unordered_map<uint64_t,std::vector<Match>> expansion;
    std::unordered_map<uint64_t,uint32_t> rep_of, nslots_of;
    for(uint32_t i=0;i<hg.num_events();++i){
        const auto& ev=hg.get_event(i); if(ev.id==INVALID_ID) continue;
        const Slots& in_s=S[ev.input_state]; const Slots& out_s=S[ev.output_state];
        auto it=rep_of.find(in_s.hash);
        if(it==rep_of.end()){ rep_of[in_s.hash]=ev.input_state; nslots_of[in_s.hash]=in_s.n; }
        else if(it->second!=ev.input_state) continue;   // a second expansion of this class: skip
        Match m; m.rule=ev.rule_index; m.to_hash=out_s.hash; m.child_slots=out_s.n;
        for(uint8_t j=0;j<ev.num_consumed;++j) m.consumed.push_back(in_s.slot_of.at(ev.consumed_edges[j]));
        std::set<uint32_t> prod(ev.produced_edges, ev.produced_edges+ev.num_produced);
        for(uint8_t j=0;j<ev.num_produced;++j) m.produced.push_back(out_s.slot_of.at(ev.produced_edges[j]));
        for(auto&kv:out_s.slot_of)
            if(!prod.count(kv.first) && in_s.slot_of.count(kv.first))
                m.survivors.push_back({in_s.slot_of.at(kv.first), kv.second});
        expansion[in_s.hash].push_back(std::move(m));
    }

    // instance propagation over the depth-unrolled skeleton
    const EventId INIT = UINT32_MAX;
    struct Inst { uint64_t hash; std::vector<EventId> prod; };
    uint64_t h0 = S[0].hash;
    std::vector<Inst> cur; cur.push_back({h0, std::vector<EventId>(S[0].n, INIT)});
    R.instances = 1;
    std::vector<uint64_t> esig;                      // reconstructed event -> signature
    std::set<std::pair<uint32_t,uint32_t>> cedges;   // deduped (producer,consumer) raw pairs
    for(int k=0;k<w.steps;++k){
        std::vector<Inst> next;
        for(const Inst& I : cur){
            auto it=expansion.find(I.hash); if(it==expansion.end()) continue;
            for(const Match& m : it->second){
                uint32_t ev=(uint32_t)esig.size();
                esig.push_back(te_of(I.hash, m.to_hash, m.rule));
                for(uint32_t s : m.consumed){
                    if(s>=I.prod.size()) { R.skeleton_ok=false; continue; }
                    EventId p=I.prod[s]; if(p!=INIT) cedges.insert({p,ev}); }
                std::vector<EventId> cp(m.child_slots, INIT);
                for(auto&pr:m.survivors) if(pr.first<I.prod.size() && pr.second<cp.size()) cp[pr.second]=I.prod[pr.first];
                for(uint32_t s : m.produced) if(s<cp.size()) cp[s]=ev;
                next.push_back({m.to_hash, std::move(cp)});
            }
        }
        R.instances += next.size();
        cur.swap(next);
    }
    R.events = esig.size();
    R.pairs_before_tr = cedges.size();

    // transitive reduction of the reconstructed RAW graph (ids are topological: a parent is
    // always created at a strictly earlier depth than its consumer)
    std::map<uint32_t,std::vector<uint32_t>> adj;
    for(auto&e:cedges) adj[e.first].push_back(e.second);
    auto bypassed=[&](uint32_t u,uint32_t v)->bool{
        std::set<uint32_t> vis; std::vector<uint32_t> st;
        auto a=adj.find(u); if(a==adj.end()) return false;
        for(uint32_t x:a->second) if(x!=v && vis.insert(x).second) st.push_back(x);
        while(!st.empty()){ uint32_t x=st.back(); st.pop_back(); if(x==v) return true;
            auto b=adj.find(x); if(b==adj.end()) continue;
            for(uint32_t y:b->second) if(y<=v && vis.insert(y).second) st.push_back(y); }
        return false; };
    for(auto&e:cedges){
        if(!bypassed(e.first,e.second)){ R.tr_pairs++; R.proj.insert({esig[e.first], esig[e.second]}); } }
    return R;
}

static void check(const WL& w){
    FullCap fc = fullcapture(w);
    size_t skel_states=0, skel_events=0;
    Recon R = reconstruct(w, skel_states, skel_events);
    bool count_ok = (R.tr_pairs == fc.tr_pairs);
    bool proj_ok  = (R.proj == fc.proj);
    bool ev_ok    = (R.events == fc.raw_events);
    printf("%-9s %-10s s=%d | FULL raw_ev=%-6zu TRpairs=%-6zu | SKEL st=%-4zu ev=%-4zu"
           " | RECON inst=%-6zu ev=%-6zu TRpairs=%-6zu | events %s | TRpairs %s | proj %s\n",
           w.name, w.kind, w.steps, fc.raw_events, fc.tr_pairs, skel_states, skel_events,
           R.instances, R.events, R.tr_pairs,
           ev_ok?"OK":"MISMATCH", count_ok?"OK":"MISMATCH", proj_ok?"OK":"MISMATCH");
    if(!count_ok || !proj_ok){
        size_t miss=0,extra=0;
        for(auto&p:fc.proj) if(!R.proj.count(p)) miss++;
        for(auto&p:R.proj) if(!fc.proj.count(p)) extra++;
        printf("            proj missing=%zu extra=%zu%s\n", miss, extra,
               R.skeleton_ok?"":"  (slot bounds violated -- skeleton/slot mismatch)");
    }
}

int main(){
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
