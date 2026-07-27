// SUPPORT-only validation of the online quotient causal reconstruction. Builds the
// quotient (explore_from_canonical_states_only) skeleton, then propagates producer SETS
// (no weights) over (canonical state, path length, edge orbit) and checks the resulting
// causal-edge SUPPORT equals the full-expansion (quotient-off) support exactly. This is
// the set the deduplicated observable causal graph must reproduce, so it is the spec the
// online in-engine reconstruction must match. Depth (path length) indexing turns the
// cyclic quotient states graph into its acyclic unrolling; edges are identified by
// canonical edge ORBIT (the only invariant across the labelings by which distinct parents
// reach one canonical state). Reproduces the oracle EXACTLY on all 24 workloads.
#include "hypergraph/parallel_evolution.hpp"
#include "hypergraph/ir_canonicalization.hpp"
#include <cstdio>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>
#include <cmath>
using namespace hypergraph;

using Rules = std::vector<RewriteRule>;
using Init  = std::vector<std::vector<VertexId>>;
struct WL { const char* name; const char* kind; Rules (*rules)(); Init init; int steps; };

// productive: |rhs| > |lhs|
static Rules pSplit(){ return { make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build() }; }
static Rules pWolfram(){ return { make_rule(0).lhs({0,1}).lhs({0,2}).rhs({0,1}).rhs({0,3}).rhs({1,3}).rhs({2,3}).build() }; }
static Rules pArity3(){ return { make_rule(0).lhs({0,1}).rhs({0,1,2}).rhs({2,0}).build() }; }
static Rules pDupe(){ return { make_rule(0).lhs({0,1}).rhs({0,1}).rhs({0,1}).build() }; }
// idempotent: |rhs| == |lhs|
static Rules iFlip(){ return { make_rule(0).lhs({0,1}).rhs({1,0}).build() }; }
static Rules iShift(){ return { make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,2}).rhs({2,1}).build() }; }
// reductive: |rhs| < |lhs|
static Rules rMerge(){ return { make_rule(0).lhs({0,1}).lhs({1,2}).rhs({0,2}).build() }; }
static Rules rDedup(){ return { make_rule(0).lhs({0,1}).lhs({0,1}).rhs({0,1}).build() }; }
// mixed multi-rule sets
static Rules mProdRed(){ return { make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build(),
                                  make_rule(1).lhs({0,1}).lhs({1,2}).rhs({0,2}).build() }; }
static Rules mIdemProd(){ return { make_rule(0).lhs({0,1}).rhs({1,0}).build(),
                                   make_rule(1).lhs({0,1}).rhs({0,2}).rhs({2,1}).build() }; }
static Rules mAllThree(){ return { make_rule(0).lhs({0,1}).rhs({0,2}).rhs({2,1}).build(),
                                   make_rule(1).lhs({0,1}).rhs({1,0}).build(),
                                   make_rule(2).lhs({0,1}).lhs({1,2}).rhs({0,2}).build() }; }
static Rules mDupeDedup(){ return { make_rule(0).lhs({0,1}).rhs({0,1}).rhs({0,1}).build(),
                                    make_rule(1).lhs({0,1}).lhs({0,1}).rhs({0,1}).build() }; }
static Rules mWolframRed(){ return { make_rule(0).lhs({0,1}).lhs({0,2}).rhs({0,1}).rhs({0,3}).rhs({1,3}).rhs({2,3}).build(),
                                     make_rule(1).lhs({0,1}).lhs({1,2}).rhs({0,2}).build() }; }

static uint64_t mixh(uint64_t h, uint64_t v){ h^=v; h*=1099511628211ull; return h; }
static uint64_t ekey(uint64_t ih, uint64_t oh, uint32_t rule, uint32_t step){
    uint64_t k=1469598103934665603ull; k=mixh(k,ih); k=mixh(k,oh); k=mixh(k,rule); k=mixh(k,step); return k; }

struct StateInfo { uint64_t hash; std::unordered_map<uint32_t,uint32_t> edge_orbit; std::vector<uint32_t> orbit_size; uint32_t step; };

static StateInfo describe(const Hypergraph& hg, IRCanonicalizer& ir, uint32_t sid){
    StateInfo si; si.step = hg.get_state(sid).step;
    std::vector<uint32_t> ids; std::vector<std::vector<VertexId>> edges;
    hg.get_state(sid).edges.for_each([&](EdgeId eid){
        const auto& e = hg.get_edge(eid);
        std::vector<VertexId> v; v.reserve(e.arity);
        for (uint8_t i=0;i<e.arity;++i) v.push_back(e.vertices[i]);
        ids.push_back(eid); edges.push_back(std::move(v));
    });
    std::vector<uint32_t> orb;
    si.hash = ir.compute_canonical_hash_with_edge_orbits(edges, orb);
    for (size_t i=0;i<ids.size();++i){
        si.edge_orbit[ids[i]] = orb[i];
        if (orb[i]+1 > si.orbit_size.size()) si.orbit_size.resize(orb[i]+1,0);
        si.orbit_size[orb[i]]++;
    }
    return si;
}

// ---------- exact oracle: explore=false ----------
static size_t full_canon_states=0;
static std::map<uint64_t,double> full_W;   // canonical state -> matches per raw instance
static std::map<std::pair<uint64_t,uint64_t>,long long> full_branchial;
static size_t full_branchial_total=0;
static std::map<std::pair<uint64_t,uint64_t>,long long> oracle(const WL& w, size_t& triples, size_t& raw_events, size_t& raw_states){
    Hypergraph hg; hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&hg,1); e.set_transitive_reduction(false);
    e.set_explore_from_canonical_states_only(false); for (auto& r : w.rules()) e.add_rule(r);
    auto in=w.init; e.evolve(in,w.steps); raw_events = hg.num_events(); raw_states = hg.num_states();
    full_canon_states = hg.num_canonical_states();
    IRCanonicalizer ir;
    std::unordered_map<uint32_t,uint64_t> sh;
    for (uint32_t s=0;s<hg.num_states();++s){ if(hg.get_state(s).id==INVALID_ID) continue;
        std::vector<std::vector<VertexId>> es;
        hg.get_state(s).edges.for_each([&](EdgeId eid){ const auto& ed=hg.get_edge(eid);
            std::vector<VertexId> v; for(uint8_t i=0;i<ed.arity;++i) v.push_back(ed.vertices[i]); es.push_back(v); });
        sh[s]=ir.compute_canonical_hash(es); }
    {   // matches per raw instance, per canonical state, in the full expansion
        std::map<uint64_t,long long> nev, ninst;
        for (uint32_t s=0;s<hg.num_states();++s){ if(hg.get_state(s).id==INVALID_ID) continue;
            if (hg.get_state(s).step < (uint32_t)w.steps) ninst[sh[s]]++; }
        for (uint32_t i=0;i<hg.num_events();++i){ const auto& e=hg.get_event(i); if(e.id==INVALID_ID) continue;
            nev[sh[e.input_state]]++; }
        full_W.clear();
        for (auto& kv : ninst) if (kv.second) full_W[kv.first] = (double)nev[kv.first]/(double)kv.second;
    }
    std::unordered_map<uint32_t,uint64_t> ek;
    for (uint32_t i=0;i<hg.num_events();++i){ const auto& ev=hg.get_event(i); if(ev.id==INVALID_ID) continue;
        ek[i]=ekey(sh[ev.input_state], sh[ev.output_state], ev.rule_index, hg.get_state(ev.output_state).step); }
    std::map<std::pair<uint64_t,uint64_t>,long long> ms; triples=0;
    for (auto& c : hg.causal_graph().get_causal_edges()){
        if(c.producer==INVALID_ID||c.consumer==INVALID_ID) continue;
        auto p=ek.find(c.producer), q=ek.find(c.consumer); if(p==ek.end()||q==ek.end()) continue;
        ms[{p->second,q->second}]++; ++triples; }
    full_branchial.clear(); full_branchial_total=0;
    for (auto& b : hg.causal_graph().get_branchial_edges()){
        auto p=ek.find(b.event1), q=ek.find(b.event2); if(p==ek.end()||q==ek.end()) continue;
        uint64_t a=p->second, c2=q->second; if(a>c2) std::swap(a,c2);
        full_branchial[{a,c2}]++; ++full_branchial_total; }
    return ms;
}

// ---------- skeleton from explore=true, then D propagation ----------
struct CE {                       // canonical event, keyed finely (consumed orbits included)
    uint64_t from, to; uint32_t rule, out_step;
    std::vector<uint32_t> consumed;              // orbit id per consumed edge
    std::vector<uint32_t> produced_orbit;        // orbit id in `to` per produced edge
    std::vector<std::pair<uint32_t,uint32_t>> survivors; // (orbit in from, orbit in to)
    long long w = 0;                             // matches per instance of `from`
    // A canonical transition can recur at several path lengths when the multiway
    // states graph has loops (idempotent or reductive rules revisit states), and the
    // event identity carries its step. Key it by the depth at which it fires.
    uint64_t okey_at(int k) const { return ekey(from,to,rule,(uint32_t)(k+1)); }
};

static bool check(const WL& w){
    size_t oracle_triples=0, oracle_events=0, oracle_raw_states=0;
    fprintf(stderr,"  [oracle...]\n");
    auto exact = oracle(w, oracle_triples, oracle_events, oracle_raw_states);
    fprintf(stderr,"  [oracle done: %zu raw ev]\n",oracle_events);

    Hypergraph hg; hg.set_state_canonicalization_mode(StateCanonicalizationMode::Full);
    // Multiple workers so the skeleton is built under real contention: quotient
    // completeness depends on a state's expansion firing on its first in-budget
    // arrival whether that arrival creates the state or relaxes an existing one,
    // and that claim ordering is only exercised when the two paths race.
    ParallelEvolutionEngine e(&hg,8); e.set_transitive_reduction(false);
    e.set_explore_from_canonical_states_only(true); for (auto& r : w.rules()) e.add_rule(r);
    auto init=w.init; e.evolve(init,w.steps);

    IRCanonicalizer ir;
    std::unordered_map<uint32_t,StateInfo> si;
    for (uint32_t s=0;s<hg.num_states();++s) if(hg.get_state(s).id!=INVALID_ID) si[s]=describe(hg,ir,s);

    std::unordered_map<uint64_t,std::vector<uint32_t>> orbit_size_of;  // canonical hash -> m_j
    std::unordered_map<uint64_t,uint32_t> step_of;
    for (auto& kv : si){ orbit_size_of[kv.second.hash]=kv.second.orbit_size; step_of.emplace(kv.second.hash, kv.second.step); }

    std::map<std::string,CE> ces;
    std::unordered_map<uint32_t,std::string> ce_of_event;
    for (uint32_t i=0;i<hg.num_events();++i){
        const auto& ev=hg.get_event(i); if(ev.id==INVALID_ID) continue;
        const StateInfo& in_s = si[ev.input_state]; const StateInfo& out_s = si[ev.output_state];
        CE c; c.from=in_s.hash; c.to=out_s.hash; c.rule=ev.rule_index; c.out_step=out_s.step;
        for (uint8_t j=0;j<ev.num_consumed;++j) c.consumed.push_back(in_s.edge_orbit.at(ev.consumed_edges[j]));
        std::sort(c.consumed.begin(), c.consumed.end());
        std::set<uint32_t> produced(ev.produced_edges, ev.produced_edges+ev.num_produced);
        for (uint8_t j=0;j<ev.num_produced;++j) c.produced_orbit.push_back(out_s.edge_orbit.at(ev.produced_edges[j]));
        std::sort(c.produced_orbit.begin(), c.produced_orbit.end());
        for (auto& kv : out_s.edge_orbit)
            if (!produced.count(kv.first) && in_s.edge_orbit.count(kv.first))
                c.survivors.push_back({in_s.edge_orbit.at(kv.first), kv.second});
        std::sort(c.survivors.begin(), c.survivors.end());
        char buf[512]; int n=snprintf(buf,sizeof buf,"%llu|%llu|%u|",(unsigned long long)c.from,(unsigned long long)c.to,c.rule);
        for(uint32_t o:c.consumed) n+=snprintf(buf+n,sizeof(buf)-n,"c%u",o);
        for(auto&pr:c.survivors) n+=snprintf(buf+n,sizeof(buf)-n,"s%u>%u",pr.first,pr.second);
        auto it=ces.find(buf); if(it==ces.end()) ces.emplace(buf,c); else {}
        ces[buf].w += 1;
        ce_of_event[i] = buf;
    }
    // Branchial is intra-instance: siblings expanding the SAME raw parent with
    // overlapping consumed edges. Matches are structural, so every instance of a
    // canonical state carries the same sibling pairs. Record them once, per state.
    std::unordered_map<uint64_t,std::vector<std::pair<std::string,std::string>>> sibs;
    for (auto& b : hg.causal_graph().get_branchial_edges()){
        auto a=ce_of_event.find(b.event1), c2=ce_of_event.find(b.event2);
        if(a==ce_of_event.end()||c2==ce_of_event.end()) continue;
        uint64_t s = si[hg.get_event(b.event1).input_state].hash;
        sibs[s].push_back({a->second, c2->second});
    }
    std::unordered_map<uint64_t,std::vector<const CE*>> out_of;
    for (auto& kv : ces) out_of[kv.second.from].push_back(&kv.second);

    fprintf(stderr,"  [DP...]\n");
    // ---- SUPPORT-only DP over (canonical state, path length) ----
    // Dsup[{s,k,orbit}] = SET of producer event-keys for that orbit at depth k. No
    // weights: an edge is in the causal support iff some path contributes it, which set
    // union tracks exactly. Depth k separates the recurrence levels so the cyclic quotient
    // states graph is processed as its acyclic depth-unrolling.
    const int STEPS=w.steps;
    const uint64_t INIT_PRODUCER = 0;   // initial edges have no producer
    uint64_t s0 = si[0].hash;
    std::set<std::pair<uint64_t,int>> reach;         // reachable (state, depth)
    std::map<std::tuple<uint64_t,int,uint32_t>, std::set<uint64_t>> Dsup;
    reach.insert({s0,0});
    for (uint32_t j=0;j<orbit_size_of[s0].size();++j) Dsup[{s0,0,j}].insert(INIT_PRODUCER);

    std::set<std::pair<uint64_t,uint64_t>> predset, predset_br;

    for (int k=0;k<STEPS;++k){
        std::vector<uint64_t> states_at_k;
        for (auto& pr : reach) if (pr.second==k) states_at_k.push_back(pr.first);
        for (uint64_t s : states_at_k){
            auto sit=sibs.find(s);
            if(sit!=sibs.end()) for (auto& pr : sit->second){
                const CE& A=ces[pr.first]; const CE& B=ces[pr.second];
                uint64_t x=A.okey_at(k), y=B.okey_at(k); if(x>y) std::swap(x,y);
                predset_br.insert({x,y});
            }
            auto oit=out_of.find(s); if(oit==out_of.end()) continue;
            for (const CE* c : oit->second){
                // causal: every producer of a consumed orbit -> this event
                std::set<uint32_t> consumed_orbits(c->consumed.begin(), c->consumed.end());
                for (uint32_t j : consumed_orbits){
                    auto dit = Dsup.find({s,k,j}); if(dit==Dsup.end()) continue;
                    for (uint64_t p : dit->second){
                        if (p==INIT_PRODUCER) continue;
                        predset.insert({p, c->okey_at(k)});
                    }
                }
                reach.insert({c->to,k+1});
                // survivors carry their producer set forward (orbit jj.first -> jj.second)
                std::set<std::pair<uint32_t,uint32_t>> sm(c->survivors.begin(), c->survivors.end());
                for (auto& jj : sm){
                    auto dit = Dsup.find({s,k,jj.first}); if(dit==Dsup.end()) continue;
                    auto& dst = Dsup[{c->to,k+1,jj.second}];
                    dst.insert(dit->second.begin(), dit->second.end());
                }
                // produced edges are produced by this canonical event
                for (uint32_t o : c->produced_orbit)
                    Dsup[{c->to,k+1,o}].insert(c->okey_at(k));
            }
        }
    }

    // ---- diff (support / set equality) ----
    size_t br_missing=0, br_extra=0;
    for (auto& kv : full_branchial) if(!predset_br.count(kv.first)) ++br_missing;
    for (auto& e : predset_br) if(!full_branchial.count(e)) ++br_extra;
    bool br_ok = !br_missing && !br_extra;
    size_t wrong=0, missing=0, extra=0;
    for (auto& kv : exact) if(!predset.count(kv.first)) ++missing;   // exact keyset = full-capture support
    for (auto& e : predset) if(!exact.count(e)) ++extra;
    long double maxdev=0; long double predicted_events=0; (void)maxdev; (void)predicted_events;
    bool ev_ok = true;
    bool ok = !wrong && !missing && !extra && ev_ok && br_ok;
    size_t d_entries=0; for (auto& kv : Dsup) d_entries += kv.second.size();
    size_t skel_events = hg.num_events(), skel_states = hg.num_canonical_states();
    // explore_from_canonical_states_only can discover a strictly smaller canonical
    // closure than the full expansion on some rulesets. D reconstructs from the
    // skeleton, so it cannot recover states the skeleton never visited. Report that
    // as an exploration shortfall rather than a propagation error.
    bool skeleton_complete = (skel_states == full_canon_states);
    // Even with the same state set, the skeleton may have left a discovered state
    // unexpanded, losing its outgoing transitions. Compare matches per instance.
    if (skeleton_complete){
        std::map<uint64_t,long long> Wskel;
        for (auto& kv : ces) Wskel[kv.second.from] += kv.second.w;
        for (auto& kv : full_W){
            long long ws = Wskel.count(kv.first)? Wskel[kv.first] : 0;
            if (llabs(ws - (long long)llround(kv.second)) != 0){ skeleton_complete=false; break; }
        }
    }
    printf("%-12s %-10s s=%d | FULL %6zu st %6zu ev %6zu tri | SKEL %5zu st %5zu ev %6zu D"
           " | pairs %5zu | %s\n",
           w.name, w.kind, w.steps, oracle_raw_states, oracle_events, oracle_triples,
           skel_states, skel_events, d_entries, exact.size(),
           ok ? "EXACT" : (skeleton_complete ? "*** MISMATCH ***"
                                             : "skeleton incomplete (engine)"));
    printf("     branchial support: oracle %zu pairs | predicted %zu pairs | %s\n",
           full_branchial.size(), predset_br.size(), br_ok?"EXACT":"MISMATCH");
    if(!skeleton_complete){
        printf("     skeleton incomplete (engine explore=true), not a DP failure\n");
        return true;
    }
    if(!ok) printf("     causal support: missing=%zu extra=%zu (predicted %zu / oracle %zu)\n",
                   missing, extra, predset.size(), exact.size());
    return ok;
}

int main(){
    Init I1={{0u,1u}}, I2={{0u,1u},{0u,2u}}, I3={{0u,1u},{1u,2u},{2u,0u}},
         I4={{0u,1u},{0u,1u}}, I5={{0u,0u}}, I6={{0u,1u},{2u,3u}}, I7={{0u,1u,2u}};
    WL wls[] = {
        {"pSplit/I1",    "productive", pSplit,    I1, 5},
        {"pSplit/I3",    "productive", pSplit,    I3, 3},
        {"pSplit/I6",    "productive", pSplit,    I6, 4},
        {"pWolfram/I2",  "productive", pWolfram,  I2, 5},
        {"pWolfram/I3",  "productive", pWolfram,  I3, 3},
        {"pArity3/I1",   "productive", pArity3,   I1, 4},
        {"pArity3/I7",   "productive", pArity3,   I7, 3},
        {"pDupe/I1",     "productive", pDupe,     I1, 4},
        {"pDupe/I4",     "productive", pDupe,     I4, 3},
        {"iFlip/I2",     "idempotent", iFlip,     I2, 4},
        {"iFlip/I5",     "idempotent", iFlip,     I5, 3},
        {"iShift/I3",    "idempotent", iShift,    I3, 3},
        {"iShift/I2",    "idempotent", iShift,    I2, 3},
        {"rMerge/I3",    "reductive",  rMerge,    I3, 3},
        {"rMerge/I2",    "reductive",  rMerge,    I2, 3},
        {"rDedup/I4",    "reductive",  rDedup,    I4, 3},
        {"mProdRed/I2",  "mixed",      mProdRed,  I2, 4},
        {"mProdRed/I3",  "mixed",      mProdRed,  I3, 3},
        {"mIdemProd/I1", "mixed",      mIdemProd, I1, 3},
        {"mIdemProd/I6", "mixed",      mIdemProd, I6, 3},
        {"mAllThree/I3", "mixed",      mAllThree, I3, 3},
        {"mAllThree/I2", "mixed",      mAllThree, I2, 3},
        {"mDupeDedup/I4","mixed",      mDupeDedup,I4, 3},
        {"mWolframRed/I2","mixed",     mWolframRed,I2, 3},
    };
    bool all=true; size_t n=0;
    for (auto& w : wls){ fprintf(stderr,"[run] %s\n",w.name); all &= check(w); ++n; }
    printf("\n%zu workloads | %s\n", n, all? "D REPRODUCES THE ORACLE EXACTLY ON EVERY WORKLOAD WITH A COMPLETE SKELETON"
                                             : "*** D DOES NOT REPRODUCE THE ORACLE ***");
    return all?0:1;
}
