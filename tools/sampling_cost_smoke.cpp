// sampling_cost_smoke.cpp — bisect which arm of sampling_cost_probe does not return.
//
// One configuration per invocation, unbuffered, so a hang is attributable rather than inferred
// from an empty output file.
//
// Run: /tmp/smoke <arm: off|cap|rate|ratenofwd|fwdon|fwdoff> <rule> <edges> <steps> <threads> <k> <canon: full|automatic|none>
//
// `off` and the canon mode exist to answer what sampling COSTS, which is a different question
// from what it PRESERVES. The sampling draw is keyed on canonical_transition_key, whose first
// act is ensure_state_edge_ranks -- an individualization-refinement pass. Under Full that pass
// happens anyway for dedup, so the key is nearly free. Under Automatic (content hash) and None
// (a per-state counter) NO IR pass otherwise runs, so switching sampling on ADDS the engine's
// most expensive operation per state. Whether the pruning repays it is a crossover, and this is
// the arm pair that locates it: `off` against `rate` at the same canon mode.

#include "hypergraph/parallel_evolution.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#include <unistd.h>
#endif

using namespace hypergraph;

// THE RUN CAPS ITS OWN ADDRESS SPACE. STANDING.
//
// Deep configurations of the disconnected shapes take gigabytes each. A batch of them was once
// launched on a 19 GB workstation shared with a dozen other sessions and no limit; it drove the
// machine into swap and had to be killed by hand, repeatedly, while every other project on the
// box waited. Capping in the DRIVING SCRIPT is not enough, because the batch that did it invoked
// this binary directly. So the binary refuses to be unbounded whoever starts it.
//
// The cap is a share of what the machine has FREE at start-up, not a constant: the same binary
// runs on a 19 GB desktop and a 503 GB box. HG_MEM_CAP_MB overrides it for a configuration known
// to need more, and 0 disables it for a machine where the caller has arranged its own limit.
//
// Exceeding it is a clean std::bad_alloc the engine reports as a truncated graph -- a RESULT.
// Paging the machine into swap is not a result; it is an outage for everyone else on the host.
static void cap_address_space() {
#if defined(__unix__) && !defined(__APPLE__)
    if (const char* env = std::getenv("HG_MEM_CAP_MB")) {
        const long mb = std::atol(env);
        if (mb <= 0) return;                       // explicit opt-out
        rlimit rl{static_cast<rlim_t>(mb) << 20, static_cast<rlim_t>(mb) << 20};
        setrlimit(RLIMIT_AS, &rl);
        return;
    }
    const long pages = sysconf(_SC_AVPHYS_PAGES), page = sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page <= 0) return;
    // Six tenths of free memory, so the rest of the machine keeps a working margin.
    const auto cap = static_cast<rlim_t>(static_cast<double>(pages) * page * 0.6);
    rlimit rl{cap, cap};
    setrlimit(RLIMIT_AS, &rl);
    std::printf("mem cap: %.1f GB of %.1f GB free (HG_MEM_CAP_MB overrides, 0 disables)\n",
                cap / 1073741824.0, static_cast<double>(pages) * page / 1073741824.0);
#endif
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    cap_address_space();

    const std::string arm  = (argc > 1) ? argv[1] : "cap";
    const std::string rule = (argc > 2) ? argv[2] : "growth";
    const size_t edges     = (argc > 3) ? std::stoul(argv[3]) : 1;
    const size_t steps     = (argc > 4) ? std::stoul(argv[4]) : 7;
    const size_t threads   = (argc > 5) ? std::stoul(argv[5]) : 4;
    const size_t k         = (argc > 6) ? std::stoul(argv[6]) : 4;
    const std::string canon = (argc > 7) ? argv[7] : "full";

    std::printf("arm=%s rule=%s edges=%zu steps=%zu threads=%zu k=%zu canon=%s\n",
                arm.c_str(), rule.c_str(), edges, steps, threads, k, canon.c_str());

    // LHS SIZE IS THE SWEPT VARIABLE, so the shapes have to differ in that and in nothing else.
    // chain_rule(n) is an n-edge CONNECTED left-hand side -- edges (0,1),(1,2),...,(n-1,n),
    // consecutive edges sharing a vertex -- rewritten by subdividing the LAST edge through one
    // fresh vertex. Every n therefore has the same net effect on the state (+1 edge) and the
    // same right-hand side shape, and the only thing that changes with n is how much of the
    // state a match must bind. chain_rule(2) IS the two-edge rule the published sweep uses;
    // it is generated here rather than written out again so the family cannot drift apart.
    auto chain_rule = [](size_t n) {
        auto b = make_rule(0);
        for (size_t i = 0; i < n; ++i)
            b.lhs({static_cast<uint8_t>(i), static_cast<uint8_t>(i + 1)});
        for (size_t i = 0; i + 1 < n; ++i)
            b.rhs({static_cast<uint8_t>(i), static_cast<uint8_t>(i + 1)});
        const auto fresh = static_cast<uint8_t>(n + 1);
        b.rhs({static_cast<uint8_t>(n - 1), fresh});
        b.rhs({fresh, static_cast<uint8_t>(n)});
        return b.build();
    };

    // THE LEFT-HAND SIDE HAS THREE INDEPENDENT PROPERTIES, AND A SWEEP OVER ONE OF THEM SAYS
    // NOTHING ABOUT THE OTHER TWO. They are: how MANY edges it has, what ARITY those edges are,
    // and how they are CONNECTED to each other. A chain of binary edges varies only the first.
    //
    //   shape        how the n edges share vertices
    //     chain      e_i covers vertices i..i+a-1, so consecutive edges overlap in a-1
    //     star       every edge contains vertex 0 and is otherwise fresh -- one join point of
    //                degree n, against the chain's n-1 join points of degree 2
    //     cycle      a chain whose last edge closes back onto vertex 0, so the LHS has no
    //                endpoint to anchor a match at and every vertex has the same degree
    //     tree       edge i attaches to vertex i/2, giving a branching LHS that is neither a
    //                path nor a single hub
    //     disc       n components sharing no vertex, so matching cannot join at all and
    //                enumerates a cartesian product
    //   arity        2, 3 or 4 vertices per edge; "mixed" alternates arity 2 and 3, which is
    //                the case a uniform-arity sweep cannot produce
    //
    // Named <shape><n>a<arity>, e.g. star3a2, cycle4a2, chain2a3, tree4a2, mixed3.
    //
    // ONE RIGHT-HAND SIDE CONVENTION ACROSS THE WHOLE FAMILY: keep every matched edge and append
    // ONE fresh edge of the same arity hanging off the LHS's highest vertex. Net effect is +1
    // edge for every shape, arity and size, so a difference between two runs is a difference in
    // the LEFT-hand side and not in how much the rule grows the state.
    auto parse_family = [](const std::string& s, size_t& n, size_t& arity,
                           std::string& shape) -> bool {
        static const char* kShapes[] = {"chain", "star", "cycle", "tree", "disc", "mixed"};
        for (const char* sh : kShapes) {
            const std::string p(sh);
            if (s.rfind(p, 0) != 0) continue;
            const std::string rest = s.substr(p.size());
            if (rest.empty() || !std::isdigit(static_cast<unsigned char>(rest[0]))) continue;
            n = static_cast<size_t>(rest[0] - '0');
            arity = 2;
            const size_t ap = rest.find('a');
            if (ap != std::string::npos && ap + 1 < rest.size())
                arity = static_cast<size_t>(rest[ap + 1] - '0');
            shape = p;
            return n >= 1 && n <= 6 && arity >= 2 && arity <= 4;
        }
        return false;
    };

    // ONE GENERATOR FOR THE PATTERN AND FOR THE STATE IT RUNS ON, which is not a tidiness point
    // but a correctness one. The initial state used to be a chain of binary edges whatever the
    // rule was, so a star, a cycle, a tree or any arity above two matched NOTHING: those runs
    // reported one state and zero events, and a sweep over them would have collected empty rows
    // that look like data. The state is now the same shape as the pattern, built by this
    // function with a different edge count, so a match exists by construction.
    auto build_shape = [](const std::string& shape, size_t n, size_t arity) {
        std::vector<std::vector<uint32_t>> out;
        uint32_t next = 0;
        auto fresh = [&next]() { return next++; };
        auto arity_at = [&](size_t i) { return (shape == "mixed") ? (2 + (i % 2)) : arity; };

        if (shape == "star") {
            const uint32_t hub = fresh();
            for (size_t i = 0; i < n; ++i) {
                std::vector<uint32_t> e{hub};
                for (size_t vi = 1; vi < arity_at(i); ++vi) e.push_back(fresh());
                out.push_back(e);
            }
        } else if (shape == "disc") {
            for (size_t i = 0; i < n; ++i) {
                std::vector<uint32_t> e;
                for (size_t vi = 0; vi < arity_at(i); ++vi) e.push_back(fresh());
                out.push_back(e);
            }
        } else if (shape == "tree") {
            std::vector<uint32_t> verts{fresh()};
            for (size_t i = 0; i < n; ++i) {
                std::vector<uint32_t> e{verts[i / 2]};
                for (size_t vi = 1; vi < arity_at(i); ++vi) {
                    const uint32_t v = fresh(); e.push_back(v); verts.push_back(v);
                }
                out.push_back(e);
            }
        } else {   // chain, cycle, mixed -- all paths; cycle closes the last edge onto the first
            // CONSECUTIVE EDGES OVERLAP IN arity-1 VERTICES: edge i covers i..i+arity-1. That
            // overlap is the whole content of the arity axis. Giving each edge its own fresh
            // extra vertices instead makes them degree-one, constraining nothing, so a path of
            // n arity-a edges has exactly the join structure of n binary edges -- measured, and
            // it showed as arity 2, 3 and 4 returning IDENTICAL states, canonical states and
            // causal edges (61,048 / 2,356 / 92,164 at depth 5). An axis that changes no
            // measured quantity is not an axis.
            uint32_t maxv = 0;
            for (size_t i = 0; i < n; ++i) {
                std::vector<uint32_t> e;
                for (size_t vi = 0; vi < arity_at(i); ++vi) {
                    const auto v = static_cast<uint32_t>(i + vi);
                    e.push_back(v);
                    if (v > maxv) maxv = v;
                }
                out.push_back(e);
            }
            next = maxv + 1;
            if (shape == "cycle" && n >= 2) out.back().back() = out.front().front();
        }
        return out;
    };

    auto family_rule = [&build_shape](const std::string& shape, size_t n, size_t arity) {
        auto b = make_rule(0);
        const auto lhs = build_shape(shape, n, arity);
        uint32_t next = 0;
        for (const auto& e : lhs) for (uint32_t v : e) next = std::max(next, v + 1);

        for (const auto& e : lhs) {
            std::vector<uint8_t> t; t.reserve(e.size());
            for (uint32_t v : e) t.push_back(static_cast<uint8_t>(v));
            b.lhs(t);
            b.rhs(t);
        }
        // The appended edge: same arity as the family, anchored on the highest LHS vertex.
        const size_t a_out = (shape == "mixed") ? 2 : arity;
        std::vector<uint8_t> add{static_cast<uint8_t>(next - 1)};
        for (size_t vi = 1; vi < a_out; ++vi) add.push_back(static_cast<uint8_t>(next++));
        b.rhs(add);
        return b.build();
    };

    size_t fam_n = 0, fam_arity = 0;
    std::string fam_shape;
    const bool is_family = parse_family(rule, fam_n, fam_arity, fam_shape);

    RewriteRule r =
        is_family        ? family_rule(fam_shape, fam_n, fam_arity) :
        rule == "pair"   ? chain_rule(2) :
        rule == "triple" ? chain_rule(3) :
        rule == "quad"   ? chain_rule(4) :
        // The published two-component shape, kept under its published name so the existing
        // table's rows keep meaning what they meant. disc2a2 is the same idea in the family
        // grammar, with the family's own right-hand side convention.
        rule == "disc"   ? make_rule(0).lhs({0, 1}).lhs({2, 3})
                                       .rhs({0, 1}).rhs({2, 4}).rhs({4, 3}).build() :
        // "growth": the one-edge LHS, which keeps its edge and appends rather than subdividing.
                           make_rule(0).lhs({0, 1}).rhs({0, 1}).rhs({1, 2}).build();

    // The initial state is the SAME SHAPE as the pattern, with `edges` edges rather than the
    // pattern's. For the named legacy rules that is a binary chain, which is what they have
    // always run on.
    std::vector<std::vector<VertexId>> init;
    {
        // A CYCLE IS THE ONE SHAPE WHOSE STATE CANNOT BE MADE LARGER THAN ITS PATTERN. An
        // n-edge cycle pattern matches an L-edge cycle only when L == n -- a longer ring
        // contains no shorter ring -- so scaling the seed by `edges` would produce a state the
        // rule can never fire on, which is what "states=1, events=0" meant before this.
        const size_t seed_n = (is_family && fam_shape == "cycle") ? fam_n : edges;
        const auto seed = is_family ? build_shape(fam_shape, seed_n, fam_arity)
                                    : build_shape("chain", seed_n, 2);
        for (const auto& e : seed) {
            std::vector<VertexId> t; t.reserve(e.size());
            for (uint32_t v : e) t.push_back(static_cast<VertexId>(v));
            init.push_back(t);
        }
    }

    // HG_CAPACITY_SCALE multiplies every append-only array's segment size, and it is the only way
    // past the container ceiling. A shape that saturates produces rows this tool marks truncated
    // and every consumer drops, so without it the shape is simply absent from the evidence.
    uint32_t cap_scale = 1;
    if (const char* cs = std::getenv("HG_CAPACITY_SCALE")) {
        const int v = std::atoi(cs);
        if (v > 0) cap_scale = static_cast<uint32_t>(v);
    }
    Hypergraph hg(cap_scale);
    hg.set_state_canonicalization_mode(
        canon == "none"      ? StateCanonicalizationMode::None :
        canon == "automatic" ? StateCanonicalizationMode::Automatic
                             : StateCanonicalizationMode::Full);
    ParallelEvolutionEngine e(&hg, threads);
    e.set_random_seed(12345);
    e.add_rule(r);
    if      (arm == "off")    { /* the baseline: no sampling of any kind */ }
    else if (arm == "cap")    e.set_max_successor_states_per_parent(k);
    else if (arm == "rate")   e.set_transition_rate(1.0 / static_cast<double>(k));
    else if (arm == "fwdon")  e.set_match_forwarding(true);
    else if (arm == "fwdoff") e.set_match_forwarding(false);
    // The rate with forwarding disabled. Forwarded matches reach a state through
    // push_match_to_children / forward_matches_from_single_ancestor, and a sampler that
    // did not reach those dispatches would bound the run here and not with forwarding on --
    // which is how the per-state reservoir this arm replaced was caught.
    else if (arm == "ratenofwd") {
        e.set_transition_rate(1.0 / static_cast<double>(k));
        e.set_match_forwarding(false);
    }

    std::printf("evolving...\n");
    const auto t0 = std::chrono::steady_clock::now();
    e.evolve(init, steps);
    const auto t1 = std::chrono::steady_clock::now();

    // Per-depth width and the branching factor between depths. A fixed thinning rate q makes
    // the expected width evolve as the product of m(d)*q, so whether q must depend on depth is
    // entirely the question of whether m does -- which is this histogram, not an argument.
    size_t max_width = 0, depth_reached = 0;
    {
        std::vector<size_t> width;
        for (uint32_t sid = 0; sid < hg.num_published_states(); ++sid) {
            const auto& st = hg.get_state(sid);
            if (st.id == INVALID_ID) continue;
            if (st.step >= width.size()) width.resize(st.step + 1, 0);
            width[st.step]++;
        }
        for (size_t d = 0; d < width.size(); ++d)
            if (width[d] > max_width) max_width = width[d];
        depth_reached = width.empty() ? 0 : width.size() - 1;
        std::printf("depth  width      m(d)=width(d+1)/width(d)\n");
        for (size_t d = 0; d < width.size(); ++d) {
            if (d + 1 < width.size() && width[d] > 0) {
                std::printf("%5zu  %-9zu %.2f\n", d, width[d],
                            static_cast<double>(width[d + 1]) / width[d]);
            } else {
                std::printf("%5zu  %-9zu -\n", d, width[d]);
            }
        }
    }

    std::printf("done %.2f ms  states=%zu canon=%zu events=%zu matches=%zu drained=%zu\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count(),
                static_cast<size_t>(hg.num_states()),
                static_cast<size_t>(hg.num_canonical_states()),
                static_cast<size_t>(hg.num_events()),
                e.total_matches(), e.states_drained());

    // ONE MACHINE-READABLE LINE CARRYING EVERY COUNT THIS RUN ALREADY COMPUTED. The `done` line
    // above reports the size of the state space; these are the sizes of the RELATIONS over it,
    // and they are what distinguishes one workload from another at equal state count -- a run
    // with 200k states and 40k causal edges is a different object from one with 200k states and
    // 4M. They cost nothing here: the causal and branchial accessors are derived from sets the
    // evolution already built, so reporting them adds no work to the measured region, which ends
    // at t1 above.
    //
    // key=value, one line, stable key order, so a sweep can parse it without a format per plot.
    // WARNINGS ARE PART OF THE RESULT, NOT DECORATION. The engine catches a container ceiling,
    // pushes a warning and returns a TRUNCATED graph rather than throwing -- and
    // parallel_evolution.cpp says why in the comment on that path: "a caller that reads the count
    // and not the warnings gets a smaller multiway system with no indication it is smaller, which
    // is the one outcome worse than the throw." This tool WAS that caller. A depth sweep built on
    // it plotted saturated runs as though the state space had stopped growing: measured,
    // chain1a2 totals exactly 2,097,149 states whether asked for depth 9, 10 or 11, and the width
    // at depth 9 comes out 1,965,054 / 116,128 / 6,528 in those three runs because the ceiling is
    // redistributed over however many levels were requested.
    //
    // truncated=1 in the RICH line is therefore a FIELD, so a consumer can drop those rows
    // without re-deriving the ceiling from the numbers.
    //
    // IT TESTS FOR THE CEILING AND NOT FOR "A WARNING". The engine warns about things that are
    // not truncation -- a disconnected left side costs a cartesian product and says so, and a run
    // that discards work without a seed says it is not reproducible. Treating any warning as
    // truncation marks every disconnected workload truncated whether or not it reached the
    // ceiling, which silently deletes the disconnected shape from every figure that filters on
    // this field.
    bool truncated = false;
    for (const auto& w : e.warnings())
        if (w.find("capacity limit reached") != std::string::npos) truncated = true;
    for (const auto& w : e.warnings()) std::printf("WARNING %s\n", w.c_str());

    // The causal and branchial sizes come from the CausalGraph the full-capture run actually
    // built. The num_reconstructed_* accessors report the quotient reconstruction and read zero
    // outside quotient mode, so they are the wrong instrument for this sweep.
    const auto& cg = hg.causal_graph();
    std::printf("RICH rule=%s shape=%s arity=%zu lhs_edges=%zu init_edges=%zu steps=%zu"
                " threads=%zu canon_mode=%s"
                " ms=%.3f states=%zu canonical=%zu events=%zu matches=%zu"
                " causal_edges=%zu causal_pairs=%zu branchial_edges=%zu branchial_claimed=%zu"
                " max_width=%zu depth_reached=%zu truncated=%d"
                " main_arena_b=%zu arena_blocks_b=%zu"
                " discovered=%zu forwarded=%zu invalidated=%zu rewalks=%zu"
                " discarded_tables_b=%zu discarded_tables=%zu"
                " installed_tables_b=%zu installed_tables=%zu\n",
                rule.c_str(),
                is_family ? fam_shape.c_str()
                          : (rule == "disc" ? "disc" : "chain"),
                is_family ? fam_arity : size_t{2},
                is_family ? fam_n
                          : (rule == "pair" ? size_t{2} : rule == "triple" ? size_t{3}
                             : rule == "quad" ? size_t{4} : rule == "disc" ? size_t{2}
                             : size_t{1}),
                edges, steps, threads, canon.c_str(),
                std::chrono::duration<double, std::milli>(t1 - t0).count(),
                static_cast<size_t>(hg.num_states()),
                static_cast<size_t>(hg.num_canonical_states()),
                static_cast<size_t>(hg.num_events()), e.total_matches(),
                cg.num_causal_edges(), cg.num_causal_event_pairs(),
                cg.num_branchial_edges(), cg.num_branchial_pairs_claimed(),
                max_width, depth_reached, truncated ? 1 : 0,
                hg.arena().bytes_allocated(), arena_block_bytes_live(),
                e.stats().total().new_matches_discovered, e.stats().total().matches_forwarded,
                e.stats().total().matches_invalidated, e.stats().total().forwarding_rewalks,
                discarded_table_bytes(), discarded_table_count(),
                installed_table_bytes(), installed_table_count());
    return 0;
}
