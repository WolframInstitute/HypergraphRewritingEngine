#pragma once
#include "hgcommon/namespace.hpp"
// "StateCost": a user expression over a state's "Invariants" statistics, one body for host and
// device. Under "MaxStatesPerStep" (or "MatchesPerStep") a step keeps the transitions into its
// N lowest-cost successor states, the successor rank (hgcommon::successor_ranks) ordering states
// of equal cost.
//
// The paclet compiles the expression into a postfix program (StateCost): constants, statistic
// loads by "Invariants" slot (step_pool_core.hpp, sp_invariant), and the operations below. The
// interpreter (state_cost_eval) uses only IEEE addition, subtraction, division and square root,
// dm_mul for products and det_math.hpp for log, exp and non-integer powers, so a program gives
// one double on every target.
//
// A state with no value for a statistic the program reads, or whose cost is NaN, has cost
// +Infinity and ranks after every state with a finite cost.
#include "hgcommon/core.hpp"
#include "hgcommon/det_math.hpp"
#include "hgcommon/ir_core.hpp"   // ir_heapsort_idx
#include "hgcommon/step_pool_core.hpp"

#include <cstdint>

namespace HG_NAMESPACE {
namespace common {

// The operations. SC_CONST pushes arg; SC_STAT pushes "Invariants" slot arg; SC_POW raises the
// top to the constant arg; the unary operations replace the top; the binary ones pop b, then a,
// and push a op b.
enum : uint8_t {
    SC_CONST = 0,
    SC_STAT = 1,
    SC_ADD = 2,
    SC_SUB = 3,
    SC_MUL = 4,
    SC_DIV = 5,
    SC_MIN = 6,
    SC_MAX = 7,
    SC_NEG = 8,
    SC_ABS = 9,
    SC_SQRT = 10,
    SC_LOG = 11,
    SC_EXP = 12,
    SC_POW = 13,
    SC_OPS = 14,
};

constexpr uint32_t SC_MAX_OPS = 64;

// A compiled cost program. n == 0 is no cost. `measures` is the `want` bits (state_record) of
// the statistics the program reads.
struct StateCost {
    uint32_t n = 0;
    uint32_t measures = 0;
    uint8_t op[SC_MAX_OPS] = {};
    double arg[SC_MAX_OPS] = {};
};

HG_HD inline bool state_cost_active(const StateCost& c) { return c.n != 0; }

// The stack change of operation `op`: +1 for a load, 0 for a unary operation, -1 for a binary.
HG_HD inline int sc_stack_delta(uint8_t op) {
    if (op == SC_CONST || op == SC_STAT) return 1;
    if (op >= SC_ADD && op <= SC_MAX) return -1;
    return 0;
}

// Whether `c` is a program state_cost_eval can run: known operations, statistic slots below
// SP_NUM_INVARIANTS, no operation reading an empty stack, and one value left. Sets c.measures.
HG_HD inline bool state_cost_validate(StateCost& c) {
    if (c.n > SC_MAX_OPS) return false;
    int depth = 0;
    uint32_t measures = 0;
    for (uint32_t i = 0; i < c.n; ++i) {
        const uint8_t op = c.op[i];
        if (op >= SC_OPS) return false;
        if (op == SC_STAT) {
            const double s = c.arg[i];
            if (!(s >= 0.0 && s < static_cast<double>(SP_NUM_INVARIANTS))) return false;
            if (s != static_cast<double>(static_cast<uint32_t>(s))) return false;
            measures |= sp_invariant_want(static_cast<uint32_t>(s));
        }
        const int need = op == SC_CONST || op == SC_STAT ? 0 : (sc_stack_delta(op) < 0 ? 2 : 1);
        if (depth < need) return false;
        depth += sc_stack_delta(op);
    }
    if (depth != 1) return false;
    c.measures = measures;
    return true;
}

// x^y for a constant y. An integer |y| <= 64 multiplies (dm_mul) by squaring, so a negative base
// is allowed; any other y is dm_pow, defined for x > 0 (x = 0 gives 0 for y > 0 and +Infinity for
// y < 0; x < 0 gives NaN).
HG_HD inline double sc_pow(double x, double y) {
    if (y == 0.0) return 1.0;
    const double ay = y < 0.0 ? -y : y;
    if (ay <= 64.0 && ay == static_cast<double>(static_cast<uint32_t>(ay))) {
        uint32_t e = static_cast<uint32_t>(ay);
        double base = x, r = 1.0;
        while (e) {
            if (e & 1u) r = dm_mul(r, base);
            e >>= 1u;
            if (e) base = dm_mul(base, base);
        }
        return y < 0.0 ? 1.0 / r : r;
    }
    if (x != x || y != y) return x + y;
    if (x < 0.0) return ::nan("");
    if (x == 0.0) return y > 0.0 ? 0.0 : HUGE_VAL;
    if (x == HUGE_VAL) return y > 0.0 ? HUGE_VAL : 0.0;
    const double t = dm_mul(y, dm_log(x));
    if (t > 709.0) return HUGE_VAL;
    if (t < -745.0) return 0.0;
    return dm_exp(t);
}

HG_HD inline double sc_log(double x) {
    if (x != x || x < 0.0) return ::nan("");
    if (x == 0.0) return -HUGE_VAL;
    if (x == HUGE_VAL) return HUGE_VAL;
    return dm_log(x);
}

HG_HD inline double sc_exp(double x) {
    if (x != x) return x;
    if (x > 709.0) return HUGE_VAL;
    if (x < -745.0) return 0.0;
    return dm_exp(x);
}

// The cost of a state with values `s` and geometry `g` under the validated program `c`;
// +Infinity when the state has no value for a statistic the program reads or the cost is NaN.
HG_HD inline double state_cost_eval(const StateCost& c, const StateInvariantValues& s,
                                    const SgGeometry& g) {
    double st[SC_MAX_OPS];
    uint32_t sp = 0;
    for (uint32_t i = 0; i < c.n; ++i) {
        const double a = c.arg[i];
        switch (c.op[i]) {
            case SC_CONST: st[sp++] = a; break;
            case SC_STAT: {
                double v = 0.0;
                if (!sp_invariant(s, g, static_cast<uint32_t>(a), v)) return HUGE_VAL;
                st[sp++] = v;
                break;
            }
            case SC_ADD: --sp; st[sp - 1] = st[sp - 1] + st[sp]; break;
            case SC_SUB: --sp; st[sp - 1] = st[sp - 1] - st[sp]; break;
            case SC_MUL: --sp; st[sp - 1] = dm_mul(st[sp - 1], st[sp]); break;
            case SC_DIV: --sp; st[sp - 1] = st[sp - 1] / st[sp]; break;
            case SC_MIN: --sp; st[sp - 1] = st[sp] < st[sp - 1] ? st[sp] : st[sp - 1]; break;
            case SC_MAX: --sp; st[sp - 1] = st[sp] > st[sp - 1] ? st[sp] : st[sp - 1]; break;
            case SC_NEG: st[sp - 1] = -st[sp - 1]; break;
            case SC_ABS: st[sp - 1] = st[sp - 1] < 0.0 ? -st[sp - 1] : st[sp - 1]; break;
            case SC_SQRT: st[sp - 1] = st[sp - 1] < 0.0 ? ::nan("") : ::sqrt(st[sp - 1]); break;
            case SC_LOG: st[sp - 1] = sc_log(st[sp - 1]); break;
            case SC_EXP: st[sp - 1] = sc_exp(st[sp - 1]); break;
            case SC_POW: st[sp - 1] = sc_pow(st[sp - 1], a); break;
            default: return HUGE_VAL;
        }
    }
    const double r = st[0];
    return r != r ? HUGE_VAL : r;
}

// A cost as a key whose unsigned order is the numeric order (sp_key_of).
HG_HD inline uint64_t state_cost_key(double cost) { return sp_key_of(cost != cost ? HUGE_VAL : cost); }

// THE COST ORDER: successors by cost key, then by successor rank (successor_ranks).
HG_HD inline bool cost_before(uint64_t cost_a, uint64_t rank_a, uint64_t cost_b, uint64_t rank_b) {
    return cost_a != cost_b ? cost_a < cost_b : rank_a < rank_b;
}

// The cut of a k-lowest selection in the cost order: the k-th smallest (cost, rank) counted with
// multiplicity, `all` when there are at most k. cost_cut_keeps keeps every entry at or below it,
// the entries with fewer than k strictly before them (cap_keeps). The device finds it by radix
// selection over the 128-bit (cost, rank) key; the host ranks by cost_successor_ranks.
struct CostCut {
    uint64_t cost;
    uint64_t rank;
    bool     all;
};

HG_HD inline bool cost_cut_keeps(CostCut c, uint64_t cost, uint64_t rank) {
    return c.all || !cost_before(c.cost, c.rank, cost, rank);
}

// out[i] is the dense position of the pair (cost[i], srank[i]) among the pairs of candidates
// 0..n in the cost order. Equal pairs take one position, so a k-lowest cut over `out` (rank_cut,
// rank_cut_where) keeps the same successors as one over the pairs. `order` is n words of scratch;
// `out` does not alias `srank`. O(n log n).
HG_HD inline void cost_successor_ranks(const uint64_t* cost, const uint64_t* srank, uint32_t n,
                                       uint32_t* order, uint64_t* out) {
    for (uint32_t i = 0; i < n; ++i) order[i] = i;
    ir_heapsort_idx(order, n, [&](uint32_t a, uint32_t b) {
        if (cost_before(cost[a], srank[a], cost[b], srank[b])) return -1;
        if (cost_before(cost[b], srank[b], cost[a], srank[a])) return 1;
        return a < b ? -1 : (a > b ? 1 : 0);
    });
    uint64_t pos = 0;
    for (uint32_t j = 0; j < n; ++j) {
        const uint32_t i = order[j], p = j ? order[j - 1] : i;
        if (j > 0 && (cost[i] != cost[p] || srank[i] != srank[p])) ++pos;
        out[i] = pos;
    }
}

}  // namespace common
}  // namespace HG_NAMESPACE
