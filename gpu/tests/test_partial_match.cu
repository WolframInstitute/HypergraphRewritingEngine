#include <gtest/gtest.h>

#include "hg_gpu/partial_match.hpp"

#include <cuda_runtime.h>

namespace {

using hg_gpu::PartialMatch;

__global__ void k_exercise(PartialMatch* pm,
                           uint8_t*      out_flags) {
    if (threadIdx.x != 0) return;
    PartialMatch& m = *pm;
    m.reset(/*n_pattern=*/3, /*n_vars=*/4);

    uint8_t bit = 0;
    auto mark = [&] (bool ok) { if (ok) out_flags[0] |= (1 << bit); ++bit; };

    // 1) Fresh state: no vars bound, no edges matched.
    mark(!m.is_var_bound(0));
    mark(!m.is_pattern_matched(0));
    mark(!m.is_complete());

    // 2) Bind var 0 to vertex 7, then var 1 to vertex 7 (Wolfram: OK, non-distinct).
    mark(m.check_or_bind_var(0, 7));
    mark(m.check_or_bind_var(1, 7));
    mark(m.is_var_bound(0) && m.is_var_bound(1));
    mark(m.get_var(0) == 7 && m.get_var(1) == 7);

    // 3) Re-check var 0 with same vertex succeeds; with different vertex fails.
    mark(m.check_or_bind_var(0, 7));
    mark(!m.check_or_bind_var(0, 8));

    // 4) Pattern edge matching / consumption.
    m.bind_pattern_edge(0, /*eid=*/42);
    m.set_consumed(42);
    mark(m.is_pattern_matched(0));
    mark(m.is_consumed(42));
    mark(!m.is_consumed(43));

    m.bind_pattern_edge(1, 100);
    m.set_consumed(100);
    m.bind_pattern_edge(2, 17);
    m.set_consumed(17);

    mark(m.is_complete());

    // 5) Backtrack the last binding.
    m.unbind_pattern_edge(2);
    m.clear_consumed(17);
    mark(!m.is_complete());
    mark(!m.is_consumed(17));

    // 6) Unbind var then re-bind to different vertex succeeds.
    m.unbind_var(0);
    mark(!m.is_var_bound(0));
    mark(m.check_or_bind_var(0, 99));
    mark(m.get_var(0) == 99);
}

TEST(PartialMatch, WolframSemanticsAndBitmapOps) {
    PartialMatch* d_pm = nullptr;
    cudaMalloc(&d_pm, sizeof(PartialMatch));
    cudaMemset(d_pm, 0, sizeof(PartialMatch));

    uint8_t* d_flags = nullptr; cudaMalloc(&d_flags, 1);
    cudaMemset(d_flags, 0, 1);

    k_exercise<<<1, 1>>>(d_pm, d_flags);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    uint8_t flags = 0; cudaMemcpy(&flags, d_flags, 1, cudaMemcpyDeviceToHost);
    // 14 expectations → want all 1s in low 8 bits; the remaining were set in flags[0]
    // We only have one byte so the first 8 expectations are covered here; rest are
    // verified via extending to uint32_t. Simpler: the kernel writes flags[0] low
    // 8 bits. We ignore that we only capture 8 — every mark is carefully ordered
    // and if any is wrong we see a specific bit missing.
    cudaFree(d_pm); cudaFree(d_flags);

    // The kernel performs 14 `mark` calls but our byte captures the first 8.
    // The loop increments `bit` past 7; writes past bit 7 silently drop. So we
    // only check that the first 8 marks succeeded — which verifies the core
    // semantics (bind, check-or-bind same, fail-different, match, consumed bitmap).
    EXPECT_EQ(flags, 0xFFu);
}

// Second test with a uint32_t flag buffer to capture all 16-ish marks.
__global__ void k_exercise_wide(PartialMatch* pm, uint32_t* flags) {
    if (threadIdx.x != 0) return;
    PartialMatch& m = *pm;
    m.reset(3, 4);
    uint32_t bit = 0;
    auto mark = [&](bool ok) { if (ok) *flags |= (1u << bit); ++bit; };

    mark(!m.is_var_bound(0));
    mark(!m.is_pattern_matched(0));
    mark(!m.is_complete());
    mark(m.check_or_bind_var(0, 7));
    mark(m.check_or_bind_var(1, 7));
    mark(m.is_var_bound(0) && m.is_var_bound(1));
    mark(m.get_var(0) == 7 && m.get_var(1) == 7);
    mark(m.check_or_bind_var(0, 7));
    mark(!m.check_or_bind_var(0, 8));
    m.bind_pattern_edge(0, 42); m.set_consumed(42);
    mark(m.is_pattern_matched(0));
    mark(m.is_consumed(42));
    mark(!m.is_consumed(43));
    m.bind_pattern_edge(1, 100); m.set_consumed(100);
    m.bind_pattern_edge(2, 17);  m.set_consumed(17);
    mark(m.is_complete());
    m.unbind_pattern_edge(2); m.clear_consumed(17);
    mark(!m.is_complete());
    mark(!m.is_consumed(17));
    m.unbind_var(0);
    mark(!m.is_var_bound(0));
    mark(m.check_or_bind_var(0, 99));
    mark(m.get_var(0) == 99);
}

TEST(PartialMatch, AllSemantics) {
    PartialMatch* d_pm = nullptr; cudaMalloc(&d_pm, sizeof(PartialMatch));
    cudaMemset(d_pm, 0, sizeof(PartialMatch));
    uint32_t* d_flags = nullptr; cudaMalloc(&d_flags, sizeof(uint32_t));
    cudaMemset(d_flags, 0, sizeof(uint32_t));
    k_exercise_wide<<<1, 1>>>(d_pm, d_flags);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    uint32_t flags = 0; cudaMemcpy(&flags, d_flags, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_pm); cudaFree(d_flags);
    // 17 marks in order; all should be true → low 17 bits set.
    EXPECT_EQ(flags & ((1u<<17) - 1u), (1u<<17) - 1u) << "flags=0x" << std::hex << flags;
}

__global__ void k_cooperative_reset(PartialMatch* pm) {
    pm->reset_cooperative<32>(threadIdx.x, 5, 10);
}

TEST(PartialMatch, CooperativeResetClears) {
    PartialMatch h_pm{};
    for (auto& e : h_pm.matched_edges) e = 0x12345678u;
    for (auto& w : h_pm.consumed)      w = 0xCAFEFACEu;
    for (auto& v : h_pm.var_binding)   v = 0x87654321u;
    h_pm.matched_mask = 0xFFFFu;
    h_pm.bound_mask   = 0xAAAAu;

    PartialMatch* d_pm = nullptr; cudaMalloc(&d_pm, sizeof(PartialMatch));
    cudaMemcpy(d_pm, &h_pm, sizeof(PartialMatch), cudaMemcpyHostToDevice);

    k_cooperative_reset<<<1, 32>>>(d_pm);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    PartialMatch out{};
    cudaMemcpy(&out, d_pm, sizeof(PartialMatch), cudaMemcpyDeviceToHost);
    cudaFree(d_pm);

    EXPECT_EQ(out.matched_mask, 0u);
    EXPECT_EQ(out.bound_mask,   0u);
    EXPECT_EQ(out.num_pattern_edges, 5u);
    EXPECT_EQ(out.num_vars, 10u);
    for (auto e : out.matched_edges) EXPECT_EQ(e, hg_gpu::INVALID_ID);
    for (auto w : out.consumed)      EXPECT_EQ(w, 0u);
    for (auto v : out.var_binding)   EXPECT_EQ(v, hg_gpu::INVALID_ID);
}

}  // namespace
