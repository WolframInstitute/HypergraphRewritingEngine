# Release acceptance checklist

*Recovered from the alpha.6 release notes (the only end-to-end release-acceptance procedure that
existed, and it was buried in a changelog). This is the gate a release must pass, tightened for v1.0.
Keep it current: a release that skips a line here is not released.*

## Build artifacts
- [ ] All 6 platform libraries built (Linux/Windows/macOS × x86-64/ARM64).
- [ ] All 6 `hg_evolve` process binaries built.
- [ ] `hg_evolve_gpu` built on both CUDA platforms.
      *(v1.0 blocker: the native Windows MSVC+nvcc whole-stack config — until it lands,
      `TargetDevice->"GPU"` silently falls back to CPU on Windows.)*
- [ ] `.paclet` archive produced.
- [ ] `DocumentationBuild` passes (was 24/24) — note this **evaluates every example cell**, so it is
      also the docs-can't-rot gate.
- [ ] **Static-link contract holds:** `hg_evolve_gpu.exe` imports only `KERNEL32`/`WS2_32`
      (static `libcudart_static` + `/MT`); the only runtime dependency is `nvcuda.dll`.

## Functional verification
- [ ] The assembled `.paclet` is installed and exercised via wolframscript.
- [ ] `HGEvolve` runs through the `hg_evolve` / `hg_evolve_gpu` **processes** (isolation confirmed).
- [ ] CPU results correct across `None` / `Automatic` / `Full`.
- [ ] GPU results match CPU `CanonicalizeStates -> Full` **with no device fallback**.
- [ ] `HGEvolve` example pages evaluate cleanly against the local engine.

## Test gates
- [ ] CPU suite green (was 190; currently 194).
- [ ] CPU↔GPU differential green — states/events/causal/branchial equivalent up to isomorphism,
      plus per-mode `NumStates` (was 24/24).
- [ ] `gpu_differential_tests` and `hg_gpu_tests` green (were 24/24 and 77/77).
- [ ] Determinism gate green **with TR on and quotient on** (the acceptance test for the causal work).
- [ ] Oracle corpus + golden corpus green, including the event-canonicalization parity columns.

## v1.0 additions to the above
- [ ] **No user-facing doc states something a user can act on and be wrong about** (see the
      inaccuracy register in `V1_SCOPING_REGISTER.md` §C3 — `HashStrategy`, `EquilibriumAnalysis`,
      the quotient/TR interaction, `Automatic` semantics, the `"States"` return shape).
- [ ] **An OSS license exists** (none today — blocking for a public release).
- [ ] No silent correctness degradation anywhere: GPU IR→1-WL fallback surfaced; unknown options
      rejected rather than skipped; quotient/TR interaction visible to the user.
- [ ] Every advertised option exists and every existing option is documented (generated reference).

## Shipped semantic changes to carry forward in release notes
*(recovered from alpha.6 — these are user-visible and currently have no other home)*
- `exploration_probability` samples **per canonical state**, not per transition.
- `quotient_initial_states` default keeps all roots, matching the reference `MultiwaySystem`.
- Quotient exploration expands each canonical state once **at its shortest depth**.
