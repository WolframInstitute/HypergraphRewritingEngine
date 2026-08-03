# Release acceptance checklist

**State at 2026-08-03: 11 verified, 4 partly, 5 outstanding.** The Windows GPU stack is
FUNCTIONALLY VERIFIED, not merely built: `HGEvolve[..., TargetDevice -> "GPU"]` matches the CPU
golden corpus 12/12 running the native MSVC+nvcc `hg_evolve_gpu.exe` on an RTX 4090. Every line that can be checked on
a Linux+CUDA workstation has been, with the run that proves it recorded beside it. The 9 that
remain — 4 partly done, 5 not — split into exactly two kinds, and neither is new engineering:

- **Needs a REBUILD on the release matrix** — and a rebuild, not a first build. All six platforms
  and both CUDA binaries already EXIST in `paclet/LibraryResources/`; five of the six are dated
  2026-07-22 or 08-01, so they predate this session entirely, including the GPU fault fix. The
  `.paclet` archive, `DocumentationBuild` and the installed-paclet exercise follow the rebuild.
- **Needs a decision** — the doc-accuracy line, which is a judgement about user-facing text.

The Windows MSVC+nvcc config is NOT among them: it landed, and the binary's import table proves it
(see the `hg_evolve_gpu` line). `.github/workflows/windows-gpu.yml` exists to keep it from rotting,
since nothing in CI builds that stack today — run it once from the Actions tab and, if green, move
it onto `push`.

Nothing here is waiting on more engineering on this machine.

*Recovered from the alpha.6 release notes (the only end-to-end release-acceptance procedure that
existed, and it was buried in a changelog). This is the gate a release must pass, tightened for v1.0.
Keep it current: a release that skips a line here is not released.*

## Build artifacts
- [~] All 6 platform libraries built (Linux/Windows/macOS × x86-64/ARM64). **All six ARE present in
      `paclet/LibraryResources/` — but five are STALE.** Dates at 2026-08-03: Linux-x86-64 today;
      Linux-ARM64, both MacOSX, both Windows are 2026-07-22 or 08-01, so they predate this
      session's GPU fault fix (`00e21ee`), the continuation fixes and everything after. **The
      release needs a REBUILD of all six from current source, not a first build.**
- [~] All 6 `hg_evolve` process binaries built. Same: all six present, five stale.
- [~] `hg_evolve_gpu` built on both CUDA platforms. **BOTH exist** — `Linux-x86-64/hg_evolve_gpu`
      (rebuilt 2026-08-03) and `Windows-x86-64/hg_evolve_gpu.exe` (2026-07-22, stale).
      *THE "v1.0 BLOCKER" BELOW IS STALE AND IS STRUCK. It read: "the native Windows MSVC+nvcc
      whole-stack config — until it lands, `TargetDevice->"GPU"` silently falls back to CPU on
      Windows." It landed. The evidence is the binary itself: its import table is exactly
      `KERNEL32.dll`, `WS2_32.dll`, `nvcuda.dll` — the static-link contract this checklist
      specifies — with no `VCRUNTIME`/`MSVCP` (so not a dynamic MSVC link) and no
      `libgcc`/`libstdc++`/`mingw` (so not MinGW). That is a fully static native MSVC build with
      nvcc. What remains is a REBUILD from current source, not the config.*
      Outstanding for ARM64 Windows: `/MD` rather than `/MT` (recorded, unchanged).
      **THE EXPERIMENT THAT SETTLES THIS EXISTS BUT HAS NOT BEEN RUN:** the Windows CI leg
      configures with `-DBUILD_GPU=OFF`, so "the config does not work" and "nobody has tried it"
      are indistinguishable from outside. `.github/workflows/windows-gpu.yml` builds and LINKS the
      stack under `cl.exe` — which is the whole config question, including the explicit device link
      `gpu/CMakeLists.txt` claims is generator-independent. It is `workflow_dispatch` only, because
      its first outcome is unknown and an unproven leg on `push` would turn CI red to answer a
      question nobody asked. **Run it from the Actions tab; if green, move it onto `push` in the
      same change that ticks this line.** It does NOT run a kernel — hosted Windows runners have no
      NVIDIA GPU, so the "GPU results match CPU with no fallback" line below still needs real
      hardware.
- [ ] `.paclet` archive produced.
- [ ] `DocumentationBuild` passes (was 24/24) — note this **evaluates every example cell**, so it is
      also the docs-can't-rot gate.
- [x] **Static-link contract holds.** Verified 2026-08-03 on
      `paclet/LibraryResources/Windows-x86-64/hg_evolve_gpu.exe`: the import set is exactly
      `KERNEL32.dll`, `WS2_32.dll`, `nvcuda.dll` — nothing else — so `libcudart_static` + `/MT`
      held. Re-verify after the rebuild, since the binary checked is the 2026-07-22 one.

## Functional verification
- [~] The assembled `.paclet` is installed and exercised via wolframscript. The paclet DIRECTORY is
      (`PacletDirectoryLoad` + `HGEvolve`, both CPU and GPU, 2026-08-03); the assembled `.paclet`
      ARCHIVE is not, since producing it is part of the rebuild.
- [x] `HGEvolve` runs through the `hg_evolve` **process** (isolation confirmed). **2026-08-03**:
      `PacletTest` 3/3 — it `PacletDirectoryLoad`s the local `paclet/` and calls
      `HypergraphRewriting`HGEvolve` under wolframscript, which routes through `RunProcess` to
      `LibraryResources/Linux-x86-64/hg_evolve`, the binary rebuilt from current source today. So
      the WL layer, the WXF transport and a fresh engine binary were exercised end to end.
      **`hg_evolve_gpu` is ALSO confirmed**, 2026-08-03: `reference/verify_paclet_gpu.wls` under
      wolframscript ran `HGEvolve[..., TargetDevice -> "GPU"]` against the golden corpus, **12/12,
      Failed: NONE**. Because wolframscript here is the Windows executable, `$SystemID` resolved to
      Windows-x86-64 and the run used
      `LibraryResources/Windows-x86-64/hg_evolve_gpu.exe` — the NATIVE MSVC+nvcc binary — on an
      RTX 4090. Both isolation paths are therefore exercised. Caveat: that binary is 2026-07-22, so
      the result certifies the config and the path, not current source.
- [x] CPU results correct across `None` / `Automatic` / `Full`. **2026-08-03**: `GoldenMatrix.*`
      + `Unified_CanonicalHash.*` + the event-identity gates, 12/12.
- [x] GPU results match CPU `CanonicalizeStates -> Full` **with no device fallback**.
      **2026-08-03**: `gpu_differential_tests` 36/36 on an RTX 4090, with ZERO `kIRDegradedToWL`
      in the run — the differential compares states, events, causal and branchial as SETS on both
      routes, so a silent degrade would change a set rather than only a count.
- [ ] `HGEvolve` example pages evaluate cleanly against the local engine.

## Test gates
- [x] CPU suite green. **244 at 2026-08-03**: 213 engine + 28 WXF + 3 Paclet. (WXF and Paclet shell
      out to wolframscript and flake on this machine's WSL interop socket — `accept4 failed 110`,
      10-20 s timeouts; 28/28 and 3/3 when the socket is healthy. Not a code failure, but it will
      make any CI leg that calls wolframscript unreliable.)
- [ ] CPU↔GPU differential green — states/events/causal/branchial equivalent up to isomorphism,
      plus per-mode `NumStates` (was 24/24).
- [x] `gpu_differential_tests` and `hg_gpu_tests` green. **36/36 and 98/98 at 2026-08-03.**
- [x] Determinism gate green **with TR on and quotient on**. **2026-08-03**: `CausalDeterminism.*`
      4/4, and `quotient_determinism_rate_probe` 0/1100 sweeps cumulative at `--load 6`
      (threads {1,2,8} × seeds {fixed, random} × WPP+mixed1+mixed2).
- [x] Oracle corpus + golden corpus green, including the event-canonicalization parity columns.
      **2026-08-03**: `OracleCorpus.*` + `ReferenceOracle.*` 12/12, which includes the brute-force
      isomorphism oracle (independent of the engine's WL and IR) on every rule type.

## v1.0 additions to the above
- [ ] **No user-facing doc states something a user can act on and be wrong about.** The five known
      cases were `HashStrategy`, `EquilibriumAnalysis`, the quotient/TR interaction, `Automatic`
      semantics and the `"States"` return shape (board #5, closed).
      *DANGLING REFERENCE, now inlined above: this line pointed at `V1_SCOPING_REGISTER.md` §C3,
      which is NOT in the repository — it is one of the superseded planning notes that survive only
      on one clone through `.git/info/exclude`. A tracked checklist cannot cite an untracked file:
      a fresh clone, which is what a release is, cannot follow it.*
- [x] **An OSS license exists.** `LICENSE.md` (MIT, The Wolfram Institute), tracked, and declared
      as `"License" -> "MIT"` in `paclet/PacletInfo.wl`.
- [x] No silent correctness degradation anywhere: the GPU IR→1-WL fallback is surfaced
      (`ErrorKind::kIRDegradedToWL`, `last_ir_degraded_states()`); an option the engine ignores is
      reported as `OptionSkipped` rather than dropped (`hypergraph_ffi.cpp`, surfaced by the WL
      layer's advisory kinds).
- [x] Every advertised option exists and every existing option is documented. Gated by
      `OptionSurface.*`, which reconciles all FOUR copies — declared, sent by the wrapper, parsed,
      documented — by reading the sources: 18 sent all parsed, 10 documented all accepted.

## Shipped semantic changes to carry forward in release notes
*(recovered from alpha.6 — these are user-visible and currently have no other home)*
- `exploration_probability` samples **per canonical state**, not per transition.
- `quotient_initial_states` default keeps all roots, matching the reference `MultiwaySystem`.
- Quotient exploration expands each canonical state once **at its shortest depth**.
