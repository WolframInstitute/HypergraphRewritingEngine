# Release acceptance checklist

**State at 2026-08-03: 15 verified, 1 partly, 4 outstanding.** The Windows GPU stack is
FUNCTIONALLY VERIFIED, not merely built: `HGEvolve[..., TargetDevice -> "GPU"]` matches the CPU
golden corpus 12/12 running the native MSVC+nvcc `hg_evolve_gpu.exe` on an RTX 4090. Every line that can be checked on
a Linux+CUDA workstation has been, with the run that proves it recorded beside it. The 9 that
remain — 4 partly done, 5 not — split into exactly two kinds, and neither is new engineering:

- **Needs the Wolfram documentation toolchain** — `DocumentationBuild` and the example pages.
  The `.paclet` archive and the installed-archive exercise are DONE (2026-08-03), and the doc
  build has an open finding of its own recorded on its line.
- **Needs a Windows host** — only `Windows-x86-64/hg_evolve_gpu.exe`, which requires native
  MSVC+nvcc. The cross attempt failed on the WSL interop socket, not on code, and the config
  itself is proven: that binary's predecessor passes the golden corpus 12/12.
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
- [x] All 6 platform libraries built (Linux/Windows/macOS × x86-64/ARM64). **REBUILT FROM CURRENT
      SOURCE 2026-08-03**: `./build_all_platforms.sh` → 6 built, 0 failed. Getting there needed two
      real fixes, because four of the six DID NOT BUILD: the Windows `.def` exported three
      functions the viz split deleted (`fc6d24a`), and macOS could not compile `park.hpp` since
      both its wait primitives were out of reach on the 12.3 SDK at a 10.15 deployment target
      (`29c9345`). Neither was visible to any gate — every gate here runs on Linux, and a Linux
      `.so` links with undefined symbols and says nothing.
- [x] All 6 `hg_evolve` process binaries built. Same run, all six dated 2026-08-03.
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
- [x] `.paclet` archive produced. **2026-08-03**: `reference/build_paclet_archive.wls` →
      `dist/WolframInstitute__HypergraphRewriteEngine-0.0.1.paclet`, **30 MB, 41 entries, a
      library present for each of the six declared SystemIDs**, checked by reading the archive
      back. It STAGES: archiving `paclet/` directly gave 96 MB from a 599 MB tree, 512 MB of it
      `Documentation/Source/generated/`, a gitignored doc-build intermediate. The script now
      copies only what `PacletInfo.wl` declares and fails if a `Documentation/Source` entry
      appears in the result.
- [ ] `DocumentationBuild` passes (was 24/24) — note this **evaluates every example cell**, so it is
      also the docs-can't-rot gate. **KNOWN FINDING, 2026-08-03**: the last build left
      `Documentation/Source/generated/Tutorials/Getting Started with Hypergraph Rewriting.nb` at
      **535 MB / 8.47 M lines**. The cause is in the notebook's own content, not the doc
      toolchain: the graph it embeds annotates EVERY edge with the full `InputStateEdges` and
      `OutputStateEdges` lists of its endpoint states, so the payload grows as events × state
      size. It does not reach a user — the archive ships `Documentation/English` and the staging
      check now fails if `Documentation/Source` appears — but a tutorial that emits a
      half-gigabyte cell is not a tutorial that can be maintained, and this line stays open on it.
- [x] **Static-link contract holds.** Verified 2026-08-03 on
      `paclet/LibraryResources/Windows-x86-64/hg_evolve_gpu.exe`: the import set is exactly
      `KERNEL32.dll`, `WS2_32.dll`, `nvcuda.dll` — nothing else — so `libcudart_static` + `/MT`
      held. Re-verify after the rebuild, since the binary checked is the 2026-07-22 one.

## Functional verification
- [x] The assembled `.paclet` is installed and exercised via wolframscript. **2026-08-03**:
      `reference/verify_paclet.wls --archive` `PacletInstall`s `dist/*.paclet` and runs the golden
      corpus against the INSTALLED copy — **12/12, Failed NONE**, and TargetDevice CPU == GPU ==
      `{5,33,32,43}`. Because wolframscript here is the Windows executable, `$SystemID` resolved to
      `Windows-x86-64`, so the library exercised was the archive's WINDOWS one. This is the check a
      directory load cannot make: the archive is a staged SUBSET, so a dropped file or a platform
      library unfindable by SystemID would show up only here. It uninstalls afterwards.
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
      **NEW AND LARGER, 2026-08-03 — 21 REFERENCE PAGES DOCUMENT SYMBOLS THAT DO NOT EXIST.**
      `paclet/Kernel/HypergraphRewriting.wl:5` is `PackageExport["HGEvolve"]` and the file carries
      exactly one `::usage`, HGEvolve's. Yet 22 pages ship under
      `ReferencePages/Symbols/`, and each of the other 21 — `EdgeId`, `HGTorus`, `HGSphere`,
      `HGToGraph`, `HGHausdorffAnalysis`, `HGBranchAlignmentBatch`, `HGGeodesicPlot`,
      `HGUniformRandom` and 13 more — names a symbol with **zero** mentions anywhere in
      `paclet/Kernel/`. Verified present inside the assembled archive. A shipped tutorial,
      `Documentation/English/Tutorials/QuantumAnalysisExamples.wl`, also calls two of them.
      This is the same class as the `.def` break in `fc6d24a`: the visualisation split (#18)
      removed the functions and both manifests that described them outlived them, unchecked.
      **Disposition is Richard's** — these belong to the split-out visualisation paclet, so either
      they are deleted here or that repo carries them. Board #108.
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
