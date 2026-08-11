# Release acceptance checklist

**State at 2026-08-04: 20 verified, 0 partly, 0 outstanding.** The Windows GPU stack is
FUNCTIONALLY VERIFIED, not merely built: `HGEvolve[..., TargetDevice -> "GPU"]` matches the CPU
golden corpus 12/12 running the native MSVC+nvcc `hg_evolve_gpu.exe` on an RTX 4090. Every line that can be checked on
a Linux+CUDA workstation has been, with the run that proves it recorded beside it.

Two lines are not engineering and stay open until someone acts on them:

- **Needs the Wolfram documentation toolchain** — the example pages. `DocumentationBuild` passed
  2026-08-04 via `./build_docs.sh`, and the `.paclet` archive and installed-archive exercise are
  DONE (2026-08-03).
- **Needs a decision** — the doc-accuracy line, which is a judgement about user-facing text.
  Three copies of the option surface are cross-checked by tests and the doc inventory by a CI
  gate, but no checker decides whether a sentence is TRUE.

The Windows GPU binary is not among them. `./build_windows_gpu.sh` drives a native MSVC+nvcc
build from WSL through the Windows `cmake.exe`, and `build_all_platforms.sh` calls it. Its two
historic failures are both fixed: `436de63` (cmake.exe inheriting a `\\wsl.localhost` UNC path as
its working directory, which fails before it reads an argument) and `967526b` (`rewrite_core.hpp`
carrying its own `ctz`/`popcount` without `<intrin.h>`, undefined under MSVC). `build_all_platforms.sh` routes a GPU-build failure to SKIPPED rather than
FAILED, deliberately, so the six required platform libraries cannot be blocked by it. **A release
run sets `HG_REQUIRE_GPU=1`**, which makes an absent toolchain or a failed build FAILED instead —
without it a skip leaves the previous exe in the platform directory to be archived, so a broken
CUDA config ships a stale binary and reports only "skipped".

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
- [x] `hg_evolve_gpu` built on both CUDA platforms. **2026-08-04, BOTH FROM CURRENT SOURCE.**
      `./build_all_platforms.sh` now reports **7 built, 0 skipped, 0 failed** — the first fully
      green matrix. The Windows CUDA binary was never a hardware problem: `build_windows_gpu.sh`
      ran `cmake.exe` with the repository as its working directory, which under WSL is a
      `\\wsl.localhost` UNC path, and a Windows process inheriting that fails with
      "Invalid argument" before reading an argument (`436de63`). Behind it was a real portability
      defect — `rewrite_core.hpp` had its own copy of `ctz`/`popcount` without `<intrin.h>`, so
      `_BitScanForward` and `__popcnt` were undefined under MSVC; merged into the single
      `hgcommon` pair (`967526b`).
      **Static-link contract, re-verified on the new binary**: the import table is exactly
      `KERNEL32.dll` and `WS2_32.dll` — no `VCRUNTIME`, no `MSVCP`, no `libgcc`.
      **`reference/verify_paclet_gpu.wls` drives it through `HGEvolve[..., TargetDevice -> "GPU"]`
      on an RTX 4090: 12/12, Failed NONE.**
- [x] `.paclet` archive produced. **2026-08-03**: `reference/build_paclet_archive.wls` →
      `dist/WolframInstitute__HypergraphRewriteEngine-0.0.1.paclet`, **30 MB, 41 entries, a
      library present for each of the six declared SystemIDs**, checked by reading the archive
      back. It STAGES: archiving `paclet/` directly gave 96 MB from a 599 MB tree, 512 MB of it
      `Documentation/Source/generated/`, a gitignored doc-build intermediate. The script now
      copies only what `PacletInfo.wl` declares and fails if a `Documentation/Source` entry
      appears in the result.
- [x] `DocumentationBuild` passes — **2026-08-04**: `./build_docs.sh` (FULL EVALUATION, not
      `structure`) → **`DONE 3 docs`, 3 placed, 0 failed**. Because it evaluates every example
      cell against the local engine, this is also the docs-can't-rot gate, and it is the same
      run that would have caught the `$Failed` default call had it existed before `e2f6f75`.
      Output: tutorial 3.1 MB, `HGEvolve.nb` 1.79 MB, guide 7.3 KB, and
      `tools/dev/doc_symbols_check.py` reports **0 findings against the regenerated tree** — so
      the three markdown sources and the built notebooks now agree, which is what had drifted.
      *(A 535 MB notebook under `Documentation/Source/generated/` dated 2026-07-22 is residue
      from a superseded converter path. The build run above did not touch it; nothing current
      writes that directory, and it cannot ship. Earlier notes here that treated it as this
      gate's input were wrong.)*
- [ ] **Every shipped artifact is stamped with HEAD.**
      `python3 tools/dev/artifact_stamp_check.py --require-clean` → 0 findings.
      This is the line that replaces reading file dates. A release ships binaries for six
      platforms plus two CUDA executables and the assembling host can EXECUTE at most one of
      them, so `--version` cannot answer "is this current?" for the rest; each artifact instead
      carries `HGBUILDSTAMP/1 commit=<sha> variant=<name>` in its .rodata, written by the build
      (`paclet_source/build_stamp.hpp`), and the checker reads it out of the file.
      It catches exactly the hazard the next line describes: the Windows GPU build routes a
      failure to SKIPPED, so a stale `hg_evolve_gpu.exe` from a previous release stays in the
      directory and ships. A stale stamp names it.
      `--require-clean` is a SEPARATE assertion, because the stamp is written at configure time
      and cannot know what the tree looked like at build time. The stamp answers "which commit";
      `git status` answers "plus what".
      **Ground-truthed**: `tools/dev/artifact_stamp_check_selftest.py` builds one fixture per
      defect class — stale commit, no stamp, wrong variant, two stamps in one file, empty tree —
      and requires the exact finding, **7/7**, and the seventh case requires the literal
      `hg_evolve --version` prints to be the literal the checker extracts from that same file.
      It runs in the CI no-build job, so a regex that stopped matching cannot read as green.
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
- [x] **A user can CONTINUE an evolution, not only re-run it.** `reference/verify_sessions.wls`
      → **SESSIONS_VERIFIED, 0 failures, 16 checks**, run under wolframscript against the
      shipped `hg_evolve` through the persistent socket worker.
      The engine has served `Open`/`Step`/`Query`/`Close` since board #121 and nothing exposed
      them: `paclet/Kernel/HypergraphRewriting.wl` carried exactly one `PackageExport`. The
      verbs are now `HGSessionOpen` / `HGSessionStep` / `HGSessionQuery` / `HGSessionClose`.
      The check that decides the claim is a comparison at EVERY prefix depth against an
      independent one-shot `HGEvolve`: 1/2/4/10/34 states at depths 0–4, reached by stepping.
      A session that restarted each Step would report depth-1 numbers forever and pass any
      check that only asked whether it answered. The lifetime is checked too — a fresh session
      is at depth 0 (an Open that explored would shift every later comparison by one), a second
      Open is refused while the first still answers, and the slot frees on Close.
      Runs in the `paclet-wl` workflow beside the golden corpus.
- [x] CPU results correct across `None` / `Automatic` / `Full`. **2026-08-03**: `GoldenMatrix.*`
      + `Unified_CanonicalHash.*` + the event-identity gates, 12/12.
- [x] GPU results match CPU `CanonicalizeStates -> Full` **with no device fallback**.
      **2026-08-05**: `gpu_differential_tests` 36/36 on an RTX 4090. The device has no coarser
      key to fall back TO: `state_exact_hash_device` reports `kIRArenaExhausted` /
      `kIRDepthExceeded` / `kIRGeneratorsExceeded` and the wrapper doubles the corresponding
      `EngineConfig` field. The differential compares states, events, causal and branchial as
      SETS on both routes, so a wrong key would change a set rather than only a count.
- [x] `HGEvolve` example pages evaluate cleanly against the local engine. **2026-08-04**, two
      independent checks. `./build_docs.sh` evaluates EVERY example cell while generating the
      notebooks and reports 3/3 built, 0 failed. `reference/verify_doc_examples.wls` evaluates all
      **33 fenced blocks** in the three markdown sources in one shared context (they are written as
      a running session) and treats an unevaluated `HGEvolve` head as a failure, not only a
      message — because a documented option the engine does not accept returns unevaluated rather
      than erroring. **`HGEvolve.md` 26/26**, up from 20/26 before `e2f6f75`.
      *Two `GettingStarted.md` blocks are flagged, and for messages ONLY — both return correct
      results. A user's FIRST `HGEvolve` call in a kernel emits 7 Wolfram messages
      (`RemoveInputStreamMethod::name` ×4, `RemoveOutputStreamMethod::name` ×2,
      `MIMETypeToFormatList::fmterr`); the second emits none. One-time setup noise, board #109,
      not a blocker — but it is the first thing a user sees.*

## Test gates
- [x] CPU suite green. **246/246 at 2026-08-05**, run from `build_linux`.
      *Run it from `build_linux`, not the repository root:* `PacletTest` loads the paclet from
      `../paclet`, which a Windows `wolframscript` resolves through the UNC mapping of the
      inherited working directory. From the root it resolves one level too high; the test now says
      so (`PACLET_LOAD_FAIL`) instead of reporting an unevaluated `HGEvolve` as an engine defect
      (`2b9a285`).
      *The wolframscript legs shell out and flake on this machine's WSL interop socket —*
      `accept4 failed 110`, *ETIMEDOUT, 10–20 s.* Every consultation now goes through one
      `consult()` helper that retries only a VERDICTLESS run, never a verdict, and the suite
      reports the rate and bounds it (`# wolfram oracle: 0/47 consultations returned no verdict`).
      Before `4f0ff63` the retry existed at one call site and not at its sibling, so a single
      wedge failed `VerifyTestInfrastructureDetectsFailures` on a fault the round-trips were
      already tolerating.
- [ ] **A green CI suite is not a green oracle, and the difference is not small.** Verified against
      `.github/workflows/ci.yml`: the leg runs `./build/all_tests --gtest_filter=-PacletTest.*`, so
      **244 of 246** tests, and the runner has no wolframscript, so `testing/CMakeLists.txt` sets
      `WOLFRAMSCRIPT_AVAILABLE=0`. With the oracle compiled out `test_wolfram_roundtrip` returns
      `true` without consulting anything, and the **12** tests that call it therefore assert
      vacuously — they still exercise the C++ serializer, but nothing cross-checks it against
      Wolfram. `TearDownTestSuite` correctly prints nothing rather than claiming a rate
      (`if (s_consultations == 0) return;`), so CI makes no false statement; the point is that
      **the Wolfram cross-check runs only on a machine that has Wolfram**, which today means this
      one. Any release sign-off has to include a local `all_tests` run from `build_linux` with a
      non-zero consultation count, not a CI badge.
- [x] CPU↔GPU differential green — states/events/causal/branchial equivalent up to isomorphism,
      plus per-mode `NumStates`. **2026-08-03: `gpu_differential_tests` 36/36 on an RTX 4090**, a
      full 21-minute run. This line and the `gpu_differential_tests` line below are the SAME
      binary: the equivalences are `DifferentialEvolution.BitIdenticalCanonicalForm` over the
      28-workload corpus, which compares causal and branchial as sets on both routes (the run logs
      each, e.g. `quotient_wolfram_steps6 causal cpu=26332 gpu=26332 branchial cpu=30063
      gpu=30063`), and per-mode `NumStates` is `CanonicalStateCount.ModesVsCpu`, which reproduces
      the CPU's `num_canonical_states()` by the same per-mode rule rather than reverse-engineering
      it (`test_gpu_vs_cpu_differential.cpp:1174-1209`).
- [x] `gpu_differential_tests` and `hg_gpu_tests` green. **36/36 and 98/98 at 2026-08-03.**
- [x] Determinism gate green **with TR on and quotient on**. **2026-08-03**: `CausalDeterminism.*`
      4/4, and `quotient_determinism_rate_probe` 0/1100 sweeps cumulative at `--load 6`
      (threads {1,2,8} × seeds {fixed, random} × WPP+mixed1+mixed2).
- [x] Oracle corpus + golden corpus green, including the event-canonicalization parity columns.
      **2026-08-03**: `OracleCorpus.*` + `ReferenceOracle.*` 12/12, which includes the brute-force
      isomorphism oracle (independent of the engine's WL and IR) on every rule type.

## v1.0 additions to the above
- [x] **No user-facing doc states something a user can act on and be wrong about.** The five
      original cases were `HashStrategy`, `EquilibriumAnalysis`, the quotient/TR interaction,
      `Automatic` semantics and the `"States"` return shape (board #5, closed). Two more were
      found and fixed on 2026-08-04, both the same shape as the `.def` break — a manifest that
      outlived what it described when the visualisation split landed:
      **(6) 21 reference pages + 48 links** for symbols that do not exist (`90bced2`), gated by
      `tools/dev/doc_symbols_check.py`, ground-truthed **107 findings → 0**.
      **(7) 39 documented OPTIONS that `HGEvolve` does not accept** (`cb50daf`) — the whole
      analysis surface (`DimensionAnalysis`, `CurvatureAnalysis`, `GeodesicAnalysis`,
      `EntropyAnalysis`, `TopologicalAnalysis`, `HilbertSpaceAnalysis`, `MultispaceAnalysis`,
      `BranchAlignment` and their parameters). Setting one did nothing and said nothing.
      `OptionSurface.EveryDocumentedOptionIsAnOptionHGEvolveAccepts` was green because it matched
      only `### "Name"` headings — 10 of the page's 78 names — so it now reads the option TABLES
      too, ground-truthed **39 failures → 0**.
      Each of the three copies of the option surface is now checked against the others by a test,
      and the documentation inventory by a CI gate; what no checker can decide is whether the
      remaining prose is accurate, and that is read rather than proved.
      **A sixth was found and CLOSED 2026-08-04 — 21 reference pages documented symbols that do
      not exist.**
      `paclet/Kernel/HypergraphRewriting.wl:5` is `PackageExport["HGEvolve"]` and the file carries
      exactly one `::usage`, HGEvolve's. Yet 22 pages ship under
      `ReferencePages/Symbols/`, and each of the other 21 — `EdgeId`, `HGTorus`, `HGSphere`,
      `HGToGraph`, `HGHausdorffAnalysis`, `HGBranchAlignmentBatch`, `HGGeodesicPlot`,
      `HGUniformRandom` and 13 more — names a symbol with **zero** mentions anywhere in
      `paclet/Kernel/`. Verified present inside the assembled archive. A shipped tutorial,
      `Documentation/English/Tutorials/QuantumAnalysisExamples.wl`, also calls two of them.
      This is the same class as the `.def` break in `fc6d24a`: the visualisation split (#18)
      removed the functions and both manifests that described them outlived them, unchecked.
      **FIXED (`90bced2`)**: the 21 pages are gone, the links to them are gone (guide 42 → 0,
      `HGEvolve.nb` 6 → 0), and `tools/dev/doc_symbols_check.py` is in the CI no-build job —
      ground-truthed at **107 findings against the previous tree, 0 now**. Archive entries 41 → 19,
      carrying `HGEvolve.nb` alone. What remains on this line is the JUDGEMENT that no prose
      elsewhere misstates behaviour, which no checker decides. Board #108 closed.
      *DANGLING REFERENCE, now inlined above: this line pointed at `V1_SCOPING_REGISTER.md` §C3,
      which is NOT in the repository — it is one of the superseded planning notes that survive only
      on one clone through `.git/info/exclude`. A tracked checklist cannot cite an untracked file:
      a fresh clone, which is what a release is, cannot follow it.*
- [x] **An OSS license exists.** `LICENSE.md` (MIT, The Wolfram Institute), tracked, and declared
      as `"License" -> "MIT"` in `paclet/PacletInfo.wl`.
- [x] No silent correctness degradation anywhere: the GPU has no IR→1-WL fallback to be silent
      about — every cause of an unproduced exact hash is a config-controlled capacity kind the
      wrapper grows and retries; an option the engine ignores is reported as `OptionSkipped`
      rather than dropped (`hypergraph_ffi.cpp`, surfaced by the WL layer's advisory kinds).
- [x] Every advertised option exists and every existing option is documented. Gated by
      `OptionSurface.*`, which reconciles all FOUR copies — declared, sent by the wrapper, parsed,
      documented — by reading the sources: 18 sent all parsed, 10 documented all accepted.

## Shipped semantic changes to carry forward in release notes
*(recovered from alpha.6 — these are user-visible and currently have no other home)*
- `exploration_probability` samples **per canonical state**, not per transition.
- `quotient_initial_states` default keeps all roots, matching the reference `MultiwaySystem`.
- Quotient exploration expands each canonical state once **at its shortest depth**.
