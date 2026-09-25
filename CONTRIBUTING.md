# Contributing

Thanks for working on the Hypergraph Rewriting Engine. This is the developer
getting-started; for the system map see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Prerequisites

- A C++20 compiler (GCC 10+, Clang 12+, or MSVC 19.29+)
- CMake 3.14+
- Optionally: the CUDA toolkit (for the GPU engine), and a Wolfram Engine /
  Mathematica 13.0+ (for the paclet and the paclet tests)

## Build

```bash
cmake -B build -DBUILD_WOLFRAM_LANGUAGE_PACLET=ON
cmake --build build -j
```

Add `-DBUILD_GPU=ON` for the CUDA engine. Cross-compilation targets (Windows, macOS,
Linux, and the GPU binary) are described in
[CROSS_COMPILATION.md](CROSS_COMPILATION.md).

## Test

The C++ suites gate correctness; run them before opening a change.

```bash
cmake --build build
ctest --test-dir build -j4 --output-on-failure
```

RUN CTEST, NOT `all_tests` ALONE. `all_tests` is the aggregate of the engine suites and is not
the whole gate: the job system and the two deques build their own binaries, so a change to
`job_system/` or `lockfree_deque/` is not covered by running `all_tests` and will look green
while its own suites were never executed. ctest runs every binary.

For a fast loop while working, the aggregate and the subset targets are still the right thing:

```bash
cmake --build build --target all_tests && ./build/all_tests --gtest_filter='Causal*'
```

GPU (when built with `-DBUILD_GPU=ON`):

```bash
./build/gpu_differential_tests   # CPU-vs-GPU exact differential
./build/hg_gpu_tests             # GPU unit tests
```

Paclet (needs a Wolfram Engine): `reference/verify_paclet.wls` loads the local paclet
and checks `HGEvolve` against the golden corpus.

```bash
wolframscript -file reference/verify_paclet.wls
```

The `ReferenceOracle` test in `all_tests` compares canonical-state counts to a
brute-force isomorphism oracle — it is the decisive correctness check for
canonicalization work.

## Conventions

- The engine is lock-free and performance-first on the hot path. Fixes stay lock-free;
  no mutexes, and no `std::` heap containers on the hot path (use the arenas).
- The GPU mirrors the CPU algorithms — don't drop a CPU data structure in a kernel
  without a measured reason, and re-run `gpu_differential_tests`.
- Measure before optimizing. Profiling harnesses live in `tools/`
  (`profile_evolve.cpp` for the CPU, `bench_gpu_evolve.cpp` for the GPU); cachegrind
  and `ncu` counters are reliable even on a noisy machine where wall-clock is not.
- Comments describe the current invariant, not the history — no "previously / used to
  / replaces the old" framing.

## Documentation

The user documentation is written in markdown under `docs/en/` and converted into the paclet's
notebooks under `paclet/Documentation/English/` by `./build_docs.sh`, which runs
`tools/build_docs.wls` with the converter in the `tools/MarkdownToNotebook` submodule. The
notebooks are tracked, because the paclet ships them; edit the markdown, never a notebook. The
quickstart for users is [docs/QUICKSTART.md](docs/QUICKSTART.md).

### Layout

```
docs/en/ReferencePages/Symbols/<Name>.md   Template: Symbol    -> English/ReferencePages/Symbols/<Name>.nb
docs/en/Guides/<Name>.md                   Template: Guide     -> English/Guides/<Name>.nb
docs/en/Tutorials/<Name>.md                Template: TechNote  -> English/Tutorials/<Name>.nb
```

A page's frontmatter has `Name`, equal to its file name; `Title` on guides and tutorials;
`` Context: HypergraphRewriting` ``; `Paclet: WolframInstitute/HypergraphRewriteEngine`; and
`URI: WolframInstitute/HypergraphRewriteEngine/{ref|guide|tutorial}/<Name>`. The build stops when
a `Name` differs from its file name. Only the seven public symbols (`HGEvolve` and the
`HGSession*` functions) have reference pages; `tools/dev/doc_symbols_check.py` checks this.

### Examples

Every example is evaluated during the build against the paclet in `paclet/`. A reference page or
the guide resets its definitions at every heading and every `---`, so each group of examples
defines the rules it uses. A tutorial evaluates as one document.

A comment after an example records its result, and the example gate compares it:

```
<!-- => {1, 2, 4, 10} -->
<!-- => 7; the message HGEvolve::warn is issued -->
<!-- => $Failed; the message HGEvolve::badrule is issued -->
```

Text after `; ` is a note. A block may issue only the messages its comment names.

### Building

```
./build_docs.sh                 # evaluate and convert every changed page
./build_docs.sh only=<regex>    # only the pages whose file name matches
./build_docs.sh structure       # input cells only, into docs/en/.generated/ (not tracked)
```

The build needs `wolframscript` (native, or the Windows install used from WSL) and the engine
built for the platform the kernel runs on. From WSL the kernel is the Windows one, which loads
`paclet/LibraryResources/Windows-x86-64/`; build those binaries with
`./build_windows_msvc.sh cpu`, since MinGW builds corrupt the heap at worker-thread exit. A page
is rebuilt when its markdown, the engine binaries, the kernel files, the converter or the build
script change; a notebook that no page maps to is deleted.

### Checks

Run these before committing a page; `.github/workflows/docs.yml` runs the ones that need no
Wolfram kernel on every push that changes documentation.

| Check | What it checks |
|---|---|
| `reference/verify_doc_examples.wls` | every example evaluates without an unstated message and matches its recorded result |
| `tools/dev/doc_messages_check.py` | no built notebook shows a message its page does not state |
| `tools/dev/lint_docs.py` | markdown lint of the pages and the other tracked markdown |
| `tools/dev/docs_fresh_check.py` | no notebook was last changed in an older commit than its markdown |
| `tools/dev/doc_symbols_check.py` | reference pages exist for the public symbols and only for them |
| `tools/dev/doc_surface_audit.py`, `tools/dev/option_names_check.py` | the documented options and properties are the ones `HGEvolve` has |
