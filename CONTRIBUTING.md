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
cmake --build build --target all_tests
./build/all_tests
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

User docs are authored in markdown under `paclet/Documentation/Source/` and converted
to paclet notebooks with `tools/build_docs.wls` (see the script header). The user-facing
quickstart is [docs/QUICKSTART.md](docs/QUICKSTART.md).
