# hgevolve

hgevolve runs the HypergraphRewriting engine from Python. `hg.evolve` is `HGEvolve`: it takes the same rules, initial states, steps and options, and it returns the states, events, causal and branchial relations, counts and step statistics as NumPy arrays. The engine runs in the Python process, with no subprocess and no WXF between Python and the engine.

```python
import hgevolve as hg

rule = ([["x", "y"], ["x", "z"]], [["x", "y"], ["x", "w"], ["y", "w"], ["z", "w"]])
r = hg.evolve(rule, [[1, 2], [1, 3]], 3, canonicalize_states="Full")

r.num_states                  # 10
r.states.step                 # int64 array, one entry per state
r.states.edges[0]             # [(edge id, vertex array), ...] of the first state
r.events.consumed_edges       # Ragged: values and offsets
r.causal_edges.array          # n x 4 int64 array: From, To, RawFrom, RawTo
```

docs/API.md lists every function, option and data name with its HGEvolve name.

## Building

The package is built from this repository with CMake. It needs Python 3.9 or later with its development headers, NumPy and nanobind (`pip install nanobind`).

```sh
pip install ./python                      # scikit-build-core builds the module and installs it
```

or, for a build tree:

```sh
cmake -S . -B build_python -DCMAKE_BUILD_TYPE=Release -DHG_BUILD_PYTHON=ON -DHG_ENGINE_STATS=OFF
cmake --build build_python --target _core -j4
PYTHONPATH=python python3 -c "import hgevolve"
```

`HG_BUILD_PYTHON` compiles the module (`hgevolve._core`) from the paclet's job code, `paclet_source/hypergraph_ffi.cpp` and `paclet_source/paclet_support.cpp`, and links the engine library. In a build tree the module is written into `python/hgevolve/`.

## How a call runs

`evolve` builds the job HGEvolve would send (its association of rules, initial states, steps and options) as Python objects. The module turns it into an in-memory `wxf::WXFValue`, and the engine reads it with the same `parse_job` that reads the paclet's WXF bytes, through `wxf::ValueParser`. The job then runs through the same code as in `hg_evolve`. Only the reply's form differs: `ArrayReply` (`paclet_source/reply_out.hpp`) takes the record and relation calls that the paclet's reply writes as WXF, and stores them as columns. The option rules and the job format therefore exist once, in the engine.

Every array of a `Result` is a read-only view of that reply, and nothing is copied. Each array holds a reference to the reply, so the reply lives as long as any of its arrays, after the `Result` and its `Session` are gone. Ragged data (each state's edges, each event's consumed and produced edges) comes as CSR pairs: a flat values array and an offsets array.

## Crashes, memory and threads

- **An engine crash ends the Python process.** The engine runs in the interpreter's process, so a segmentation fault or abort in the engine takes the interpreter with it, and no exception is raised. Work that may crash, or that should be stoppable from outside, belongs in a child process (`examples/parallel_sweep.py`).
- A job the engine refuses raises `hg.EngineError` with the engine's message, and the reply's warnings are issued as `hg.EngineWarning`.
- The engine's host memory limit applies in the Python process as in `hg_evolve`: `tuning={"MemoryLimit": bytes}` or `HG_TUNE_MEMORY_LIMIT`. An evolution that reaches it is truncated with a "memory exhausted" `EngineWarning`, and the reply holds what was built (`tests/test_memory_limit.py`). An address-space cap (`resource.setrlimit(resource.RLIMIT_AS, ...)` or `ulimit -v`) bounds the whole process in the same way. Under such a cap on a many-core machine, set `OPENBLAS_NUM_THREADS` before NumPy is imported, or NumPy's import stalls.
- The engine uses `threads` worker threads, by default `HG_TUNE_THREADS` or one per logical CPU. One job runs at a time per process, with the GIL released while it runs. A process has at most one open `Session`.

## Examples

| Script | Shows |
|---|---|
| `examples/multiway.py` | a multiway evolution and its arrays |
| `examples/sampled_history.py` | one sampled history from a 30 x 30 grid (`uniform_random`, `matches_per_step`) |
| `examples/brill_lindquist_history.py` | a sampled history from a Brill-Lindquist initial condition exported from the paclet |
| `examples/generational_and_orderings.py` | `generational` and `event_ordering_function` |
| `examples/undirected_edges.py` | `undirected_edges` |
| `examples/step_statistics.py` | step statistics with measures and branchial metrics |
| `examples/parallel_sweep.py` | a sweep across processes, each under an address-space cap |

## Tests

```sh
cd python && python3 -m pytest
```

`tests/test_parity.py` runs every job that the examples of docs/en/ReferencePages/Symbols/HGEvolve.md send through HGEvolve: 234 distinct jobs. Each is run from the rules and initial states the example gave HGEvolve, and the result is compared with the reply the paclet got for the same job (`tests/fixtures/HGEvolve.json.gz`). The comparison covers states, events, causal and branchial edges, global edges, state bitvectors, counts and step statistics. All 234 match exactly, ids included, with one engine thread. The test also accepts a match up to the run-local ids of docs/SPEC.md (states by their canonical hash, events and pairs through the states they join). The same comparison fails for 224 of the 234 jobs when the Python call runs one step further. For the other 10, one more step gives the same reply. One example job is not a fixture. `HGEvolve[chain, {{1, 2}}, 20, ..., "ExploreFromCanonicalStatesOnly" -> True]` asks for counts only, and with every list requested it unfolds every raw event of 20 steps.

`tests/generate_fixtures.wls` writes the fixtures in two steps. The first evaluates the page with the engine call replaced, so no engine runs. `tests/measure_fixture_jobs.py` then runs every captured job with `hg_evolve` under a memory cap, and only the measured jobs go to the paclet.

`tests/test_views.py` checks that the arrays are views of the reply (the same data pointer as the reply's buffer, not owning their data, read-only), and that they keep their values after their `Result` and their `Session` are deleted.

## Performance

Three HGEvolve calls, measured by counters only (the machine was shared, so there is no wall-clock timing):

- **multiway**: `HGEvolve[wolfram, {{1, 2}, {1, 3}}, 4]` with states, events, causal and branchial edges and the counts. 127 states, 126 events.
- **Brill-Lindquist history**: `HGEvolve[wolfram, "BrillLindquist", 10, "SprinklingDensity" -> 1600, "RandomSeed" -> 1, "UniformRandom" -> True, "MatchesPerStep" -> 1]` with states, events, causal edges and counts. The initial state has 1600 vertices and 23,409 edges.
- **statistics**: `HGEvolve[wolfram, {{1, 2}, {1, 3}}, 5, "StepStatistics", "CanonicalizeStates" -> Full, "StepStatisticsMeasures" -> {"VolumeGrowth", "Geodesics"}]`. 302 states.

Here `wolfram` is `{{1, 2}, {1, 3}} -> {{1, 2}, {1, 4}, {2, 4}, {3, 4}}`.

| Call | WXF bytes sent | WXF bytes received | Expressions built from the reply | hg_evolve instructions | hgevolve instructions |
|---|---:|---:|---:|---:|---:|
| multiway | 1,195 | 58,857 | 8,675 | 3,272,771 | 3,979,405 |
| Brill-Lindquist history | 325,184 | 4,326,514 | 1,030,680 | 1,675,602,872 | 1,829,964,614 |
| statistics | 1,145 | 52,239 | 3,524 | 91,267,008 | 96,585,833 |

Method:

- The WXF bytes and expressions are from the paclet. Each call ran through HGEvolve with the job and reply recorded at `hgSendJob`. "Sent" is the length of `BinarySerialize` of the job, "received" is the reply's length, and the expression count is `LeafCount` of the decoded reply. HGEvolve's result for these properties is the decoded reply itself.
- The instruction counts are callgrind `Ir` totals over every thread, with one engine thread. "hg_evolve" is `hg_evolve` reading the job the paclet sent on stdin. Its reply is byte-identical to the reply the paclet received. "hgevolve" is a Python process making the same call with `hg.evolve`, including the conversion of the Python arguments and the construction of the `Result`. Each count has a trivial job (Steps 0, one edge) run the same way subtracted: 3,591,469 for `hg_evolve` and 470,199,770 for the Python process (interpreter start, NumPy and the module).
- The binaries are Release builds with `HG_ENGINE_STATS=OFF`, GCC 15.2, link-time optimisation, from the same sources, on the same machine.

Most of the difference for the Brill-Lindquist call lies outside the engine. 37.8e6 instructions turn the Python edge lists into the job value (`to_value`), and 35.6e6 are `__tls_get_addr`: the engine's `thread_local` scratch, which a shared object reaches through that call and an executable reaches directly. TLS descriptors (`-mtls-dialect=gnu2`) were tried for the module and refused: the module then requires `GLIBC_ABI_GNU2_TLS`, which the glibc of the measuring machine (2.39) does not provide.
