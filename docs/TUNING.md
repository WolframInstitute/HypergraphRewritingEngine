# Tuning constants

This document lists every fixed tuning constant in the engine, on the host and on the device, and states for each one what it trades, which hardware property it depends on, and how it is set. The first part describes the code as it is at master 0abd13e6. The second part, marked **Proposed**, gives the derivation, the expert knob and the measurement for each constant, and the design for C1 (active worker count scaled to the available work).

The rule for every setting: a default derived from the machine, a knob for experts, and a per-device table entry only where a value was measured and cannot be derived.

Constants that are not tuning are listed at the end with the reason: sentinels, algorithm limits, pathology ceilings, sizes set only for the model checker, and stats-build storage.

## 1. What exists today

### 1.1 Hardware queries

| Query | Site | Consumers |
|---|---|---|
| `hgcommon::performance_cpus()` (P-cores on Intel hybrid Linux via `/sys/devices/cpu_core/cpus`, big cluster on ARM via `cpu_capacity`, highest `EfficiencyClass` on Windows; one CPU per physical core; empty on homogeneous parts and under WSL) | `job_system/src/affinity.cpp:196` | `JobSystem::ensure_default_cpu_order` |
| `hgcommon::cache_domains_of(cpus)` (deepest cache level with an `id` in sysfs; `RelationCache` on Windows) | `common/include/hgcommon/affinity.hpp:121`, `job_system/src/affinity.cpp` | `ensure_default_cpu_order`, `build_domain_map`, `build_cache_peers` |
| `hgcommon::allowed_cpus()` (`sched_getaffinity`, `GetProcessAffinityMask`) | `affinity.hpp:137` | `ensure_default_cpu_order` |
| `std::thread::hardware_concurrency()` | `hypergraph/src/parallel_evolution.cpp:61`, `paclet_source/cpu_engine_holder.hpp:41`, `job_system/include/job_system/job_system.hpp:796`, `paclet_source/paclet_support.cpp:870` | default worker count; branchial pass threads |
| `hg_gpu::probe_device()`: device count, driver version, name, compute capability, whether a kernel image loads; cached | `gpu/src/device_probe.cu` | `hg_evolve_main.cpp:389`, `hg_gpu_backend.cpp:869` |
| `cudaDeviceGetAttribute(MultiProcessorCount)` | `gpu/src/persistent.cu:1714` | `default_persistent_grid` |
| `cudaDeviceGetAttribute(MultiProcessorCount, MaxThreadsPerMultiProcessor)` | `gpu/src/persistent.cu:1752` | `device_resident_threads` (stack reservation estimate) |
| `cudaFuncGetAttributes(...).localSizeBytes` | `persistent.cu:1736`, `match_kernels_stack_bytes` | `EngineState::device_stack_bytes` (`gpu/src/engine_state.cu:327`) |
| `cudaMemGetInfo` total | `gpu/src/evolve.cu:983` `resolved_memory_cap` | memory cap, 90% of total VRAM |

`probe_device` already calls `cudaGetDeviceProperties` (`device_probe.cu:65`) and keeps only the name and the compute capability. Nothing in `gpu/` reads the L2 size, shared memory per SM, registers per SM or the maximum blocks per SM. Nothing on the host reads cache sizes, SMT sibling groups, the huge page size or available memory.

Placement today: `ensure_default_cpu_order` pins workers only when the allowed CPUs span more than one last-level cache (`job_system.hpp:434`, `if (!many) return`). On an i9-14900K (one L3) and under WSL nothing is pinned and the operating system places every worker.

### 1.2 Existing expert knobs

| Knob | Kind | Effect |
|---|---|---|
| `HG_GPU_PERSISTENT_BLOCKS` | environment | persistent grid (`persistent.cu:1709`) |
| `EvolveInput::max_blocks_per_launch`, `slice_scan_max_edges`, `max_device_memory_bytes` | C++ API | launch chunking, slice scan bound, memory cap |
| `num_threads` of `ParallelEvolutionEngine` / `JobSystem` | C++ API | worker count |
| `JobSystem::set_worker_cpus`, `set_worker_cache_domains` | C++ API | placement |
| `HG_CONCURRENT_MAP_INITIAL_CAPACITY`, `HG_KEY_SET_PROBE_LIMIT`, `HG_KEY_SET_SHARDS`, `HG_JOB_QUEUE_CAPACITY`, `HG_JOB_INJECTOR_CAPACITY`, `HG_JOB_CHUNK_SLOTS`, `HG_JOB_MAX_POOLS`, `HG_MAX_ARENA_WORKERS`, `HG_SEGMENTED_ARRAY_MAX_SEGMENTS` | compile-time macro | sizes; most exist so the model checker can shrink them |

The paclet has no worker-count option: `CpuEngineHolder` is built with `hardware_concurrency()` (`hypergraph_ffi.cpp:944`).

### 1.3 Host constants

| # | Constant | Value | Site | Trade |
|---|---|---|---|---|
| H1 | default worker count | `hardware_concurrency()` | `parallel_evolution.cpp:61`, `cpu_engine_holder.hpp:41`, `job_system.hpp:796` | parallelism against park/wake churn, steal traffic, SMT and E-core sharing (C1) |
| H2 | wake policy | wake one parked worker on every submit while any worker is idle | `job_system.hpp:351`, `common/include/hgcommon/park_gate.hpp:92` | wake latency against wakes that find nothing |
| H3 | steal attempts per search | near peers once each, then `n` random victims | `job_system.hpp:553-575` | finding work against reads of other workers' deque lines |
| H4 | `kExpandChunkSize` | 16 matches per task | `hypergraph/include/hypergraph/parallel_evolution.hpp:1237` | spread of a wide state against per-task cost |
| H5 | causal set working capacity (`ConcurrentKeySet::WORKING_CAPACITY`) | 1024 per shard | `hypergraph/include/hypergraph/concurrent_key_set.hpp:99`; sets at `causal_graph.hpp:133,137`, `hypergraph.hpp:309` | table chain length (paid on every insert) against fixed footprint; 8192 measured -2.5 to -4.7% time at one thread, +5.7 MB on small runs (board H3) |
| H6 | `HG_CONCURRENT_MAP_INITIAL_CAPACITY` | 1024 entries (16 KB), allocated on first insert | `concurrent_map.hpp:82` | chain length against footprint of the 17 engine maps |
| H7 | `LOAD_FACTOR_THRESHOLD` | 0.75 | `concurrent_map.hpp:101`, `concurrent_key_set.hpp:87` | probe length against table bytes |
| H8 | `HG_KEY_SET_PROBE_LIMIT` | 32 | `concurrent_key_set.hpp:414` | grow trigger; probe length against capacity |
| H9 | `HG_KEY_SET_SHARDS` | 64 | `concurrent_key_set.hpp:651` | contention on one table against object size |
| H10 | `kInstShards` | 8 | `hypergraph.hpp:275` | contention on one class's instance list against 128 B per entry |
| H11 | `kIdBlock`, `kEventIdBlock` | 64 | `hypergraph.hpp:350`, `:383` | shared-counter refills against unused ids (id space up to `QR_ID_LIMIT`) |
| H12 | `QcPairChunk::kCap` | 252 pairs (2 KB chunk) | `hypergraph.hpp:450` | chunk allocations against tail waste per worker |
| H13 | `HG_MAX_ARENA_WORKERS` | 256 | `hypergraph/include/hypergraph/arena.hpp:59` | registry size; a thread past 256 uses the shared slot (`arena.hpp:125`) |
| H14 | arena block | 2 MB (`kHugePageBytes`) | `arena.hpp:184` | one huge page per block against unused tail |
| H15 | arena first block | 64 KB, doubling to the block size | `arena.hpp:205` | malloc count against bytes for a light arena |
| H16 | cache line | 64 B: `kLine` and 36 `alignas(64)` sites | `concurrent_map.hpp:162`, host and device headers | false sharing against padding bytes |
| H17 | `HG_JOB_QUEUE_CAPACITY` | 4096 per worker deque | `job_system.hpp:12` | inline runs on overflow against 32 KB per worker |
| H18 | `HG_JOB_INJECTOR_CAPACITY` | 32768 | `job_system.hpp:15` | inline runs on overflow against 256 KB |
| H19 | job slot | 512 B payload, 16 B header | `job_system/include/job_system/job_pool.hpp:42-43` | malloc fallback for large jobs against bytes per job |
| H20 | `HG_JOB_CHUNK_SLOTS` | 256 slots (about 135 KB per chunk) | `job_pool.hpp:14` | malloc count against per-thread footprint |
| H21 | `HG_JOB_MAX_POOLS` | 1024 | `job_pool.hpp:11` | registry size |
| H22 | branchial pass threads | `min(k/64 + 1, hardware_concurrency())`, own `std::thread` pool | `paclet_support.cpp:870` | runs outside the job system (C3) |
| H23 | glibc malloc arenas | glibc default (8 x cores) | no `mallopt` in the tree | address space grows about 64 MB per worker (C24, board H7) |
| H24 | `keyed_claim_limit` | 1024 | `hypergraph.hpp:726` | keyed twin claims kept against memory |

### 1.4 Device constants

| # | Constant | Value | Site | Trade |
|---|---|---|---|---|
| D1 | persistent grid | SMs x 8; 32 SMs when the query fails | `gpu/src/persistent.cu:1704-1722` | worker blocks against IR arena, TR scratch and survivor scratch, which scale with the grid (`evolve.cu:435`, `:935-941`) |
| D2 | `kMatchBlockThreads` | 32 (one warp per block) | `gpu/include/hg_gpu/match.hpp:92` | per-item serial work against warps per SM; with D1 gives 8 warps per SM |
| D3 | `kLaneBatchBacklog` | 64 records | `persistent.cu:708` | spread of a short backlog over the grid against batch efficiency |
| D4 | `kBatchRecords` | 8 | `persistent.cu:714` | records per claim |
| D5 | `kDeepStreak` | 16 | `persistent.cu:715` | iterations before a block claims 32 records |
| D6 | `kMaxTile` | 4 lanes | `persistent.cu:716` | IR tile width for a batch |
| D7 | `kFollowEdges` | 32 edges | `persistent.cu:719` | wait for a twin against running IR |
| D8 | `kBatchBusyStreak` | 2 | `persistent.cu:723` | iterations before batching after idle |
| D9 | idle backoff | `__nanosleep` 64 ns doubling to 4096 ns | `persistent.cu:659`, `:699-700`, `:1183` | memory bandwidth taken by idle polling against reaction time |
| D10 | `kExactHashRangeThreads` | 1024 | `gpu/src/ir_canon.cu:430` | IR threads in the range path |
| D11 | `kExactHashArenaBudgetBytes` | 512 MB | `ir_canon.cu:429` | IR concurrency against device memory |
| D12 | one-shot launch shapes | `k_measure_states` 128, `k_exact_hash_range` 64, `k_rewrite` 64, seed kernels 128, `k_seed_root_hashes` 64, `k_qe_redrive` 64, `k_init_indices` 128, ring seq 256, hash table gather/clear 256 | `ir_canon.cu:446,465`, `rewrite.cu:693`, `persistent.cu:1674,1861,1906,2221,2225,2247`, `initial_upload.cu:110,130`, `hash_table.hpp:391,427` | occupancy per kernel |
| D13 | fixed grids | `k_qe_count_branchial<<<1024,128>>>`, `k_expand_log_clear<<<64,256>>>`, `k_clear_regions` up to 1024 x 256 per region | `quotient.cu:71`, `persistent.cu:2348`, `engine_state.cu:114-117` | thread count independent of SM count: 16,384 threads for the log clear on a 128-SM part that holds 196,608 |
| D14 | `max_blocks_per_launch` | 0 (no chunking) | `evolve.hpp:450` | launch size against watchdog time on display GPUs |
| D15 | device stack | derived: largest `localSizeBytes` of the kernels, at least 1024 B; 8,117 B/thread at e93154b2 | `engine_state.cu:327` | reserved for SMs x max threads per SM (1,522 MB on the RTX 4090) |
| D16 | memory cap | 90% of total VRAM | `evolve.cu:983` `resolved_memory_cap` | ignores memory other processes hold; a first engine past the free memory fails and is halved (`kFirstShrinks` 3) |
| D17 | pool sizes (`config_from_input`) | growth 4^steps capped at 32; edges max(2^20, n x growth x 512); states max(2^17, n x growth x 32); causal 8x events, triples 16x, pairs 8x, branchial 8x/16x, `branchial_index_buckets` 2^20 | `gpu/src/evolve.cu:44-117` | device bytes and per-run clears against grow-and-retry reruns (D78, C11): bigpath n256 d3 quotient asks 524,288 states for 769 |
| D18 | grow-and-retry | 8 doublings, 3 first-attempt halvings, 1/6 headroom | `evolve.cu:996`, `:1026`, `estimated_device_bytes` | retries against overallocation |
| D19 | `EngineConfig` defaults | `ir_arena_share_words` 65536, `slice_scan_max_edges` 256, `keyed_claim_limit` 1024, `event_canon_slots` 2^16, `match_dedup_slots` 2^16 | `gpu/include/hg_gpu/evolve.hpp:400-470` | capacities with overflow recording and retry |
| D20 | `kDrainCapBuffer` | 512 matches, 4 KB shared memory | `match.hpp:98` | a pair or state past 512 is ranked again in a spill buffer of its exact count from the block's IR arena (`SuccessorCtx::spill`); a short arena records `kIRArenaExhausted`, which grow-and-retry doubles |
| D21 | `kQeInstShards` | 16 | `quotient_expansion.hpp:364` | device mirror of H10 |
| D22 | L2 | not queried; no access policy window | none | hot claim maps compete with streaming pools for L2 |

Count: 24 host rows holding 26 values and 22 device rows holding 48 values (D12 holds nine block sizes, D13 three grids, D17 eleven pool fields, D18 three, D19 five), 74 tuning values in all.

## 2. Proposed

### 2.1 One probe per side

**Proposed.** All derivations below read two cached structs. No consumer calls the OS or the CUDA runtime for a hardware property itself.

Host, `hgcommon::host_topology()` in `common/include/hgcommon/affinity.hpp`, implemented in `job_system/src/affinity.cpp` beside the readers it reuses (`performance_cpus`, `cache_domains_of`, `allowed_cpus`):

| Field | Linux | Windows |
|---|---|---|
| `allowed` logical CPUs | `sched_getaffinity` | `GetProcessAffinityMask` |
| `core_of[cpu]`, SMT width | `topology/core_id`, `thread_siblings_list` | `RelationProcessorCore` |
| `class_of[cpu]` (P or E) | `/sys/devices/cpu_core/cpus`, `cpu_capacity` | `EfficiencyClass` |
| `llc_of[cpu]`, `llc_bytes[d]` | `cache/indexK/{id,size}` | `RelationCache` |
| `l2_bytes` per core | `cache/index2/size` | `RelationCache` level 2 |
| `available_bytes` | `MemAvailable` from `/proc/meminfo` | `GlobalMemoryStatusEx` |
| `huge_page_bytes` | `/sys/kernel/mm/transparent_hugepage/hpage_pmd_size` | `GetLargePageMinimum` |

Under WSL sysfs presents vCPUs (a 14900K appears as 16 cores x 2 threads, `affinity.hpp:23-26`); the struct reports what sysfs says and marks `virtualised = true` (`/proc/sys/kernel/osrelease` contains `microsoft`), so derivations that depend on physical placement fall back to the homogeneous rule.

Device, `hg_gpu::DeviceProbe` (`gpu/include/hg_gpu/device_probe.hpp`), filled from the `cudaDeviceProp` that `run_probe` already reads: `sm_count`, `max_threads_per_sm`, `max_blocks_per_sm`, `regs_per_sm`, `shared_per_sm`, `shared_per_block_optin`, `l2_bytes`, `persisting_l2_max`, `total_bytes`. `default_persistent_grid` (`persistent.cu:1714`) and `device_resident_threads` (`:1752`) read the probe and lose their fallback constants (32 SMs, 2048 threads): a run reaches them only after the probe reported the device usable. Kernel-specific occupancy (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`) is computed once per kernel in the file that owns the kernel and cached there, because it needs the kernel symbol.

Both structs print through one flag: `hg_evolve --print-tuning` (and `hg_evolve_gpu --print-tuning`) writes every derived value with its source, so a report from another machine carries its settings.

**Implemented.** `host_topology()` lists every online CPU with its dense core id, core class and LLC id; `performance_cpus()`, `cache_domains_of()` and `default_worker_cpus()` (the job system's default placement) read it. On Linux, Intel hybrid parts without `/sys/devices/cpu_core` fall back to CPUID leaf 0x1A when CPUID.07H:EDX[15] is set. macOS fills counts, caches and memory from `sysctl` and has no per-CPU entries. `DeviceProbe` carries the fields above plus driver and runtime versions; `usable_device()` throws when the probe failed, and `default_persistent_grid`, `device_resident_threads` and `resolved_memory_cap` read it. Readings on the i9-14900K and RTX 4090:

| Source | Reading |
|---|---|
| WSL (`hg_evolve --print-tuning`) | 32 logical, 16 cores x 2, one class, one 36 MB L3, 2 MB L2, `virtualised`; placement left to the OS (one cache domain). `lscpu` reports the same layout. CPUID leaf 0x1A reads 0 on every vCPU and the hybrid bit is clear. |
| Windows (the same reader, cross-compiled with MinGW) | 32 logical, 24 cores, 2 classes, 8 performance cores (CPUs 0-15, SMT), 16 efficiency cores (CPUs 16-31), one 36 MB L3, 2 MB L2 |
| `hg_evolve_gpu --print-tuning` | RTX 4090, cc 8.9, 128 SMs, 1536 threads and 24 blocks per SM, 65536 registers and 100 KB shared memory per SM, 72 MB L2, 49.5 MB persisting L2, 24564 MB; grid 1024, 196,608 resident threads, memory cap 22,107 MB. `nvidia-smi` reports cc 8.9 and 24564 MiB. |

### 2.2 Knob scheme

**Proposed.** One option, `"Tuning" -> <|name -> value, ...|>`, carried in the job to `hg_evolve` and to the C++ entry points as a `TuningOverrides` struct; one environment variable per name, `HG_TUNE_<NAME>`, read once by the probe for the standalone tools. An override replaces the derived value and is echoed by `--print-tuning`. `HG_GPU_PERSISTENT_BLOCKS` stays as an alias of `HG_TUNE_GPU_BLOCKS`. Compile-time macros stay for the sizes the model checker shrinks. A separate `"Threads" -> n` option exposes the worker count, the one knob a general user needs.

**Implemented.** `hgcommon::TuningOverrides` (`common/include/hgcommon/tuning.hpp`) holds the names; `hgcommon::tuned(job, name, derive)` resolves a value as the job's override, else `HG_TUNE_<NAME>`, else the derived default, and every consumer calls it. The FFI reads `"Threads"` (0 is `Automatic`) and `"Tuning"` into `ParsedJob::tuning`; a `"Tuning"` entry with an unknown name or a value below the name's minimum is skipped with an `OptionSkipped` warning and the other entries apply. The C++ entry points take the values as their own arguments (`ParallelEvolutionEngine` `num_threads`, `EvolveInput::max_device_memory_bytes`), where 0 is the derived default. The `"Device"` reply reports the values a run used: `"Threads"` on both engines, `"MemoryCap"` on the GPU. Test: `WxfSerializationPin.TuningOverridesReachTheEngine` (all_tests and gpu_ffi_tests).

| Name | Environment | Consumer | Derived default |
|---|---|---|---|
| `"Threads"` (its own option) | `HG_TUNE_THREADS` | `ParallelEvolutionEngine`, `JobSystem`, the GPU reply's StepStatistics | `hgcommon::default_worker_count()`: the logical CPUs the process may run on (`sched_getaffinity`, `GetProcessAffinityMask`), at most `MAX_ARENA_WORKERS` for the engine |
| `GPUMemoryLimit` | `HG_TUNE_GPU_MEMORY_LIMIT` | `resolved_memory_cap` | D16 |

### 2.3 Derivations

Notation. Host: `L` allowed logical CPUs, `C` physical cores among them, `P` performance-class physical cores (equal to `C` on homogeneous parts), `D` LLC domains, `LLC` bytes per domain, `M` available bytes, `HP` huge page bytes, `W` workers, `A` active workers (C1). Device: `G` SMs, `T` max threads per SM, `B` max blocks per SM, `Bocc` occupancy-limited blocks per SM for a kernel, `L2`, `V` total VRAM, `F` free VRAM, `cc` compute capability.

| Row | Formula | Knob | Validation now (counts, occupancy) | Validation on a quiet box |
|---|---|---|---|---|
| H1 worker count | `W = min(L, MAX_ARENA_WORKERS)`; active workers `A` chosen by the C1 mechanism | `"Threads"`, `HG_TUNE_THREADS` | C1 counters (2.4) | mono_check 1..32, no violation at the 5% band (C1 gate) |
| H2, H3 wake and steal | C1 mechanism (2.4) | `HG_TUNE_WAKE_POLICY` = `per-submit` or `throttled` while the A/B runs; deleted with the losing policy | wakes per job, empty wakes, failed steals per job at 16 and 32 workers | as H1 |
| H4 expand chunk | `chunk = clamp(ceil(count / A), 4, 64)` for a state with `count` matches | `HG_TUNE_EXPAND_CHUNK` (fixed value) | callgrind Ir per match at 1/4/16/32 workers; task count per run | wide-state rows (bigpath, wpp d8 q) and narrow rows (cycle4) at 4/16/32 |
| H5 causal set capacity | per shard `clamp(pow2(E / (SHARDS x 0.75)), 1024, 65536)` where `E` is the event bound when caps give one (`1 + steps x MaxStatesPerStep` events times the largest LHS), the previous call's count on a session `extend`, and otherwise 8192 when `M` > 4 GB, else 1024 | `"CausalSetCapacity"`, `HG_CAUSAL_SET_WORKING_CAPACITY` (macro from `.scratch/opt/patches/causal_set_wc8192_trade.patch`) | Ir and cachegrind D1/LL misses on full-multiway rows at 1 and 16 threads; peak RSS on the small rows | wall at 1/16/32 |
| H6, H7, H8 map tables | unchanged; the per-map capacity becomes a constructor argument so a map with a known size (state count on extend) starts there | none new | chain length counter (stats) | none |
| H9, H10, D21 shards | `SHARDS = pow2(max(64, 2 x W_max))` as the compile-time size; `kInstShards = pow2(min(16, max(4, A / 4)))` chosen per run | macros stay | per-shard insert counts (stats) at 32 and 64 workers: max/mean below 1.5 | EPYC 8-domain box at 16 workers |
| H11 id blocks | unchanged at 64; refill count added to stats | macro | refills per 1000 ids | none |
| H13 arena workers | `MAX_ARENA_WORKERS` at least the default `W`; `start()` warns when `W` exceeds it | macro | none | none |
| H14 arena block | `HP` when 2 MB <= `HP` <= 32 MB, else 2 MB | `HG_TUNE_ARENA_BLOCK` | minor faults (`perf stat -e minor-faults`) on aarch64 16 KB and 64 KB page kernels | none |
| H16 cache line | one constant `hgcommon::kDestructiveLine`: 64 on x86-64 and generic aarch64, 128 on Apple aarch64 (128 B lines); every `alignas(64)` and `kLine` reads it | `HG_DESTRUCTIVE_LINE` macro | `perf c2c` HITM counts on the padded heads; struct sizes by `static_assert` | macOS arm64 runner |
| H17-H21 job capacities | unchanged; overflow already runs inline | macros | inline-run counter (stats) must stay 0 on the corpus | none |
| H22 branchial threads | `A` from the job system (C3 moves the pass into it) | `"Threads"` | as C3 | as C3 |
| H23 malloc arenas | `mallopt(M_ARENA_MAX, W)` at engine start when C24 measures no time loss | `MALLOC_ARENA_MAX` | RSS and VSZ at 1..32 workers (C24) | wall at 1..32 |
| D1 persistent grid | `grid = G x min(Bocc(k_persistent_evolve, 32), k_cc)`, `k_cc` from a per-arch table: 8 for 8.9 (the sweep at `persistent.cu:1680-1696`), 8 for unmeasured arches until their sweep runs | `"GPUBlocks"`, `HG_TUNE_GPU_BLOCKS`, `HG_GPU_PERSISTENT_BLOCKS` | `ncu` achieved occupancy and `sm__warps_active` on `k_persistent_evolve`; device bytes from `estimated_device_bytes` at each grid | the existing sweep (`bench_gpu_evolve 7 5 2` over 32..3072 blocks) per arch |
| D3 backlog | `kLaneBatchBacklog = max(8, grid / 16)` (64 at grid 1024, the measured point) | `HG_TUNE_GPU_BACKLOG` | claims per record and idle cycles from `HG_GPU_DBG_TIME` | as D1 |
| D4-D8 batch shape | per-arch table, 8.9 values as today; passed in the kernel's parameter struct (one constant load) | `HG_TUNE_GPU_BATCH_*` | `HG_GPU_DBG_TIME` phase split | sweep on a second arch |
| D9 idle backoff | per-arch table, 4096 ns cap on 8.9 | none | DRAM throughput while idle (`ncu dram__throughput`) | none |
| D10, D11 range IR | threads `= min(n, G x T / 4, affordable)`; arena budget `= min(512 MB, F / 8)` | `HG_TUNE_GPU_IR_ARENA` | `ncu` occupancy on `k_exact_hash_range` | none |
| D12 block sizes | `cudaOccupancyMaxPotentialBlockSize` once per kernel, cached; kernels whose body assumes a shape (`kMatchBlockThreads`) keep it | none | `ncu` occupancy per kernel against today's shape | none |
| D13 fixed grids | one helper `grid_stride_blocks(n, block) = min(ceil(n / block), G x T / block)` used by every grid-stride kernel | none | `ncu` duration of `k_expand_log_clear`, `k_qe_count_branchial` | none |
| D16 memory cap | `cap = min(0.9 x V, F - 256 MB)` read at engine construction | `max_device_memory_bytes` (exists) | first-attempt failures logged by `run_with_growth` with another process holding VRAM | none |
| D17 pools (C11) | each field `min(bound, budget)`: `bound` from the run's caps when set (states <= 1 + steps x MaxStatesPerStep, events likewise, edges from `max_state_edges`), from the previous call's counts on a `PersistentEvolver` or session, else today's guess; `budget = cap / 4` split by the bytes per unit in `estimated_device_bytes`; `branchial_index_buckets = pow2(events / 4)` | `"GPUMemoryLimit"` | retries per corpus row (target 0), device bytes and clear bytes per run | wall on bigpath n256 d3 q and small rows |
| D20 cap buffer | dynamic shared memory: `min(shared_per_block_optin / 8, 4096)` matches | none | `kDrainCapBufferFull` count on MatchesPerStateRule tests | none |
| D22 L2 | on `cc >= 8.0`, an access policy window over the state claim map when its bytes are at most `persisting_l2_max`; `cudaLimitPersistingL2CacheSize = min(persisting_l2_max, map bytes)` | `HG_TUNE_GPU_L2_PERSIST` = 0 disables | `ncu lts__t_sector_hit_rate` on the persistent kernel | as D1 |

### 2.4 C1: active workers scaled to the available work

**Observed (rc2, `.scratch/bench_rc2/summary.txt`).** 16 against 32 threads: multirule d7 quotient 20.7 against 30.2 ms, bigpath n128 d3 quotient 17.4 against 21.1, cycle4 d6 full 14.4 against 15.0; spreads 41-53% at 32. On the i9-14900K, workers 17-32 add 8 SMT siblings of busy P-cores and 16 E-cores (0.59 of a P-core each, `affinity.hpp:11`). Eight times the workers adds 0.21% instructions (`docs/OPTIMISATION_DISPOSITIONS.md`, "The parallel overhead is not work"), so the loss is stalls and waiting, not extra work.

**How idle workers behave today** (`job_system.hpp:642-711`, `park_gate.hpp:92-155`):

1. A worker with an empty deque runs `find_work`: its cache peers once each, then `n` random victims, each a read of the victim's `bottom_` and `top_` lines (`work_stealing_deque.hpp:127-131`), then the injector.
2. It then announces itself idle (two seq_cst increments), takes an exhaustive look (every deque and the injector) and parks on its domain's futex word.
3. Every submit that finds any idle worker runs a seq_cst barrier, bumps a domain sequence word and calls `unpark_one` (a futex wake system call).

So while the run is narrower than the pool, each submit wakes one worker, that worker reads up to `2n + n` deque lines owned by busy workers and parks again. The reads pull the owners' `bottom_` lines to shared state, and each owner's next push or pop pays a write miss. The submit pays a system call. Both costs grow with `n` and with the number of idle workers, which is what a run narrower than 32 workers has.

**Mechanism (proposed).** A searching-worker throttle in `ParkGate`, lock-free, with no new wait:

- A count `searching` (its own cache line) of workers that are awake and have no job.
- Submit side: after the push, the existing barrier, then wake one worker only when `searching == 0` and `idle > 0`, or when the submitter's own deque holds more than `2 x searching` jobs (a burst). The woken worker is counted into `searching` by the waker before the wake, so concurrent submitters do not wake a second worker for the same job.
- Worker side: a worker that finds a job leaves `searching`; if it was the last searcher and `idle > 0` and work remains visible (its own deque or the victim it stole from is non-empty), it wakes one more. Wakes therefore chain: a further worker wakes only after the previous one found work.
- A worker that finds nothing leaves `searching` with a seq_cst decrement, then follows today's protocol (announce idle, last exhaustive look, park).
- At most `max(1, busy / 2)` workers search at once; a worker that would exceed it skips the random-victim loop and goes to the exhaustive look and park.
- Wake target: within the submitter's domain, prefer a parked worker whose physical core has no busy sibling, and a P-core over an E-core, from `host_topology()`. Under WSL or on homogeneous parts this reduces to today's choice.

No thread waits on another: a submit never blocks, a searching worker never waits on a count, and parked workers park exactly as today. A worker that skipped searching still runs the exhaustive look before it parks.

**Proof obligation.** The skip of a wake when `searching > 0` is a new StoreLoad pair: the submitter writes the job and reads `searching`; the searcher decrements `searching` and reads the deques. Both sides need the seq_cst order the park gate already uses (`HG_PG_BARRIER`). Extend `verification/genmc/job_system_no_lost_wakeup.cpp` and `job_system_no_lost_wakeup_domains.cpp` with the `searching` count, and add a calibration arm (`HG_PARK_GATE_WEAK_ORDERS` applied to the new pair) that the checker must report as a lost wakeup. `verification/tla/Quiescence.tla` gains the searching state so `MCQuiescence` covers it.

**Diagnostic that runs on a busy box.** Counters under `HG_STAT` (stats builds only), per worker, summed in `JobSystem::get_statistics` and printed by `tools/bench_cpu_evolve.cpp` beside the existing `threads=... speedup=...` line (the existing probe is extended; no new tool):

| Counter | Exists | Meaning |
|---|---|---|
| `jobs_executed`, `jobs_stolen`, `jobs_stolen_near` | yes (`job_system.hpp:138-144`) | per-worker share of the work |
| `park_waits` | yes, global (`park_gate.hpp:149`) | parks |
| `wakes_issued` | no | `unpark_one` calls from submits |
| `wakes_empty` | no | a worker woke and parked again without running a job |
| `steal_attempts`, `steal_failures` | no (C31) | victim deque reads, and those that found nothing |
| `inline_runs` | no | jobs run on the submitter because both queues were full |
| `searching_max` | no (with the mechanism) | largest number of simultaneous searchers |

Derived per run: wakes per job, `wakes_empty / wakes_issued`, failed steals per job, parks per job, and the max/min worker share of `jobs_executed`. Counts depend on the schedule, so each configuration runs 5 times and the report gives medians and ranges. Rows: multirule d7 quotient, bigpath n128 d3 quotient, cycle4 d6 full, wolftri d6 full, wpp d7 full, at 16 and 32 workers, before and after the mechanism. Expected direction if the mechanism is right: at 32 workers wakes per job and failed steals per job fall to near their 16-worker values, and on the narrow rows the number of workers with a non-zero share falls below 32. Callgrind Ir at 16 and 32 workers confirms that the throttle removes no work. Wall clock on a quiet box (the C1 gate, mono_check over 1..32) decides adoption; the losing policy is deleted in the same commit.

**Implemented: the counters.** `JobSystem::scheduler_counters()` and `ParallelEvolutionEngine::scheduler_counters()` exist only in stats builds. `wakes_issued` counts `unpark_one` calls (one `FUTEX_WAKE` each on Linux), whether or not a worker was asleep. `wakes_empty` counts a return from a park followed by another park with no job run between. `steal_attempts` counts victim deque reads in the search and in the last look before a park. `searching_max` is the high-water mark of workers between an empty own deque and either a job or the idle announcement. `bench_cpu_evolve` prints one `sched:` line per evolve.

**Baseline (current scheduler).** `HG_BENCH_CANON_ONLY=<1 quotient, 0 full> HG_BENCH_LARGE_N=128 bench_cpu_evolve <depth> 5 <threads> <workload>`, stats build, 5 evolves, medians with [min-max]; bench default record set, so these runs are not the paclet jobs of `.scratch/bench_rc2`:

| Row | Workers | Jobs | Wakes | Wakes empty | Steal attempts | Failed steals | Parks | Searching max | Inline |
|---|---|---|---|---|---|---|---|---|---|
| multirule d7 q | 16 | 335 | 207 [92-255] | 12 [2-30] | 3,227 | 3,053 [1,466-3,543] | 66 | 2 | 0 |
| multirule d7 q | 32 | 370 | 248 [167-300] | 22 [12-35] | 9,047 | 8,751 [8,148-12,150] | 103 | 2 | 0 |
| bigpath n128 d3 q | 16 | 1,419 | 555 [402-1,407] | 4 [0-16] | 6,179 | 5,174 [5,089-6,750] | 47 | 2 | 0 |
| bigpath n128 d3 q | 32 | 1,424 | 577 [396-1,536] | 22 [5-71] | 18,211 | 17,098 [14,897-23,754] | 115 | 2 | 0 |
| cycle4 d6 f | 16 | 99,002 | 21,025 [224-41,776] | 10 [3-14] | 18,097 | 12,396 [11,289-70,887] | 56 | 3 | 0 |
| cycle4 d6 f | 32 | 99,002 | 53,681 [6,263-54,309] | 47 [14-60] | 58,192 | 46,238 [39,924-160,199] | 149 | 4 | 0 |
| wpp d8 q | 16 | 629,980 | 12,572 [338-16,812] | 5 [1-12] | 798,308 | 745,475 [576,491-1,014,882] | 41 | 6 | 61,218 [41,475-73,832] |
| wpp d8 q | 32 | 627,092 | 34,339 [588-41,374] | 3 [2-23] | 899,250 | 867,187 [565,664-1,279,965] | 71 | 7 | 2 [0-66,556] |

Failed steals per job, 16 to 32 workers: multirule 9.3 to 23.0, bigpath 3.7 to 10.5, cycle4 0.13 to 0.47, wpp 1.18 to 1.38. Wakes exceed parks by 3x (multirule) to 480x (wpp at 32). A park is woken at most once, so at least wakes minus parks `FUTEX_WAKE` calls found no sleeper: the idle count (`park_gate.hpp:138-153`) covers a worker from its announcement until its park has returned, and every submit in that window signals. wpp d8 quotient at 16 workers runs 41,475 to 73,832 jobs inline (a worker's 4,096-slot deque and the 32,768-slot injector both full), so the H17-H21 row's expectation of zero inline runs does not hold there.

**Implemented: the throttle.** `hgcommon::ParkGate` (`common/include/hgcommon/park_gate.hpp`) carries the searching count on its own line, and `JobSystem` drives it:

- A worker with an empty deque calls `search_begin`: counted as a searcher, or declined past `max(1, busy / 2)` searchers unless its last park returned (a woken worker always searches). A declined worker takes the last look and parks.
- `wake_one` skips the wake while a worker searches, unless the submitter's own deque holds more than twice the searchers. A thread that is not a worker has backlog 0.
- `search_found`: the last searcher out wakes one more worker when a worker is idle and the queue it took from still holds work: at least two jobs in a worker's deque, whose owner takes the next one itself, or any job in the injector.
- `park_unless` takes a counted searcher out of the count before the idle announcement, so the barrier orders the decrement against the submitter's read.
- Park domains are (cache, tier) pairs; tier is 0 for the first hardware thread of a fastest-class core, 1 for its sibling, 2 and 3 for the next class. A submit wakes from its cache's first domain. Workers are bound only on multi-cache parts or when the caller names CPUs, so on the i9-14900K under WSL and on bare metal with its one L3 there is one tier.

Measuring the specified mechanism alone on this box, interleaved with the step-2 binary, showed failed steals per job at 32 workers unchanged or higher (bigpath 10.5 to 15.2, multirule 23.0 to 26.9) and no fall in empty wakes. Two changes to the steal search remove the failed reads:

- **Awake mask.** Bit i is clear from worker i's idle announcement until its park returns, the span in which its deque is empty (only the owner pushes to it, and it found the deque empty before announcing). `steal_work` reads each awake worker's deque once, from a starting worker upward. Pools over 64 workers keep the random draw. The last look before a park still reads every deque.
- **Starting victim.** The search starts at the worker's last successful victim, or after a park at the wake hint (the worker whose submit signalled, or the deque a chaining searcher found work in).

**After.** Same command as the baseline, the step-2 binary (f8334d8a) and this one alternating process by process, 6 rounds of 5 evolves (n = 30 each), medians:

| Row | Workers | Failed steals per job, before / after | Empty wakes, before / after | Wakes, before / after |
|---|---|---|---|---|
| multirule d7 q | 16 | 8.10 / 4.59 | 13 / 13 | 98 / 122 |
| multirule d7 q | 32 | 30.47 / 18.24 | 28.5 [7-58] / 31 [5-53] | 301 / 360 |
| bigpath n128 d3 q | 16 | 3.70 / 0.91 | 3 / 4.5 | 185 / 234 |
| bigpath n128 d3 q | 32 | 16.11 / 4.23 | 66.5 [4-104] / 28.5 [4-95] | 487 / 755 |
| cycle4 d6 f | 16 | 0.13 / 0.01 | 2 / 1 | 652 / 210 |
| cycle4 d6 f | 32 | 0.41 / 0.07 | 40.5 [1-193] / 34 [2-104] | 31,260 / 39,646 |
| wpp d8 q | 16 | 1.09 / 0.82 | 3 / 4 | 4,162 / 5,575 |
| wpp d8 q | 32 | 1.89 / 1.69 | 4 [2-103] / 4 [0-41] | 30,098 / 31,436 |

Failed steals per job fall on every row at both counts, by 11% (wpp at 32) to 92% (cycle4 at 16). Empty wakes at 32 fall on bigpath and cycle4 and do not move on multirule (a second interleaved run: 28 to 26) or wpp (4 in about 30,000 wakes). Total wakes rise on the narrow rows: the chain wakes add to the submit wakes. Job and state counts are identical (`CausalDeterminism.*`, `SamplingReproducibility.*` at 1, 2, 8, 16 and 32 threads pass under both policies).

Proof: `verification/genmc/job_system_no_lost_wakeup{,_domains}.cpp` drive two looping workers, a submitter that is not a worker, and a caller that stops them, through the real header, with calibrations `HG_PARK_GATE_SEARCH_LEAVE_AFTER_BARRIER` (RC11 only) and `HG_PARK_GATE_SEARCH_LEAVE_LATE`; `verification/tla/Quiescence.tla` adds the searching state with `NoLostWakeup` and the cells `MCQuiescencePerSubmit`, `MCQuiescenceThreeWorkers` and `MCQuiescenceSearchLeaveLate` (VIOLATION).

**Wall-clock A/B on a quiet box (deferred).** `HG_TUNE_WAKE_POLICY=per-submit` restores the per-submit wake; the awake mask and the starting victim apply under both policies. Build `bench_cpu_evolve` in a stats-OFF release directory (`build_linux`), then for each row interleave the two policies process by process:

    cmake --build build_linux --target bench_cpu_evolve -j4
    mkdir -p .scratch/c1
    T=$(seq -s, 1 32)
    for round in 1 2 3 4 5; do
      for policy in throttled per-submit; do
        for row in "multirule 7 1" "bigpath 3 1" "cycle4 6 0" "wolftri 6 0" "wpp 7 0" "wpp 8 1"; do
          set -- $row
          HG_TUNE_WAKE_POLICY=$policy HG_BENCH_CANON_ONLY=$3 HG_BENCH_LARGE_N=128 \
            build_linux/bench_cpu_evolve $2 15 $T $1 > .scratch/c1/mono_${policy}_$1_d$2_q$3_r$round.log
        done
      done
    done
    python3 tools/dev/mono_check.py .scratch/c1/mono_throttled_*.log
    python3 tools/dev/mono_check.py .scratch/c1/mono_per-submit_*.log

The C1 gate is `mono_check` with no violation at the 5% band for the throttle; the policy with the lower median at 16 and 32 workers on each row is kept and the other is deleted with the `HG_TUNE_WAKE_POLICY` switch.

### 2.5 Order of implementation

1. Host and device probes (2.1) and `--print-tuning`. Every later row reads them.
2. Scheduler counters in `bench_cpu_evolve` (2.4 table), then the 16/32 counter runs on the current scheduler. This is the C1 baseline and needs no quiet box.
3. C1 searching-worker throttle with the GenMC and TLA+ extensions, then the counter runs again.
4. `"Threads"` option and `HG_TUNE_*`/`"Tuning"` plumbing (2.2).
5. D17 pool sizing from bounds and session counts (C11); D16 memory cap from free VRAM.
6. D1 grid from occupancy with the per-arch table; D3 backlog scaled with the grid.
7. H5 causal set capacity (board H3).
8. H4 adaptive expand chunk.
9. D22 L2 persistence window; D13 grid-stride helper; D12 block sizes.
10. H16 cache-line constant, H14 arena block from the huge page size, H23 malloc arenas (after C24), H9/H10 shard counts.

### 2.6 Implemented rows

**H5 causal set capacity.** `ParallelEvolutionEngine::configure_tuning` sets, at each `evolve()`, the first table size of the two causal sets (`CausalGraph::set_causal_set_capacity`): `"Tuning" -> <|"CausalSetCapacity" -> n|>`, `HG_TUNE_CAUSAL_SET_CAPACITY`, or `derived_causal_set_capacity`. With a step cap N the keys are at most (1 + steps x N) x the largest left-hand side, and a shard takes that over 64 shards at load 0.75, as a power of two from 1024 to 65536; without a cap the value is 8192 when `MemAvailable` is above 4 GB, else 1024. Model-checked harnesses keep 1024. The `"Device"` reply of the CPU engine reports `"CausalSetCapacity"`. Measured on the i9-14900K, stats build, one evolve at one thread, 1024 against 8192 on the same binary through the override: callgrind Ir cycle4 depth 6 full 1,738,356,884 -> 1,718,747,106 (-1.13%), multirule depth 6 full 2,639,556,621 -> 2,598,951,896 (-1.54%); peak RSS multirule depth 3 full 15.4 -> 18.9 MB (3 runs each, spread 0.2 MB). Test: `ParallelEvolution.CausalSetCapacityChangesNoRelation` (causal edges, causal pairs and events equal at 1, 1024, 65536 and the default, at 1, 4 and 32 threads).

**H4 expand chunk.** `dispatch_expansion` splits a state's match set into tasks of `derived_expand_chunk(count, workers)` = ceil(count / workers) from 4 to 64 matches, or the fixed `"Tuning" -> <|"ExpandChunk" -> n|>` / `HG_TUNE_EXPAND_CHUNK`. The set reaches one expansion when a scan task completes every match of a one-edge left-hand side; a rule with more edges completes its matches one at a time, which the chunk size does not change. Job counts, `bench_cpu_evolve` stats build, 16 against the derived size on the same binary through the override (instruction counts at more than one thread are not used: valgrind serialises the threads and the idle loops then dominate):

| Row | Workers | Jobs, 16 / derived | Wakes, 16 / derived |
|---|---|---|---|
| binary depth 6 full | 1 | 308 / 308 | 1 / 1 |
| binary depth 6 full | 4 | 308 / 452 | 16 / 23 |
| binary depth 6 full | 16 | 308 / 452 | 53 / 74 |
| binary depth 6 full | 32 | 308 / 452 | 201 / 203 |
| arimix depth 5 full | 4 | 459 / 579 | 19 / 413 |
| arimix depth 5 full | 32 | 459 / 579 | 164 / 441 |

States on these rows hold at most 7 matches, so at 4 or more workers the derived size splits a 5-to-7-match state into tasks of 4 where 16 ran it inline; at one worker the job counts are equal. Test: `ParallelEvolution.ExpandChunkChangesNoResult` (the derivation's cases; canonical states, events and causal pairs equal at chunk sizes 1, 16, 64 and derived, at 1, 4 and 32 threads).

**D16 memory cap.** `resolved_memory_cap` is `"Tuning" -> <|"GPUMemoryLimit" -> n|>`, `HG_TUNE_GPU_MEMORY_LIMIT`, or `memory_cap_of(90% of total, free + held - 256 MB)` with free from `cudaMemGetInfo` at each run; held is the live engine's pools on a `PersistentEvolver` (freed before a larger engine is built; the device stack stays). On the RTX 4090 with 1,152 MiB used by other processes: 22,107 MB (90% of total is the smaller term). Test: `test_smoke`.

**D17 pools.** A step cap that counts transitions (`"MatchesPerStep"`, or `"MaxStatesPerStep"` under `CanonicalizeStates -> None`) bounds events at steps x cap and raw states at roots + events, and `config_from_input` takes at most twice that for states, events and edges (2^10 states, 2^12 edges at least). `"MaxStatesPerStep"` under `Automatic` or `Full` keeps every transition into a kept successor state, so it bounds nothing. The first engine of a run is fitted to a quarter of the cap (`initial_config`; 5,527 MB on the RTX 4090); grow-and-retry continues to the cap. `branchial_index_buckets` is one per four events, a power of two from 2^12 to 2^20 (2^15 at the 2^17-event floor, against 2^20 before). The previous call's size on a `PersistentEvolver` is the live engine's config, which `covering_config` starts from. `HG_GPU_DBG_TIME` prints each attempt's config. Measured: a 3-transition cap at 4 steps sizes 2^10 states and events against 2^17 (`CapacityOverflow.AStepCapThatCountsTransitionsBoundsThePools`, which also runs it to completion on the first attempt); grow-and-retry attempts over `gpu_differential_tests`: 15, the same 15 as master.

**D1 persistent grid and D3 backlog.** `default_persistent_grid` is SMs x min(`persistent_blocks_per_sm()`, the architecture's plateau), or `HG_TUNE_GPU_BLOCKS` (`HG_GPU_PERSISTENT_BLOCKS`); `persistent_blocks_per_sm()` is `cudaOccupancyMaxActiveBlocksPerMultiprocessor` of `k_persistent_evolve` at 32 threads, and the plateau is 8 per SM for every architecture (8.9 measured, the rest until their sweep runs). A run's grid is `EngineConfig::persistent_grid` (`"Tuning" -> <|"GPUBlocks" -> n|>`), read through `persistent_grid_of` by the launch, the IR arena, the reachability and survivor scratch, the replay lanes and the memory estimate. The backlog is `DeviceState::lane_backlog` = `lane_backlog_of(cfg)`: `"GPUBacklog"`, `HG_TUNE_GPU_BACKLOG`, or max(8, grid / 16). On the RTX 4090: occupancy 8 blocks per SM, plateau 8, grid 1024, backlog 64, the values the constants held. The `"Device"` reply of the GPU reports `"Blocks"` and `"Backlog"`; `WxfSerializationPin.TuningOverridesReachTheEngine` runs at grid 256 and backlog 12 and gives the counts of the default run.

**D13 fixed grids.** `grid_stride_blocks(n, block)` = ceil(n / block), at most the device's resident threads over `block`, at least 1. `k_qe_count_branchial` (1024 x 128 before), `k_expand_log_clear` (64 x 256) and `k_clear_regions` (up to 1024 x 256 per region) use it; on the RTX 4090 the cap is 768 blocks of 256 and 1,536 of 128. Test: `test_smoke`.

**D22 L2 window.** On compute capability 8.0 and above, the `k_persistent_evolve` launch (`cudaLaunchKernelEx`) carries an access policy window over the state claim map with hit property persisting, when the map's bytes are at most `l2_persist_limit_of`: `"GPUL2Persist"`, `HG_TUNE_GPU_L2_PERSIST`, or `persisting_l2_max` (49.5 MB on the RTX 4090; `0` turns the window off). The persisting set-aside is set to the window's bytes and the persisting lines are reset after the kernel. At the default 2^17 states the map is 2^18 slots of 16 B, 4 MB. The `"Device"` reply reports `"L2Persist"`. The hit rate needs `ncu` and is in the quiet-box queue.

**D12 block sizes (refuted).** `cudaOccupancyMaxPotentialBlockSize` was measured on every one-shot kernel the differential corpus launches (`HG_GPU_DBG_TIME`, RTX 4090): theoretical occupancy at its block size against the fixed one, `k_clear_slots` 768 / 256: 1.00 / 1.00, `k_gather_keys` 768 / 256: 1.00 / 1.00, `k_init_indices` 768 / 128: 1.00 / 1.00, `k_seed_frontier` 768 / 128: 1.00 / 1.00, `k_seq_ramp` 768 / 256: 1.00 / 1.00, `k_qe_redrive` 256 / 64: 0.17 / 0.17, `k_seed_root_hashes` 256 / 64: 0.17 / 0.17. No kernel gains occupancy, so the fixed sizes stay.

**H16 cache line.** `hgcommon::kDestructiveLine` (`common/include/hgcommon/cache_line.hpp`) is 128 bytes on Apple aarch64 and 64 elsewhere, `HG_DESTRUCTIVE_LINE` at compile time overrides it, and the 37 `alignas` sites of per-thread data in `common/`, `hypergraph/`, `job_system/` and `lockfree_deque/`, `ConcurrentMap`'s table line and the arena's largest alignment (`kMaxAlign`) read it. On x86-64 the value is 64, the constant every site held, so the layout of every type is unchanged. Check: the engine, job system and FFI sources compile with `-DHG_DESTRUCTIVE_LINE=128` (clang `-fsyntax-only`); test `CacheLine.EveryPerThreadSlotIsOneConstantAligned`.

## 3. Constants that are not tuning

| Kind | Constants |
|---|---|
| Sentinels and masks | `kEmpty`, `kInvalid`, `kNoSpine`, `kNoPivotVar`, `kEmptyHead`, `kQeNoParent`, `kQeNoClaims`, `kKeyedNoNewEdge`, `kExploreNoDepth`, `kUnset`, `kAddrMask`, `kFullWarp`, `kVertexOne`, `kStateStream` |
| Algorithm limits (change what can be represented) | `HG_MAX_PATTERN_EDGES` 16, `kMaxArity`, `kMaxVars`, `kMaxConsumedBits` 1024, `kMaxDedupProbes` 8, `depth_join` `kSegments` 27, `kIrDepthRungs`, `ir_depth` 8, `ir_generators` 32, `kQeEntryLimit`, `kQeWordLimit`, `QR_ID_LIMIT`, `kQeKeptStride`, `termination` `kMaxRoles` |
| Inline capacities with a spill path | `ScratchReachCtx` 64/128 (`scratch_alloc.hpp:190`), `kQeReachStack`/`kQeReachTable` 64/128, `kReachStack`/`kReachVisited` 256/512 (`rewrite.cu:70`, C30), `kTrScratchStack`/`kTrScratchVisited` 2048/4096, `kLocalSurvivors` 256, `kSnapshot` 64 (`concurrent_map.hpp:735`), `ClearBatch::kMaxRegions` 64, `keyed` `kDepth` 32 |
| Pathology ceilings | `kMaxDetectorRounds`, `kMaxWorkerIdleSpins` (`persistent.cu:517-518`) |
| Sized for the model checker only | `HG_ARENA_BLOCK_SIZE` under `HG_VERIFICATION`, `HG_SEGMENTED_ARRAY_MAX_SHIFT`, `HG_SEGMENTED_ARRAY_MAX_SEGMENTS` (capacity ceiling) |
| Stats builds only | `kClaimRing` 8192, `kClaimRingWorkers` 64 (`parallel_evolution.hpp:1278-1279`, inside `#if HG_ENGINE_STATS`), `kSilentWitness`, `kDrainProbes`, `kMaxTimedThreads`, `kAllocSites`, counter slot counts |

Board H6 lists `kClaimRing` and `kClaimRingWorkers` as per-host tuning; both exist only in stats builds and have no effect on a release run.

## 4. Quiet-box queue

Wall-clock checks that need a quiet box. Each names the command, the jobs and the direction the instruction counts predict. The jobs are written by `tools/dev/reply_jobs.py`; the worker count is set with the `get_nprocs` shim that `tools/dev/worker_fuzz.py` compiles into its output directory as `nprocs_shim.so` (`SHIM` below), through `HG_FUZZ_NPROCS`.

**Q1. Replies written by parallel chunks.** In a stats-OFF release directory, at the commit before 36780be6 and at HEAD, five runs per worker count, interleaved:

    python3 tools/dev/reply_jobs.py .scratch/jobs
    for t in 1 4 16 32; do for r in 1 2 3 4 5; do for j in wpp_d7_q_graphs wpp_d7_q_structure wpp_d6_full_evograph wpp_d6_full_all; do
      /usr/bin/time -f "$j $t %e %M" env HG_FUZZ_NPROCS=$t LD_PRELOAD=$SHIM \
        paclet/LibraryResources/Linux-x86-64/hg_evolve < .scratch/jobs/$j.wxf > /dev/null
    done; done; done

Expected: the time after evolution falls as workers are added, and at one worker it is no higher than before. Callgrind at 4 workers: wpp_d7_q_graphs 12.31e9 to 6.55e9 instructions, wpp_d6_full_all 0.98e9 to 0.74e9.

**Q2. StepStatistics pooled in jobs.** The same loop on wpp_d7_q_stats, wpp_d8_q_stats and wpp_d6_auto_stats_all, before and after e95879f2. Expected: the time after evolution falls as workers are added; the last step's pooling, one job before, runs as one job per 256 classes and one per slot.

**Q3. StepStatistics pooled on the device.** `hg_evolve_gpu` on wpp_d8_q_stats, before and after e95879f2, five runs each, with `nvidia-smi` read before and after every run to confirm no other process uses the GPU. Expected: the host time after the device run falls.

The tuning rows below use `bench_cpu_evolve` from a stats-OFF release directory (`build_linux`) and `bench_gpu_evolve` from `build_gpu`; each pair of arms runs interleaved process by process, 5 rounds, and medians are compared.

**Q4. H5 causal set capacity.** Arms `HG_TUNE_CAUSAL_SET_CAPACITY=1024` and unset (8192 on a machine with more than 4 GB available). Rows: multirule depth 6 and 7, cycle4 depth 6, wpp depth 7, all full (`HG_BENCH_CANON_ONLY=0`), at 1, 16 and 32 threads.

    for round in 1 2 3 4 5; do for cap in 1024 ""; do for row in "multirule 6" "multirule 7" "cycle4 6" "wpp 7"; do
      set -- $row
      HG_TUNE_CAUSAL_SET_CAPACITY=$cap HG_BENCH_CANON_ONLY=0 build_linux/bench_cpu_evolve $2 15 1,16,32 $1 \
        > .scratch/h5/cap${cap:-default}_$1_d$2_r$round.log
    done; done; done

Expected: the default at most equal to 1024 on every row (the instruction count falls 1.1-1.5% at one thread).

**Q5. H4 expand chunk.** Arms `HG_TUNE_EXPAND_CHUNK=16` and unset. Rows with one-edge left-hand sides: binary depth 8 and 10, arimix depth 6, allfour depth 5, multirule depth 7, full and quotient, at 4, 16 and 32 threads; bigpath n128 depth 3 quotient as a row the chunk does not reach.

    for round in 1 2 3 4 5; do for chunk in 16 ""; do for row in "binary 8" "binary 10" "arimix 6" "allfour 5" "multirule 7"; do
      set -- $row
      for q in 0 1; do
        HG_TUNE_EXPAND_CHUNK=$chunk HG_BENCH_CANON_ONLY=$q build_linux/bench_cpu_evolve $2 15 4,16,32 $1 \
          > .scratch/h4/chunk${chunk:-derived}_$1_d$2_q${q}_r$round.log
      done
    done; done; done

Expected: the derived size at most equal to 16 at 16 and 32 workers. If 16 is lower on a row, the floor of 4 is the term to raise.

**Q6. D16/D17 first engine.** `build_gpu/bench_gpu_evolve` at the commit before 705dfc8c and at HEAD, five runs each, interleaved, `nvidia-smi` read before and after every run: rows wpp depth 2, 3 and 7, bigpath n256 depth 3 quotient, and a capped row (`"MatchesPerStep" -> 4` with `"UniformRandom"`, depth 20) through `hg_evolve_gpu`; then wpp depth 7 again with another process holding 8 GB of VRAM. Expected: the capped row and the small rows at most equal (smaller pools and clears); no first-attempt failure with VRAM held.

**Q7. D1/D3/D22 launch.** `HG_TUNE_GPU_L2_PERSIST=0` against unset, and `HG_TUNE_GPU_BLOCKS` 512, 1024 and 2048, on `build_gpu/bench_gpu_evolve 7 5 2` and wpp depth 8 quotient, five runs each, interleaved, `nvidia-smi` read before and after every run. Expected: the window at most equal to no window; the grid minimum at 1024 as in the sweep at `persistent.cu`. `ncu --metrics lts__t_sector_hit_rate.pct` on `k_persistent_evolve` with and without the window gives the L2 hit rate (only with the GPU otherwise idle and a kernel under a second).

**Q8. The overlap rows with the state matrices.** `hg_evolve` on rule A from an 8 x 8 grid, 10 steps, `"MaxStatesPerStep" -> 400`, `"StepStatistics"`, `"StepStatisticsBranchial" -> All`, at 1, 4 and 16 workers, master against the commit that adds `"StepStatisticsMatrices"`, 6 interleaved runs each. Expected: equal within the spread (callgrind on `StateStatistics.BranchialMetricsDoNotDependOnTheRowSplit`: 290,543,143 -> 290,848,055 Ir). The same run with `"StepStatisticsMatrices" -> All` gives the cost of the matrices; expected below 2x the overlap-only run.

**Q9. Double-double distribution sums.** `hg_evolve` on `tools/dev/reply_jobs.py` `wpp_d7_q_stats` and `wpp_d8_q_stats`, `HG_TUNE_THREADS` 1, 4 and 16, the binaries of b6998964^ and b6998964, 5 interleaved runs each. Expected: at most 2.4% slower at one worker (callgrind: 10,906,463,240 -> 11,166,579,253 Ir at one worker on `wpp_d7_q_stats`; `sg_dist_add` 165M Ir).

**Q11. The device quotient replay.** `build_gpu/bench_gpu_evolve 7 5 1 multirule` and `3 5 1 bigpath`, `6 5 1 cycle4`, `7 5 1 wpp` (quotient, all records) at 60e4e9f6^ and at HEAD, five runs each, interleaved, `nvidia-smi` read before and after every run; against `build_linux/bench_cpu_evolve 7 5 16 multirule` and the same rows at 16 threads. Expected: the device below 60e4e9f6^ on every row (replay branch cycles 438.6e9 -> 8.67e9 on multirule) and the `[evolve dbg]` recon part (HG_GPU_DBG_TIME=1) a few ms on multirule d7.

**Q12. Per-vertex curvature on the device.** `hg_evolve_gpu` on `tools/dev/reply_jobs.py` `wpp_d6_none_curv` (wpp depth 6, CanonicalizeStates None, "States" with IncludeVertexCurvature) and `wpp_d7_full_curv`, the binaries of the commit before the device curvature and of it, 5 interleaved runs each, `nvidia-smi` read before and after every run. Expected: the reply time falls with the host work moved to the device (the GPU worker at 0.61e9 Ir after, 4 workers; the curvature it no longer runs on the host is 2.65e9 Ir on the CPU worker on the same job).

**Q13. Reply assembly off the calling thread.** `hg_evolve` on `tools/dev/reply_jobs.py` `wpp_d7_q_graphs`, `wpp_d6_full_all` and `wpp_d7_q_stats`, and `hg_evolve_gpu` on `wpp_d6_auto_stats_all` and `wpp_d7_q_stats`, `HG_TUNE_THREADS` 1, 4 and 16, the binaries of 1916dfac^ and of 4e9c0822, 5 interleaved runs each, `nvidia-smi` read before and after every GPU run. Expected: the time after evolution falls with the calling thread's Ir (callgrind at 4 workers: CPU wpp_d7_q_graphs 1,339,455,485 -> 945,435,874, wpp_d7_q_stats 58,742,371 -> 42,009,530; GPU wpp_d6_auto_stats_all 308,739,715 -> 196,225,708, wpp_d7_q_stats 509,290,447 -> 482,861,381) and equal at one worker.

**Q10. Explored branchial pairs recorded only when read.** `hg_evolve` on `wpp_d8_q_counts` (NumStates, NumEvents, NumCausalEdges, NumBranchialEdges, quotient), `HG_TUNE_THREADS` 1, 4 and 16, the binaries of 68fb91d9^ and 68fb91d9, 5 interleaved runs each, wall clock and `/usr/bin/time %M`. Expected: at most equal time; peak RSS 2,114,592-2,156,096 KB before, 2,029,188-2,041,184 KB after (default workers, loaded box).
