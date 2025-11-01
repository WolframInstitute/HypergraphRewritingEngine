# Hypergraph Engine Benchmarks - Plots Only

## System Information

- **CPU**: Intel(R) Core(TM) i9-14900K
- **Cores**: 32
- **Architecture**: x86_64
- **OS**: Linux 5.15.167.4-microsoft-standard-WSL2
- **Memory**: 23 GB
- **Compiler**: GNU 13.3.0
- **Hash Type**: commit
- **Hash**: b6c7372ea3355015e3ab9cb39a71a5629f21a958
- **Date**: 2025-10-30 01:01:23
- **Timestamp**: 2025-11-01T03:21:27

## Pattern Matching Benchmarks

### pattern_matching_2d_sweep_threads_size

2D parameter sweep of pattern matching across thread count (1-32) and graph size (5-100 edges) for parallel scalability analysis

![pattern_matching_2d_sweep_threads_size](plots/pattern_matching_2d_sweep_threads_size_2d.png)

### pattern_matching_by_graph_size

Evaluates pattern matching scalability as target graph size increases from 5 to 15 edges

![pattern_matching_by_graph_size](plots/pattern_matching_by_graph_size_1d.png)

### pattern_matching_by_pattern_size

2D parameter sweep: pattern matching time vs pattern complexity (1-5 edges) and graph size (5-15 edges)

![pattern_matching_by_pattern_size](plots/pattern_matching_by_pattern_size_2d.png)

## Event Relationships Benchmarks

### causal_edges_overhead

Measures the overhead of computing causal edges during evolution (1-3 steps)

![causal_edges_overhead](plots/causal_edges_overhead_2d.png)

### transitive_reduction_overhead

Isolates transitive reduction overhead by comparing evolution with it enabled vs disabled

![transitive_reduction_overhead](plots/transitive_reduction_overhead_2d.png)

## Uniqueness Trees Benchmarks

### uniqueness_tree_2d_sweep

2D parameter sweep: edges vs symmetry_groups for surface plots

![uniqueness_tree_2d_sweep](plots/uniqueness_tree_2d_sweep_2d.png)

### uniqueness_tree_by_arity

Tests impact of hyperedge arity on performance

![uniqueness_tree_by_arity](plots/uniqueness_tree_by_arity_1d.png)

### uniqueness_tree_by_edge_count

Measures uniqueness tree performance as graph size increases (arity=2)

![uniqueness_tree_by_edge_count](plots/uniqueness_tree_by_edge_count_1d.png)

### uniqueness_tree_by_edge_count_arity3

Measures uniqueness tree performance as graph size increases (arity=3, higher complexity)

![uniqueness_tree_by_edge_count_arity3](plots/uniqueness_tree_by_edge_count_arity3_1d.png)

### uniqueness_tree_by_symmetry

Shows how graph symmetry affects uniqueness tree time

![uniqueness_tree_by_symmetry](plots/uniqueness_tree_by_symmetry_1d.png)

### uniqueness_tree_by_vertex_count

Measures performance as vertex count increases

![uniqueness_tree_by_vertex_count](plots/uniqueness_tree_by_vertex_count_1d.png)

## Evolution Benchmarks

### evolution_2d_sweep_threads_steps

2D sweep: evolution with rule {{1,2},{2,3}} -> {{3,2},{2,1},{1,4}} on init {{1,1},{1,1}} across thread count and steps

![evolution_2d_sweep_threads_steps](plots/evolution_2d_sweep_threads_steps_2d.png)

### evolution_multi_rule_by_rule_count

Tests evolution performance with increasing rule complexity (1-3 rules with mixed arities)

![evolution_multi_rule_by_rule_count](plots/evolution_multi_rule_by_rule_count_1d.png)

### evolution_thread_scaling

Evaluates parallel speedup from 1 thread up to full hardware concurrency (3-step evolution)

![evolution_thread_scaling](plots/evolution_thread_scaling_1d.png)

### evolution_with_self_loops

Tests evolution performance on hypergraphs containing self-loop edges

![evolution_with_self_loops](plots/evolution_with_self_loops_1d.png)

## Job System Benchmarks

### job_system_2d_sweep

2D parameter sweep of job system across thread count and batch size for parallel scalability analysis

![job_system_2d_sweep](plots/job_system_2d_sweep_2d.png)

### job_system_overhead

Measures job system overhead with minimal workload across varying batch sizes

![job_system_overhead](plots/job_system_overhead_1d.png)

### job_system_scaling_efficiency

Evaluates parallel efficiency with fixed total work across different thread counts

![job_system_scaling_efficiency](plots/job_system_scaling_efficiency_1d.png)

## Canonicalization Benchmarks

### canonicalization_2d_sweep

2D parameter sweep: edges vs symmetry_groups for surface plots

![canonicalization_2d_sweep](plots/canonicalization_2d_sweep_2d.png)

### canonicalization_by_edge_count

Measures canonicalization performance as graph size increases (arity=2)

![canonicalization_by_edge_count](plots/canonicalization_by_edge_count_1d.png)

### canonicalization_by_edge_count_arity3

Measures canonicalization performance as graph size increases (arity=3, higher complexity)

![canonicalization_by_edge_count_arity3](plots/canonicalization_by_edge_count_arity3_1d.png)

### canonicalization_by_symmetry

Shows how graph symmetry affects canonicalization time

![canonicalization_by_symmetry](plots/canonicalization_by_symmetry_1d.png)

## WXF Serialization Benchmarks

### wxf_deserialize_flat_list

Measures WXF deserialization time for flat integer lists of varying sizes

![wxf_deserialize_flat_list](plots/wxf_deserialize_flat_list_1d.png)

### wxf_deserialize_nested_list

Measures WXF deserialization time for nested lists (outer_size x inner_size)

![wxf_deserialize_nested_list](plots/wxf_deserialize_nested_list_2d.png)

### wxf_roundtrip

Measures WXF round-trip (serialize + deserialize) time for various data sizes

![wxf_roundtrip](plots/wxf_roundtrip_1d.png)

### wxf_serialize_flat_list

Measures WXF serialization time for flat integer lists of varying sizes

![wxf_serialize_flat_list](plots/wxf_serialize_flat_list_1d.png)

### wxf_serialize_nested_list

Measures WXF serialization time for nested lists (outer_size x inner_size)

![wxf_serialize_nested_list](plots/wxf_serialize_nested_list_2d.png)

## State Management Benchmarks

### full_capture_overhead

Compares evolution performance with and without full state capture enabled

![full_capture_overhead](plots/full_capture_overhead_1d.png)

### state_storage_by_steps

Measures state storage and retrieval overhead as evolution progresses from 1 to 3 steps

![state_storage_by_steps](plots/state_storage_by_steps_1d.png)

## Other Benchmarks

### comparative_2d_edges_steps

![comparative_2d_edges_steps](plots/comparative_2d_edges_steps_2d.png)

### comparative_2d_edges_steps_speedup

![comparative_2d_edges_steps_speedup](plots/comparative_2d_edges_steps_speedup_2d.png)

### comparative_config1

![comparative_config1](plots/comparative_config1_1d.png)

### comparative_config1_speedup

![comparative_config1_speedup](plots/comparative_config1_speedup_1d.png)

### comparative_config2

![comparative_config2](plots/comparative_config2_1d.png)

### comparative_config2_speedup

![comparative_config2_speedup](plots/comparative_config2_speedup_1d.png)

### comparative_config3

![comparative_config3](plots/comparative_config3_1d.png)

### comparative_config3_speedup

![comparative_config3_speedup](plots/comparative_config3_speedup_1d.png)
