#include <hypergraph/wolfram_evolution.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <iomanip>

using namespace hypergraph;

void thread_scaling_benchmark() {
    printf("=== Thread Scaling Performance Benchmark ===\n");
    printf("Testing hypergraph rewriting performance across different thread counts\n\n");
    
    // Test rule: {{x,y},{y,z}} -> {{x,y},{y,z},{y,w}}
    // This creates branching and exercises the parallel system
    PatternHypergraph lhs, rhs;
    
    // LHS: {{x,y},{y,z}}
    lhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(2)
    });
    lhs.add_edge(PatternEdge{
        PatternVertex::variable(2), PatternVertex::variable(3)
    });
    
    // RHS: {{x,y},{y,z},{y,w}} - note w is fresh variable
    rhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(2)
    });
    rhs.add_edge(PatternEdge{
        PatternVertex::variable(2), PatternVertex::variable(3)
    });
    rhs.add_edge(PatternEdge{
        PatternVertex::variable(2), PatternVertex::variable(4)  // w vertex
    });
    
    RewritingRule rule(lhs, rhs);
    
    // Test with complex initial state for better parallelization
    std::vector<std::vector<GlobalVertexId>> initial_state = {
        {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}
    };
    
    printf("Rule: {{x,y},{y,z}} -> {{x,y},{y,z},{y,w}}\n");
    printf("Initial state: {{1,2},{2,3},{3,4},{4,5},{5,6}}\n");
    printf("Evolution steps: 3\n\n");
    
    printf("%-8s %-12s %-8s %-8s %-10s %-12s %-12s\n", 
           "Threads", "Time (μs)", "States", "Events", "Causal", "Branchial", "Speedup");
    printf("------------------------------------------------------------------------\n");
    
    std::chrono::microseconds baseline_time{0};
    
    // Test different thread counts
    std::vector<int> thread_counts = {1, 2, 4, 8, 16};
    for (int threads : thread_counts) {
        if (threads > std::thread::hardware_concurrency()) {
            break;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        WolframEvolution evolution(3, threads, true, false);
        evolution.add_rule(rule);
        evolution.evolve(initial_state);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        if (threads == 1) {
            baseline_time = duration;
        }
        
        const auto& graph = evolution.get_multiway_graph();
        double speedup = baseline_time.count() / static_cast<double>(duration.count());
        
        printf("%-8d %-12ld %-8zu %-8zu %-10zu %-12zu %.2fx\n",
               threads, duration.count(), graph.num_states(), graph.num_events(),
               graph.get_causal_edge_count(), graph.get_branchial_edge_count(), speedup);
    }
    
    printf("\n");
}

void complexity_scaling_benchmark() {
    printf("=== Complexity Scaling Benchmark ===\n");
    printf("Testing performance vs evolution complexity\n\n");
    
    // Simple growth rule: {{x,y}} -> {{x,y},{y,z}}  
    PatternHypergraph lhs, rhs;
    lhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(2)
    });
    rhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(2)
    });
    rhs.add_edge(PatternEdge{
        PatternVertex::variable(2), PatternVertex::variable(3)
    });
    
    RewritingRule rule(lhs, rhs);
    std::vector<std::vector<GlobalVertexId>> initial_state = {{1, 2}};
    
    printf("Rule: {{x,y}} -> {{x,y},{y,z}}\n");
    printf("Initial state: {{1,2}}\n");
    printf("Using all available threads: %d\n\n", std::thread::hardware_concurrency());
    
    printf("%-6s %-12s %-8s %-8s %-10s %-12s\n", 
           "Steps", "Time (μs)", "States", "Events", "Causal", "Branchial");
    printf("----------------------------------------------------------\n");
    
    for (int steps : {1, 2, 3, 4, 5}) {
        auto start = std::chrono::high_resolution_clock::now();
        
        WolframEvolution evolution(steps, std::thread::hardware_concurrency(), true, false);
        evolution.add_rule(rule);
        evolution.evolve(initial_state);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        const auto& graph = evolution.get_multiway_graph();
        
        printf("%-6d %-12ld %-8zu %-8zu %-10zu %-12zu\n",
               steps, duration.count(), graph.num_states(), graph.num_events(),
               graph.get_causal_edge_count(), graph.get_branchial_edge_count());
    }
    
    printf("\n");
}

void determinism_benchmark() {
    printf("=== Determinism Verification Benchmark ===\n");
    printf("Verifying consistent results across multiple runs\n\n");
    
    // Rule: {{x,y}} -> {{x,z},{z,y}}
    PatternHypergraph lhs, rhs;
    lhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(2)
    });
    rhs.add_edge(PatternEdge{
        PatternVertex::variable(1), PatternVertex::variable(3)
    });
    rhs.add_edge(PatternEdge{
        PatternVertex::variable(3), PatternVertex::variable(2)
    });
    
    RewritingRule rule(lhs, rhs);
    std::vector<std::vector<GlobalVertexId>> initial_state = {{1, 2}, {3, 4}};
    
    struct Result {
        std::size_t states, events, causal, branchial;
        bool operator==(const Result& other) const {
            return states == other.states && events == other.events &&
                   causal == other.causal && branchial == other.branchial;
        }
    };
    
    printf("Rule: {{x,y}} -> {{x,z},{z,y}}\n");
    printf("Initial state: {{1,2},{3,4}}\n");
    printf("Evolution steps: 2\n");
    printf("Threads: %d\n\n", std::thread::hardware_concurrency());
    
    std::vector<Result> results;
    std::vector<long> times;
    
    for (int run = 1; run <= 5; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        
        WolframEvolution evolution(2, std::thread::hardware_concurrency(), true, false);
        evolution.add_rule(rule);
        evolution.evolve(initial_state);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        const auto& graph = evolution.get_multiway_graph();
        results.push_back({graph.num_states(), graph.num_events(),
                          graph.get_causal_edge_count(), graph.get_branchial_edge_count()});
        times.push_back(duration.count());
        
        printf("Run %d: States=%zu, Events=%zu, Causal=%zu, Branchial=%zu, Time=%ld μs\n",
               run, results.back().states, results.back().events, 
               results.back().causal, results.back().branchial, times.back());
    }
    
    // Check determinism
    bool deterministic = true;
    for (size_t i = 1; i < results.size(); ++i) {
        if (!(results[i] == results[0])) {
            deterministic = false;
            break;
        }
    }
    
    if (deterministic) {
        printf("\n✅ DETERMINISTIC: All runs produced identical results\n");
    } else {
        printf("\n❌ NON-DETERMINISTIC: Results varied between runs\n");
    }
    
    // Calculate average time
    long avg_time = 0;
    for (long time : times) avg_time += time;
    avg_time /= times.size();
    printf("Average time: %ld μs\n\n", avg_time);
}

int main() {
    printf("Hypergraph Rewriting Performance Benchmark Suite\n");
    printf("===============================================\n\n");
    
    try {
        thread_scaling_benchmark();
        complexity_scaling_benchmark();
        determinism_benchmark();
        
        printf("✅ All benchmarks completed successfully!\n");
        
    } catch (const std::exception& e) {
        printf("❌ Error during benchmarking: %s\n", e.what());
        return 1;
    }
    
    return 0;
}