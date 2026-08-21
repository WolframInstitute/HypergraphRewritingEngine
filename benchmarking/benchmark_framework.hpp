#pragma once

#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <functional>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <memory>
#include <optional>
#include <cstdlib>
#include <regex>
#include <sys/stat.h>
#include <filesystem>

// Enforce release build
#ifndef NDEBUG
    static_assert(false, "Benchmarks MUST be built in release mode! Use -DCMAKE_BUILD_TYPE=Release");
#endif

namespace benchmark {

// Forward declarations
class BenchmarkRegistry;

// Sanitize string for filesystem use
std::string sanitize_for_filename(const std::string& str);

// Helper function for formatting parameters
std::string params_to_string(const std::map<std::string, std::string>& params);

// Configuration constants
constexpr size_t BENCHMARK_MIN_SAMPLES = 2;  // Absolute minimum for stddev/CI calculation
constexpr size_t BENCHMARK_MAX_SAMPLES = 150;
constexpr double BENCHMARK_CI_THRESHOLD = 0.10;  // 10% relative confidence interval width (±5% at 95% confidence)
constexpr size_t BENCHMARK_WARMUP_RUNS = 2;

/**
 * Result type - determines how values are interpreted and displayed
 */
enum class ResultType {
    TIME,     // Timing result in microseconds (default)
    RATIO,    // Dimensionless ratio (e.g., speedup, efficiency)
    CUSTOM    // Custom units specified in metadata
};

/**
 * Statistical results for a benchmark run
 */
struct BenchmarkResult {
    std::string benchmark_name;
    std::map<std::string, std::string> params;  // param_name -> param_value
    std::map<std::string, std::string> metadata;  // Metadata for visualization (y_label, y_scale, etc.)
    ResultType result_type = ResultType::TIME;
    size_t samples = 0;
    size_t outliers_removed = 0;
    double min_us = 0.0;
    double max_us = 0.0;
    double avg_us = 0.0;
    double stddev_us = 0.0;
    double median_us = 0.0;  // Robust central tendency (primary metric)
    double mad_us = 0.0;     // Median absolute deviation (robust variance)
    double ci_lower_us = 0.0;  // 95% confidence interval lower bound
    double ci_upper_us = 0.0;  // 95% confidence interval upper bound
    double ci_width_percent = 0.0;  // relative CI width: (ci_upper - ci_lower) / avg * 100
    std::vector<double> raw_timings_us;

    // Sub-timings: name -> vector of timings (one per sample)
    std::map<std::string, std::vector<double>> sub_timings_us;

    // Aggregated sub-timing statistics: name -> {avg, stddev, min, max}
    struct SubTimingStats {
        double avg_us = 0.0;
        double stddev_us = 0.0;
        double min_us = 0.0;
        double max_us = 0.0;
    };
    std::map<std::string, SubTimingStats> sub_timing_stats;

    // Convergence tracking: snapshot of stats after each sample
    struct ConvergencePoint {
        size_t sample_num;
        double cumulative_avg_us;
        double cumulative_stddev_us;
        double cumulative_median_us;
        double ci_width_percent;
    };
    std::vector<ConvergencePoint> convergence_history;
};

/**
 * Get t-distribution critical value for 95% confidence (two-tailed)
 */
double get_t_value(size_t n);

/**
 * Calculate percentile from sorted data (linear interpolation)
 */
double percentile(const std::vector<double>& sorted_data, double p);

/**
 * Calculate median from sorted data
 */
double median(const std::vector<double>& sorted_data);

/**
 * Calculate Median Absolute Deviation (MAD) - robust measure of variance
 * Returns MAD scaled by 1.4826 to approximate standard deviation for normal distributions
 */
double median_absolute_deviation(const std::vector<double>& data);

/**
 * Filter outliers using Tukey's method (IQR-based)
 * Uses asymmetric bounds: conservative on low end (perf can't be < 0),
 * wider on high end (allows legitimate variance from OS scheduling)
 */
std::vector<double> filter_outliers(const std::vector<double>& data, size_t& outliers_removed);

/**
 * Calculate statistics from timing samples
 */
void calculate_statistics(BenchmarkResult& result);

/**
 * Extract git commit hash and date
 */
struct GitInfo {
    std::string hash;          // either commit hash or tree hash
    std::string hash_type;     // "commit" or "tree"
    std::string commit_date;   // YYYY-MM-DD format

    static std::string run_git_cmd(const char* git_cmd);

    static GitInfo get();
};

/**
 * Get current timestamp in ISO 8601 format
 */
std::string get_timestamp();

/**
 * Benchmark context - tracks current benchmark and parameters
 */
class BenchmarkContext {
public:
    std::string current_benchmark;
    std::map<std::string, std::string> current_params;
    std::map<std::string, std::string> current_metadata;
    ResultType current_result_type = ResultType::TIME;

    // Sub-timing tracking
    std::map<std::string, std::chrono::high_resolution_clock::time_point> timing_starts;
    std::map<std::string, std::vector<double>> current_sub_timings;

    // Manual timing tracking (BENCHMARK_BEGIN/END and BENCHMARK_SUBMIT)
    std::chrono::high_resolution_clock::time_point manual_timing_start;
    std::vector<double> manual_timings;
    bool inside_benchmark_code = false;

    void reset();

    void set_benchmark(const std::string& name);

    void set_param(const std::string& name, const std::string& value);

    template<typename T>
    void set_param(const std::string& name, const T& value) {
        std::ostringstream oss;
        oss << value;
        current_params[name] = oss.str();
    }

    void start_timing(const std::string& name);

    void stop_timing(const std::string& name);

    void reset_sub_timings();

    std::map<std::string, std::vector<double>> get_sub_timings() const;
};

/**
 * Global benchmark context (singleton)
 */
BenchmarkContext& get_context();

/**
 * Adaptive sampler - runs benchmark until variance converges
 */
class AdaptiveSampler {
public:
    template<typename Func, typename SetupFunc = std::nullptr_t>
    static BenchmarkResult run(const std::string& benchmark_name,
                              const std::map<std::string, std::string>& params,
                              Func&& benchmark_code,
                              SetupFunc&& setup_code = nullptr) {
        BenchmarkResult result;
        result.benchmark_name = benchmark_name;
        result.params = params;

        // Use adaptive sampling
        printf("[   TIMING ] Adaptive sampling (target CI width < %.1f%%)...\n", BENCHMARK_CI_THRESHOLD * 100.0);
        for (size_t i = 0; i < BENCHMARK_MIN_SAMPLES; ++i) {
            get_context().reset_sub_timings();
            double timing = time_single_run(benchmark_code, setup_code);

            // If NOT using manual timings, collect automatic timing
            auto& ctx = get_context();
            if (!ctx.inside_benchmark_code || ctx.manual_timings.empty()) {
                result.raw_timings_us.push_back(timing);
            }

            // Collect sub-timings from this sample
            auto sub_timings = get_context().get_sub_timings();
            for (const auto& [name, timings] : sub_timings) {
                for (double t : timings) {
                    result.sub_timings_us[name].push_back(t);
                }
            }

            // If manual timings are being collected, use those for statistics
            if (ctx.inside_benchmark_code && !ctx.manual_timings.empty()) {
                result.raw_timings_us = ctx.manual_timings;
            }

            calculate_statistics(result);

            // Record convergence point
            BenchmarkResult::ConvergencePoint point;
            point.sample_num = result.raw_timings_us.size();
            point.cumulative_avg_us = result.avg_us;
            point.cumulative_stddev_us = result.stddev_us;
            point.cumulative_median_us = result.median_us;
            point.ci_width_percent = result.ci_width_percent;
            result.convergence_history.push_back(point);
        }

        // Keep adding samples until CI converges or max samples reached
        while (result.raw_timings_us.size() < BENCHMARK_MAX_SAMPLES) {
            // If manual timings are being collected, use those for convergence checking
            auto& ctx = get_context();
            if (ctx.inside_benchmark_code && !ctx.manual_timings.empty()) {
                result.raw_timings_us = ctx.manual_timings;
            }

            calculate_statistics(result);

            if (result.ci_width_percent < BENCHMARK_CI_THRESHOLD * 100.0) {
                printf("[   TIMING ]   Converged at %zu samples (CI width: %.2f%%)\n",
                       result.raw_timings_us.size(), result.ci_width_percent);
                break;  // CI converged
            }

            // Add 1 more sample
            printf("\r[   TIMING ]   CI width %.2f%% > %.1f%%, adding 1 more sample...   ",
                   result.ci_width_percent, BENCHMARK_CI_THRESHOLD * 100.0);
            fflush(stdout);

            get_context().reset_sub_timings();
            double timing = time_single_run(benchmark_code, setup_code);

            // If NOT using manual timings, collect automatic timing
            if (!ctx.inside_benchmark_code || ctx.manual_timings.empty()) {
                result.raw_timings_us.push_back(timing);
            }

            // Collect sub-timings from this sample
            auto sub_timings = get_context().get_sub_timings();
            for (const auto& [name, timings] : sub_timings) {
                for (double t : timings) {
                    result.sub_timings_us[name].push_back(t);
                }
            }

            // If manual timings are being collected, sync them now
            if (ctx.inside_benchmark_code && !ctx.manual_timings.empty()) {
                result.raw_timings_us = ctx.manual_timings;
            }

            calculate_statistics(result);

            // Record convergence point
            BenchmarkResult::ConvergencePoint point;
            point.sample_num = result.raw_timings_us.size();
            point.cumulative_avg_us = result.avg_us;
            point.cumulative_stddev_us = result.stddev_us;
            point.cumulative_median_us = result.median_us;
            point.ci_width_percent = result.ci_width_percent;
            result.convergence_history.push_back(point);
        }

        // Final sync if using manual timings
        auto& ctx = get_context();
        if (ctx.inside_benchmark_code && !ctx.manual_timings.empty()) {
            result.raw_timings_us = ctx.manual_timings;
        }

        calculate_statistics(result);
        return result;
    }

private:
    template<typename Func, typename SetupFunc>
    static double time_single_run(Func&& benchmark_code, SetupFunc&& setup_code) {
        if constexpr (!std::is_same_v<std::decay_t<SetupFunc>, std::nullptr_t>) {
            setup_code();  // Run setup, not timed
        }
        auto start = std::chrono::high_resolution_clock::now();
        benchmark_code();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(end - start).count();
    }
};

/**
 * CSV writer with atomic file operations
 */
class CSVWriter {
public:
    static void ensure_directory(const std::string& path);

    static void write_file(const std::string& filepath, const std::string& content);

    static void append_file(const std::string& filepath, const std::string& content);
};

/**
 * Centralized CSV column schema - SINGLE SOURCE OF TRUTH
 * Used by both BenchmarkRegistry and ReferenceDataLoader
 */
struct CSVSchema {
    enum class ColumnType {
        INTEGER,
        FLOATING_POINT
    };

    struct ColumnMetadata {
        std::string name;
        ColumnType type;
    };

    // Column metadata in exact order (after benchmark_name, params, metadata, result_type)
    static std::vector<ColumnMetadata> get_column_metadata();

    // Column names in exact order (backward compatibility)
    static std::vector<std::string> get_column_names();

    // Extract column values from result in same order
    static std::vector<double> get_column_values(const BenchmarkResult& result);

    // Format a column value based on its type
    static std::string format_column_value(double value, ColumnType type);

    // Write column values with proper formatting
    static void write_column_values(std::ostream& os, const BenchmarkResult& result);

    // Parse column values into result in same order
    static void parse_column_values(BenchmarkResult& result, const std::vector<std::string>& tokens);

    // Total number of base columns (including benchmark_name, params, metadata, result_type + data columns)
    static size_t base_column_count();
};

/**
 * Reference data loader - lazily loads reference benchmark data from CSV
 */
class ReferenceDataLoader {
public:
    static ReferenceDataLoader& instance();

    // Get reference timing for a benchmark with specific parameters
    std::optional<double> get_reference_timing(const std::string& benchmark_name,
                                               const std::map<std::string, std::string>& params);

    // Check if reference data exists for a benchmark
    bool has_reference_data(const std::string& benchmark_name);

    // Clear loaded data to force reload (after generating new reference data)
    void invalidate_cache(const std::string& benchmark_name);

private:
    struct ReferenceEntry {
        std::map<std::string, std::string> params;
        double avg_us;
    };

    std::map<std::string, std::vector<ReferenceEntry>> data_;
    std::set<std::string> loaded_benchmarks_;

    void load_if_needed(const std::string& benchmark_name);

    ReferenceDataLoader() = default;
};

/**
 * Benchmark registry - stores all registered benchmarks
 */
class BenchmarkRegistry {
public:
    using BenchmarkFunc = std::function<void()>;

    static BenchmarkRegistry& instance();

    void register_benchmark(const std::string& name, BenchmarkFunc func, const std::string& description = "", bool is_reference = false);

    void register_dependency(const std::string& benchmark_name, const std::string& reference_name);

    const std::map<std::string, BenchmarkFunc>& get_benchmarks() const;

    std::string get_description(const std::string& name) const;

    bool is_reference(const std::string& name) const;

private:
    // Helper to create a gtest-style filter matcher
    // Returns a lambda that tests if a name matches the filter pattern
    static std::function<bool(const std::string&)> create_filter_matcher(const std::string& filter);

public:
    void list_benchmarks(const std::string& filter = "") const;

    // Run a single reference benchmark and save results
    void run_reference_benchmark(const std::string& name, const std::string& output_dir);

    void run_all(const std::string& output_dir, const std::string& filter = "", bool include_reference = false, bool only_reference = false);

    void add_result(const BenchmarkResult& result);

    const BenchmarkResult& get_last_result() const;

private:
    std::map<std::string, BenchmarkFunc> benchmarks_;
    std::map<std::string, std::string> descriptions_;
    std::map<std::string, bool> reference_flags_;
    std::map<std::string, std::vector<std::string>> dependencies_;
    std::vector<BenchmarkResult> current_results_;
    BenchmarkResult last_result_;

    BenchmarkRegistry() = default;

    static std::string make_column_name(const std::string& benchmark_name,
                                       const std::map<std::string, std::string>& params,
                                       const std::string& metric);

    static std::vector<BenchmarkResult> read_all_benchmark_results(const std::string& results_dir);

    static void write_benchmark_result_csv(const std::string& results_dir, const BenchmarkResult& result);

    static void write_summary_csv(const std::string& dir, const GitInfo& git_info,
                                  const std::string& timestamp,
                                  const std::vector<BenchmarkResult>& results);

    static void write_detailed_csv(const std::string& dir, const std::vector<BenchmarkResult>& results);

    static void write_raw_timings_csv(const std::string& dir, const std::vector<BenchmarkResult>& results);

    static void write_samples_convergence_csv(const std::string& dir, const std::vector<BenchmarkResult>& results);

    static void write_config_txt(const std::string& dir, const GitInfo& git_info, const std::string& timestamp);
};

/**
 * Helper for benchmark registration
 */
struct BenchmarkRegistration {    BenchmarkRegistration(const std::string& name, BenchmarkRegistry::BenchmarkFunc func, const std::string& description = "", bool is_reference = false);
};

/**
 * Macros for defining benchmarks (GoogleTest-style)
 * Usage:
 *   BENCHMARK(name) { ... }                              // No description
 *   BENCHMARK(name, "description") { ... }               // With description
 */
#define BENCHMARK(...) \
    BENCHMARK_IMPL(__VA_ARGS__, BENCHMARK_WITH_DESC, BENCHMARK_NO_DESC)(__VA_ARGS__)

#define BENCHMARK_IMPL(_1, _2, NAME, ...) NAME

#define BENCHMARK_NO_DESC(name) \
    void benchmark_##name(); \
    static benchmark::BenchmarkRegistration reg_##name(#name, benchmark_##name); \
    void benchmark_##name()

#define BENCHMARK_WITH_DESC(name, desc) \
    void benchmark_##name(); \
    static benchmark::BenchmarkRegistration reg_##name(#name, benchmark_##name, desc); \
    void benchmark_##name()

#define BENCHMARK_WITH_REFERENCE(name, desc, reference_name) \
    void benchmark_##name(); \
    static benchmark::BenchmarkRegistration reg_##name(#name, benchmark_##name, desc); \
    static void __attribute__((constructor)) register_dep_##name() { \
        benchmark::BenchmarkRegistry::instance().register_dependency(#name, reference_name); \
    } \
    void benchmark_##name()

#define BENCHMARK_REFERENCE(name, desc) \
    void benchmark_##name(); \
    static benchmark::BenchmarkRegistration reg_##name(#name, benchmark_##name, desc, true); \
    void benchmark_##name()

/**
 * Helper to set parameter in current context
 */
template<typename T>
inline void BENCHMARK_PARAM(const std::string& name, const T& value) {
    benchmark::get_context().set_param(name, value);
}

/**
 * Helper to run benchmark code with timing
 * Supports both automatic timing and manual timing (BENCHMARK_BEGIN/END or BENCHMARK_SUBMIT)
 */
template<typename Func>
inline void BENCHMARK_CODE(Func&& code) {
    auto& ctx = benchmark::get_context();
    ctx.manual_timings.clear();
    ctx.inside_benchmark_code = true;

    auto result = benchmark::AdaptiveSampler::run(
        ctx.current_benchmark,
        ctx.current_params,
        std::forward<Func>(code)
    );

    ctx.inside_benchmark_code = false;

    // If manual timings were collected, use those instead
    if (!ctx.manual_timings.empty()) {
        result.raw_timings_us = std::move(ctx.manual_timings);
        ctx.manual_timings.clear();
        calculate_statistics(result);
    }

    // Set metadata and result type
    result.metadata = ctx.current_metadata;
    result.result_type = ctx.current_result_type;

    benchmark::BenchmarkRegistry::instance().add_result(result);

    // Clear params and metadata for next benchmark
    ctx.current_params.clear();
    ctx.current_metadata.clear();
    ctx.current_result_type = ResultType::TIME;
}

/**
 * Manual timing control - begin benchmark timing block
 * Usage: Setup code, then BENCHMARK_BEGIN(), then code to time, then BENCHMARK_END()
 */
void BENCHMARK_BEGIN();

/**
 * Manual timing control - end benchmark timing block and record sample
 */
void BENCHMARK_END();

/**
 * Submit manually collected timings (from BENCHMARK_BEGIN/END)
 */
void BENCHMARK_SUBMIT_MANUAL();

/**
 * Helper to submit pre-collected timings (for external benchmarks like WolframScript)
 * Can be used both inside BENCHMARK_CODE (for adaptive sampling) or outside (for complete results)
 *
 * Usage inside BENCHMARK_CODE:
 *   BENCHMARK_CODE([&]() {
 *       double timing = run_external_tool();
 *       BENCHMARK_SUBMIT(timing);  // Single value, will be collected for adaptive sampling
 *   });
 *
 * Usage outside BENCHMARK_CODE:
 *   BENCHMARK_PARAM("size", 100);
 *   BENCHMARK_SUBMIT({t1, t2, t3});  // Complete result set
 */
void BENCHMARK_SUBMIT(const std::vector<double>& timings_us, ResultType type = ResultType::TIME);

// Overload for single value submission (convenience)
void BENCHMARK_SUBMIT(double timing_us, ResultType type = ResultType::TIME);

/**
 * Helper to submit results to a different benchmark name
 * Useful for submitting related metrics with different units (e.g., timings vs speedup)
 */
void BENCHMARK_SUBMIT_AS(const std::string& benchmark_name, const std::vector<double>& timings_us, ResultType type = ResultType::TIME);

// Overload for single value submission
void BENCHMARK_SUBMIT_AS(const std::string& benchmark_name, double timing_us, ResultType type = ResultType::TIME);

/**
 * Helper to set benchmark metadata (for plotting customization)
 * Common keys: "y_label" (e.g., "Speedup (x)", "Time (μs)"), "y_scale" (e.g., "linear", "log")
 */
void BENCHMARK_META(const std::string& key, const std::string& value);

/**
 * Sub-timing macros for measuring portions of a benchmark
 */
#define BENCHMARK_TIMING_START(name) \
    benchmark::get_context().start_timing(name)

#define BENCHMARK_TIMING_STOP(name) \
    benchmark::get_context().stop_timing(name)

} // namespace benchmark

