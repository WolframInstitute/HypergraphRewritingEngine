#ifndef BENCHMARK_FRAMEWORK_HPP
#define BENCHMARK_FRAMEWORK_HPP

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
inline std::string sanitize_for_filename(const std::string& str) {
    std::string result = str;
    for (char& c : result) {
        if (c == '/' || c == '\\' || c == ':' || c == '*' || c == '?' ||
            c == '"' || c == '<' || c == '>' || c == '|') {
            c = '_';
        }
    }
    return result;
}

// Helper function for formatting parameters
inline std::string params_to_string(const std::map<std::string, std::string>& params) {
    std::ostringstream oss;
    bool first = true;
    for (const auto& [key, value] : params) {
        if (!first) oss << ", ";
        oss << key << "=" << value;
        first = false;
    }
    return oss.str();
}

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
inline double get_t_value(size_t n) {
    if (n < 2) return 0.0;
    if (n == 2) return 12.706;
    if (n == 3) return 4.303;
    if (n == 4) return 3.182;
    if (n == 5) return 2.776;
    if (n <= 10) return 2.262;
    if (n <= 20) return 2.093;
    if (n <= 30) return 2.045;
    if (n <= 40) return 2.021;
    if (n <= 60) return 2.000;
    if (n <= 100) return 1.984;
    return 1.96;  // For large n, approaches normal distribution
}

/**
 * Calculate percentile from sorted data (linear interpolation)
 */
inline double percentile(const std::vector<double>& sorted_data, double p) {
    if (sorted_data.empty()) return 0.0;
    if (sorted_data.size() == 1) return sorted_data[0];

    double index = (p / 100.0) * (sorted_data.size() - 1);
    size_t lower = static_cast<size_t>(std::floor(index));
    size_t upper = static_cast<size_t>(std::ceil(index));

    if (lower == upper) {
        return sorted_data[lower];
    }

    double weight = index - lower;
    return sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight;
}

/**
 * Calculate median from sorted data
 */
inline double median(const std::vector<double>& sorted_data) {
    return percentile(sorted_data, 50.0);
}

/**
 * Calculate Median Absolute Deviation (MAD) - robust measure of variance
 * Returns MAD scaled by 1.4826 to approximate standard deviation for normal distributions
 */
inline double median_absolute_deviation(const std::vector<double>& data) {
    if (data.empty()) return 0.0;

    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    double med = median(sorted);

    std::vector<double> abs_deviations;
    abs_deviations.reserve(data.size());
    for (double val : data) {
        abs_deviations.push_back(std::abs(val - med));
    }
    std::sort(abs_deviations.begin(), abs_deviations.end());

    double mad = median(abs_deviations);
    return mad * 1.4826;  // Scale to approximate stddev for normal distributions
}

/**
 * Filter outliers using Tukey's method (IQR-based)
 * Uses asymmetric bounds: conservative on low end (perf can't be < 0),
 * wider on high end (allows legitimate variance from OS scheduling)
 */
inline std::vector<double> filter_outliers(const std::vector<double>& data, size_t& outliers_removed) {
    outliers_removed = 0;
    if (data.size() < 4) return data;  // Need at least 4 samples for quartiles

    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());

    double Q1 = percentile(sorted, 25.0);
    double Q3 = percentile(sorted, 75.0);
    double IQR = Q3 - Q1;

    // Tukey's fences: asymmetric for right-skewed benchmark distributions
    double lower_fence = Q1 - 1.5 * IQR;  // Standard lower bound
    double upper_fence = Q3 + 3.0 * IQR;  // Wider upper bound for legitimate spikes

    std::vector<double> filtered;
    filtered.reserve(data.size());

    for (double val : sorted) {
        if (val >= lower_fence && val <= upper_fence) {
            filtered.push_back(val);
        } else {
            outliers_removed++;
        }
    }

    return filtered.empty() ? data : filtered;  // Fallback if all filtered
}

/**
 * Calculate statistics from timing samples
 */
inline void calculate_statistics(BenchmarkResult& result) {
    if (result.raw_timings_us.empty()) return;

    result.samples = result.raw_timings_us.size();

    // Filter outliers using Tukey's method (IQR-based)
    std::vector<double> filtered = filter_outliers(result.raw_timings_us, result.outliers_removed);
    size_t n = filtered.size();

    // Sort for percentile calculations
    std::vector<double> sorted = filtered;
    std::sort(sorted.begin(), sorted.end());

    // Robust statistics (PRIMARY METRICS)
    result.median_us = median(sorted);
    result.mad_us = median_absolute_deviation(filtered);

    // Min/Max from filtered data
    result.min_us = sorted.front();
    result.max_us = sorted.back();

    // Mean and stddev from filtered data (for compatibility/comparison)
    double sum = 0.0;
    for (double t : filtered) sum += t;
    result.avg_us = sum / n;

    double sq_sum = 0.0;
    for (double t : filtered) {
        double diff = t - result.avg_us;
        sq_sum += diff * diff;
    }
    result.stddev_us = (n > 1) ? std::sqrt(sq_sum / (n - 1)) : 0.0;  // Sample stddev

    // 95% Confidence interval using filtered data
    double t_value = get_t_value(n);
    double margin = t_value * result.stddev_us / std::sqrt(static_cast<double>(n));
    result.ci_lower_us = result.avg_us - margin;
    result.ci_upper_us = result.avg_us + margin;
    result.ci_width_percent = (result.avg_us > 0.0)
        ? ((result.ci_upper_us - result.ci_lower_us) / result.avg_us * 100.0)
        : 0.0;

    // Calculate sub-timing statistics (also with outlier filtering)
    for (auto& [name, timings] : result.sub_timings_us) {
        if (timings.empty()) continue;

        size_t sub_outliers = 0;
        std::vector<double> sub_filtered = filter_outliers(timings, sub_outliers);
        std::vector<double> sub_sorted = sub_filtered;
        std::sort(sub_sorted.begin(), sub_sorted.end());

        BenchmarkResult::SubTimingStats stats;
        stats.min_us = sub_sorted.front();
        stats.max_us = sub_sorted.back();

        double sub_sum = 0.0;
        for (double t : sub_filtered) sub_sum += t;
        stats.avg_us = sub_sum / sub_filtered.size();

        double sub_sq_sum = 0.0;
        for (double t : sub_filtered) {
            double diff = t - stats.avg_us;
            sub_sq_sum += diff * diff;
        }
        stats.stddev_us = (sub_filtered.size() > 1)
            ? std::sqrt(sub_sq_sum / (sub_filtered.size() - 1))
            : 0.0;

        result.sub_timing_stats[name] = stats;
    }
}

/**
 * Extract git commit hash and date
 */
struct GitInfo {
    std::string hash;          // either commit hash or tree hash
    std::string hash_type;     // "commit" or "tree"
    std::string commit_date;   // YYYY-MM-DD format

    static std::string run_git_cmd(const char* git_cmd) {
        #ifdef _WIN32
        // Windows: wrap with bash -c using double quotes, cd to source directory
        std::string wrapped_cmd = std::string("bash -c \"cd ") + BENCHMARK_SOURCE_DIR + " && " + git_cmd + "\"";
        const char* final_cmd = wrapped_cmd.c_str();
        #else
        const char* final_cmd = git_cmd;
        #endif

        FILE* pipe = popen(final_cmd, "r");
        if (!pipe) {
            return "";
        }

        char buffer[128];
        std::string result;
        if (fgets(buffer, sizeof(buffer), pipe)) {
            result = buffer;
            if (!result.empty() && result.back() == '\n') {
                result.pop_back();
            }
        }
        pclose(pipe);
        return result;
    }

    static GitInfo get() {
        GitInfo info;

        // Always get the last commit date with time (even for tree hashes)
        info.commit_date = run_git_cmd("git log -1 --format=%cd --date=format:'%Y-%m-%d %H:%M:%S' 2>/dev/null");

        // Check if there are staged changes
        // git diff --cached --quiet returns empty string (exit 0) if NO changes, non-empty (exit 1) if changes exist
        std::string staged_check = run_git_cmd("git diff --cached --quiet 2>/dev/null || echo has_changes");
        bool has_staged_changes = !staged_check.empty();

        if (has_staged_changes) {
            // Use tree hash of staging area
            info.hash_type = "tree";
            info.hash = run_git_cmd("git write-tree 2>/dev/null");
        } else {
            // Use commit hash
            info.hash_type = "commit";
            info.hash = run_git_cmd("git rev-parse HEAD 2>/dev/null");
        }

        return info;
    }
};

/**
 * Get current timestamp in ISO 8601 format
 */
inline std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);

    // Use std::localtime - thread-safe on MSVC, otherwise use static storage
    std::tm* tm_ptr = std::localtime(&time_t_now);
    if (!tm_ptr) {
        return "unknown";
    }

    std::ostringstream oss;
    oss << std::put_time(tm_ptr, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

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

    void reset() {
        current_benchmark.clear();
        current_params.clear();
        current_metadata.clear();
        current_result_type = ResultType::TIME;
        timing_starts.clear();
        current_sub_timings.clear();
    }

    void set_benchmark(const std::string& name) {
        current_benchmark = name;
        current_params.clear();
        current_metadata.clear();
        current_result_type = ResultType::TIME;
        timing_starts.clear();
        current_sub_timings.clear();
    }

    void set_param(const std::string& name, const std::string& value) {
        current_params[name] = value;
    }

    template<typename T>
    void set_param(const std::string& name, const T& value) {
        std::ostringstream oss;
        oss << value;
        current_params[name] = oss.str();
    }

    void start_timing(const std::string& name) {
        timing_starts[name] = std::chrono::high_resolution_clock::now();
    }

    void stop_timing(const std::string& name) {
        auto end = std::chrono::high_resolution_clock::now();
        auto it = timing_starts.find(name);
        if (it != timing_starts.end()) {
            double duration_us = std::chrono::duration<double, std::micro>(end - it->second).count();
            current_sub_timings[name].push_back(duration_us);
        }
    }

    void reset_sub_timings() {
        timing_starts.clear();
        current_sub_timings.clear();
    }

    std::map<std::string, std::vector<double>> get_sub_timings() const {
        return current_sub_timings;
    }
};

/**
 * Global benchmark context (singleton)
 */
inline BenchmarkContext& get_context() {
    static BenchmarkContext ctx;
    return ctx;
}

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
    static void ensure_directory(const std::string& path) {
        std::filesystem::create_directories(path);
    }

    static void write_file(const std::string& filepath, const std::string& content) {
        // Write to temp file first
        std::string temp_file = filepath + ".tmp";
        std::ofstream ofs(temp_file);
        if (!ofs) {
            throw std::runtime_error("Failed to open file: " + temp_file);
        }
        ofs << content;
        ofs.close();

        // Atomic rename
        std::filesystem::rename(temp_file, filepath);
    }

    static void append_file(const std::string& filepath, const std::string& content) {
        std::ofstream ofs(filepath, std::ios::app);
        if (!ofs) {
            throw std::runtime_error("Failed to open file: " + filepath);
        }
        ofs << content;
        ofs.flush();
    }
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
    static std::vector<ColumnMetadata> get_column_metadata() {
        return {
            {"samples", ColumnType::INTEGER},
            {"outliers_removed", ColumnType::INTEGER},
            {"median_us", ColumnType::FLOATING_POINT},
            {"mad_us", ColumnType::FLOATING_POINT},
            {"min_us", ColumnType::FLOATING_POINT},
            {"max_us", ColumnType::FLOATING_POINT},
            {"avg_us", ColumnType::FLOATING_POINT},
            {"stddev_us", ColumnType::FLOATING_POINT},
            {"ci_lower_us", ColumnType::FLOATING_POINT},
            {"ci_upper_us", ColumnType::FLOATING_POINT},
            {"ci_width_percent", ColumnType::FLOATING_POINT}
        };
    }

    // Column names in exact order (backward compatibility)
    static std::vector<std::string> get_column_names() {
        std::vector<std::string> names;
        for (const auto& meta : get_column_metadata()) {
            names.push_back(meta.name);
        }
        return names;
    }

    // Extract column values from result in same order
    static std::vector<double> get_column_values(const BenchmarkResult& result) {
        return {
            static_cast<double>(result.samples),
            static_cast<double>(result.outliers_removed),
            result.median_us,
            result.mad_us,
            result.min_us,
            result.max_us,
            result.avg_us,
            result.stddev_us,
            result.ci_lower_us,
            result.ci_upper_us,
            result.ci_width_percent
        };
    }

    // Format a column value based on its type
    static std::string format_column_value(double value, ColumnType type) {
        std::ostringstream oss;
        if (type == ColumnType::INTEGER) {
            oss << static_cast<size_t>(value);
        } else {
            oss << std::fixed << std::setprecision(6) << value;
        }
        return oss.str();
    }

    // Write column values with proper formatting
    static void write_column_values(std::ostream& os, const BenchmarkResult& result) {
        auto metadata = get_column_metadata();
        auto values = get_column_values(result);
        for (size_t i = 0; i < values.size(); ++i) {
            os << "," << format_column_value(values[i], metadata[i].type);
        }
    }

    // Parse column values into result in same order
    static void parse_column_values(BenchmarkResult& result, const std::vector<std::string>& tokens) {
        if (tokens.size() < 11) return;
        result.samples = std::stoul(tokens[0]);
        result.outliers_removed = std::stoul(tokens[1]);
        result.median_us = std::stod(tokens[2]);
        result.mad_us = std::stod(tokens[3]);
        result.min_us = std::stod(tokens[4]);
        result.max_us = std::stod(tokens[5]);
        result.avg_us = std::stod(tokens[6]);
        result.stddev_us = std::stod(tokens[7]);
        result.ci_lower_us = std::stod(tokens[8]);
        result.ci_upper_us = std::stod(tokens[9]);
        result.ci_width_percent = std::stod(tokens[10]);
    }

    // Total number of base columns (including benchmark_name, params, metadata, result_type + data columns)
    static size_t base_column_count() {
        return 4 + get_column_names().size();  // 4 metadata cols + 11 data cols = 15
    }
};

/**
 * Reference data loader - lazily loads reference benchmark data from CSV
 */
class ReferenceDataLoader {
public:
    static ReferenceDataLoader& instance() {
        static ReferenceDataLoader loader;
        return loader;
    }

    // Get reference timing for a benchmark with specific parameters
    std::optional<double> get_reference_timing(const std::string& benchmark_name,
                                               const std::map<std::string, std::string>& params) {
        load_if_needed(benchmark_name);

        auto it = data_.find(benchmark_name);
        if (it == data_.end()) return std::nullopt;

        // Find matching parameter set
        for (const auto& entry : it->second) {
            if (entry.params == params) {
                return entry.avg_us;
            }
        }
        return std::nullopt;
    }

    // Check if reference data exists for a benchmark
    bool has_reference_data(const std::string& benchmark_name) {
        load_if_needed(benchmark_name);
        auto it = data_.find(benchmark_name);
        return it != data_.end() && !it->second.empty();
    }

    // Clear loaded data to force reload (after generating new reference data)
    void invalidate_cache(const std::string& benchmark_name) {
        data_.erase(benchmark_name);
        loaded_benchmarks_.erase(benchmark_name);
    }

private:
    struct ReferenceEntry {
        std::map<std::string, std::string> params;
        double avg_us;
    };

    std::map<std::string, std::vector<ReferenceEntry>> data_;
    std::set<std::string> loaded_benchmarks_;

    void load_if_needed(const std::string& benchmark_name) {
        if (loaded_benchmarks_.count(benchmark_name)) return;
        loaded_benchmarks_.insert(benchmark_name);

        std::string csv_path = "benchmark_results/reference/" + benchmark_name + ".csv";
        std::ifstream file(csv_path);
        if (!file) return;  // No reference data available

        std::string line;
        std::vector<std::string> headers;

        // Read header
        if (std::getline(file, line)) {
            std::istringstream ss(line);
            std::string col;
            while (std::getline(ss, col, ',')) {
                headers.push_back(col);
            }
        }

        // Read data rows
        while (std::getline(file, line)) {
            std::istringstream ss(line);
            std::string col;
            std::vector<std::string> row;
            while (std::getline(ss, col, ',')) {
                row.push_back(col);
            }

            if (row.size() != headers.size()) continue;

            ReferenceEntry entry;

            // Build set of data column names to exclude (not parameters)
            auto column_names = CSVSchema::get_column_names();
            std::set<std::string> data_columns(column_names.begin(), column_names.end());

            for (size_t i = 0; i < headers.size(); ++i) {
                if (headers[i] == "avg_us") {
                    entry.avg_us = std::stod(row[i]);
                } else if (data_columns.find(headers[i]) == data_columns.end()) {
                    // Not a data column, so it's a parameter column
                    entry.params[headers[i]] = row[i];
                }
            }

            data_[benchmark_name].push_back(entry);
        }
    }

    ReferenceDataLoader() = default;
};

/**
 * Benchmark registry - stores all registered benchmarks
 */
class BenchmarkRegistry {
public:
    using BenchmarkFunc = std::function<void()>;

    static BenchmarkRegistry& instance() {
        static BenchmarkRegistry registry;
        return registry;
    }

    void register_benchmark(const std::string& name, BenchmarkFunc func, const std::string& description = "", bool is_reference = false) {
        benchmarks_[name] = func;
        descriptions_[name] = description;
        reference_flags_[name] = is_reference;
    }

    void register_dependency(const std::string& benchmark_name, const std::string& reference_name) {
        dependencies_[benchmark_name].push_back(reference_name);
    }

    const std::map<std::string, BenchmarkFunc>& get_benchmarks() const {
        return benchmarks_;
    }

    std::string get_description(const std::string& name) const {
        auto it = descriptions_.find(name);
        return (it != descriptions_.end()) ? it->second : "";
    }

    bool is_reference(const std::string& name) const {
        auto it = reference_flags_.find(name);
        return (it != reference_flags_.end()) ? it->second : false;
    }

private:
    // Helper to create a gtest-style filter matcher
    // Returns a lambda that tests if a name matches the filter pattern
    static std::function<bool(const std::string&)> create_filter_matcher(const std::string& filter) {
        if (filter.empty()) {
            return [](const std::string&) { return true; };
        }

        // Parse gtest-style filter: positive_patterns-negative_patterns
        // Patterns are ':'-separated and support '*' (any string) and '?' (any char)
        std::vector<std::string> positive_patterns;
        std::vector<std::string> negative_patterns;

        std::string positive_str, negative_str;
        if (filter[0] == '-') {
            positive_str = "*";
            negative_str = filter.substr(1);
        } else {
            size_t dash_pos = filter.find('-');
            positive_str = (dash_pos == std::string::npos) ? filter : filter.substr(0, dash_pos);
            negative_str = (dash_pos == std::string::npos) ? "" : filter.substr(dash_pos + 1);
        }

        // Split positive patterns by ':'
        std::istringstream pos_stream(positive_str);
        std::string pattern;
        while (std::getline(pos_stream, pattern, ':')) {
            if (!pattern.empty()) positive_patterns.push_back(pattern);
        }

        // Split negative patterns by ':'
        if (!negative_str.empty()) {
            std::istringstream neg_stream(negative_str);
            while (std::getline(neg_stream, pattern, ':')) {
                if (!pattern.empty()) negative_patterns.push_back(pattern);
            }
        }

        // Convert wildcard pattern to regex
        auto wildcard_to_regex = [](const std::string& pat) -> std::string {
            std::string result;
            for (char c : pat) {
                if (c == '*') result += ".*";
                else if (c == '?') result += ".";
                else if (std::string(".^$()[]{}\\|+").find(c) != std::string::npos) {
                    result += '\\';
                    result += c;
                } else {
                    result += c;
                }
            }
            return result;
        };

        // Compile regex patterns
        std::vector<std::regex> positive_regex;
        std::vector<std::regex> negative_regex;

        for (const auto& p : positive_patterns) {
            try {
                positive_regex.emplace_back(wildcard_to_regex(p));
            } catch (const std::regex_error& e) {
                fprintf(stderr, "Error: Invalid pattern '%s': %s\n", p.c_str(), e.what());
                throw;
            }
        }

        for (const auto& p : negative_patterns) {
            try {
                negative_regex.emplace_back(wildcard_to_regex(p));
            } catch (const std::regex_error& e) {
                fprintf(stderr, "Error: Invalid pattern '%s': %s\n", p.c_str(), e.what());
                throw;
            }
        }

        // Return matcher lambda
        return [positive_regex, negative_regex](const std::string& name) -> bool {
            // Must match at least one positive pattern
            bool matches_positive = false;
            for (const auto& r : positive_regex) {
                if (std::regex_search(name, r)) {
                    matches_positive = true;
                    break;
                }
            }
            if (!matches_positive) return false;

            // Must not match any negative pattern
            for (const auto& r : negative_regex) {
                if (std::regex_search(name, r)) return false;
            }
            return true;
        };
    }

public:
    void list_benchmarks(const std::string& filter = "") const {
        std::function<bool(const std::string&)> matches_filter;
        try {
            matches_filter = create_filter_matcher(filter);
        } catch (const std::regex_error&) {
            return;
        }

        // Separate benchmarks into regular and reference
        std::vector<std::string> regular_benchmarks;
        std::vector<std::string> reference_benchmarks;

        for (const auto& [name, func] : benchmarks_) {
            if (matches_filter(name)) {
                if (is_reference(name)) {
                    reference_benchmarks.push_back(name);
                } else {
                    regular_benchmarks.push_back(name);
                }
            }
        }

        // Print regular benchmarks
        if (!regular_benchmarks.empty()) {
            printf("Available benchmarks:\n");
            for (const auto& name : regular_benchmarks) {
                printf("  %s", name.c_str());
                auto desc = get_description(name);
                if (!desc.empty()) {
                    printf(" - %s", desc.c_str());
                }
                printf("\n");
            }
        }

        // Print reference benchmarks
        if (!reference_benchmarks.empty()) {
            printf("\nReference benchmarks (use --reference to run):\n");
            for (const auto& name : reference_benchmarks) {
                printf("  %s", name.c_str());
                auto desc = get_description(name);
                if (!desc.empty()) {
                    printf(" - %s", desc.c_str());
                }
                printf("\n");
            }
        }

        size_t total = regular_benchmarks.size() + reference_benchmarks.size();
        printf("\nTotal: %zu benchmark(s)", total);
        if (!reference_benchmarks.empty()) {
            printf(" (%zu regular, %zu reference)", regular_benchmarks.size(), reference_benchmarks.size());
        }
        printf("\n");
    }

    // Run a single reference benchmark and save results
    void run_reference_benchmark(const std::string& name, const std::string& output_dir) {
        auto it = benchmarks_.find(name);
        if (it == benchmarks_.end()) {
            fprintf(stderr, "Error: Reference benchmark '%s' not found\n", name.c_str());
            return;
        }

        if (!is_reference(name)) {
            fprintf(stderr, "Error: Benchmark '%s' is not marked as reference\n", name.c_str());
            return;
        }

        GitInfo git_info = GitInfo::get();
        std::string timestamp = get_timestamp();
        std::string commit_dir = output_dir + "/reference";
        CSVWriter::ensure_directory(commit_dir);

        printf("[  DEPEND ] Running reference benchmark: %s\n", name.c_str());

        current_results_.clear();
        get_context().set_benchmark(name);
        it->second();  // Run the benchmark

        // Save results immediately
        if (!current_results_.empty()) {
            for (const auto& result : current_results_) {
                std::string csv_path = commit_dir + "/" + result.benchmark_name + ".csv";
                bool file_exists = std::filesystem::exists(csv_path);

                std::ostringstream csv;
                if (!file_exists) {
                    // Write header
                    csv << "samples,outliers_removed,median_us,mad_us,avg_us,stddev_us,min_us,max_us,ci_lower_us,ci_upper_us,ci_width_percent";
                    for (const auto& [key, val] : result.params) {
                        csv << "," << key;
                    }
                    csv << "\n";
                }

                // Write data row
                csv << result.samples << ","
                    << result.outliers_removed << ","
                    << result.median_us << ","
                    << result.mad_us << ","
                    << result.avg_us << ","
                    << result.stddev_us << ","
                    << result.min_us << ","
                    << result.max_us << ","
                    << result.ci_lower_us << ","
                    << result.ci_upper_us << ","
                    << result.ci_width_percent;
                for (const auto& [key, val] : result.params) {
                    csv << "," << val;
                }
                csv << "\n";

                CSVWriter::append_file(csv_path, csv.str());
            }

            // Invalidate cache so it gets reloaded
            ReferenceDataLoader::instance().invalidate_cache(name);

            printf("[  CACHED ] Reference data saved for: %s\n", name.c_str());
        }

        current_results_.clear();
    }

    void run_all(const std::string& output_dir, const std::string& filter = "", bool include_reference = false, bool only_reference = false) {
        // Create filter matcher
        std::function<bool(const std::string&)> matches_filter;
        try {
            matches_filter = create_filter_matcher(filter);
        } catch (const std::regex_error&) {
            return;
        }

        // Auto-enable reference mode if filter matches only reference benchmarks
        if (!filter.empty() && !include_reference && !only_reference) {
            bool has_regular_match = false;
            bool has_reference_match = false;

            for (const auto& [name, func] : benchmarks_) {
                if (matches_filter(name)) {
                    if (is_reference(name)) {
                        has_reference_match = true;
                    } else {
                        has_regular_match = true;
                    }
                }
            }

            // If filter matches only reference benchmarks, enable reference mode
            if (has_reference_match && !has_regular_match) {
                only_reference = true;
            }
        }

        GitInfo git_info = GitInfo::get();
        std::string timestamp = get_timestamp();

        // Validate git info before creating any folders (not needed for reference benchmarks)
        if (!only_reference && git_info.hash.empty()) {
            fprintf(stderr, "FATAL: Git hash is empty - refusing to create folders. Exiting.\n");
            exit(1);
        }

        // Create output directory
        std::string commit_dir;
        if (only_reference) {
            commit_dir = output_dir + "/reference";
        } else {
            commit_dir = output_dir + "/" + git_info.hash_type + "-" + git_info.hash;
        }
        CSVWriter::ensure_directory(commit_dir);

        std::vector<BenchmarkResult> all_results;

        // Count matching benchmarks
        size_t matching = 0;
        for (const auto& [name, func] : benchmarks_) {
            bool is_reference_bench = is_reference(name);

            // Apply reference filtering
            if (only_reference && !is_reference_bench) continue;
            if (!include_reference && !only_reference && is_reference_bench) continue;

            if (matches_filter(name)) {
                matching++;
            }
        }

        printf("Running %zu %sbenchmarks", matching, only_reference ? "reference " : "");
        if (!filter.empty()) {
            printf(" (filter: %s)", filter.c_str());
        }
        printf("...\n");
        if (only_reference) {
            printf("Mode: Reference benchmarks (output: reference/)\n");
        } else {
            printf("%s: %s (%s)\n",
                   git_info.hash_type == "commit" ? "Commit" : "Tree",
                   git_info.hash.c_str(),
                   git_info.commit_date.c_str());
            if (git_info.hash_type == "tree") {
                printf("⚠ WARNING: Benchmarking with staged changes - results are from uncommitted code\n");
            }
        }
        printf("\n");

        for (const auto& [name, func] : benchmarks_) {
            bool is_reference_bench = is_reference(name);

            // Apply reference filtering
            if (only_reference && !is_reference_bench) continue;
            if (!include_reference && !only_reference && is_reference_bench) continue;

            // Apply filter
            if (!matches_filter(name)) {
                continue;
            }

            // Check and run dependencies if needed
            auto dep_it = dependencies_.find(name);
            if (dep_it != dependencies_.end()) {
                for (const std::string& dep_name : dep_it->second) {
                    if (!ReferenceDataLoader::instance().has_reference_data(dep_name)) {
                        run_reference_benchmark(dep_name, output_dir);
                    }
                }
            }

            printf("[ RUN      ] %s\n", name.c_str());
            get_context().set_benchmark(name);

            // Run the benchmark function (will collect results internally)
            func();

            // Results are stored in current_results_ during benchmark execution
            for (const auto& result : current_results_) {
                all_results.push_back(result);

                // Determine units based on result type
                std::string units;
                switch (result.result_type) {
                    case ResultType::TIME:
                        units = "μs";
                        break;
                    case ResultType::RATIO:
                        units = "x";
                        break;
                    case ResultType::CUSTOM:
                        // Check metadata for custom unit
                        auto it = result.metadata.find("unit");
                        units = (it != result.metadata.end()) ? it->second : "";
                        break;
                }

                // Display: median (primary), [min, max], sample count, outliers removed
                if (result.outliers_removed > 0) {
                    printf("[       OK ] %s (%s) - %.2f %s [%.2f, %.2f] (n=%zu, %zu outliers, CI width=%.2f%%)\n",
                           result.benchmark_name.c_str(),
                           params_to_string(result.params).c_str(),
                           result.median_us, units.c_str(),
                           result.min_us, result.max_us, result.samples, result.outliers_removed, result.ci_width_percent);
                } else {
                    printf("[       OK ] %s (%s) - %.2f %s [%.2f, %.2f] (n=%zu, CI width=%.2f%%)\n",
                           result.benchmark_name.c_str(),
                           params_to_string(result.params).c_str(),
                           result.median_us, units.c_str(),
                           result.min_us, result.max_us, result.samples, result.ci_width_percent);
                }
            }
            current_results_.clear();

            printf("\n");
        }

        // Create per-benchmark results directory
        std::string results_subdir = commit_dir + "/per_benchmark";
        CSVWriter::ensure_directory(results_subdir);

        // Delete old results for benchmarks that are about to be written
        // (prevents mixing old and new results when benchmark parameters change)
        std::unordered_set<std::string> benchmarks_to_clear;
        for (const auto& result : all_results) {
            benchmarks_to_clear.insert(result.benchmark_name);
        }
        for (const auto& benchmark_name : benchmarks_to_clear) {
            // Remove all CSV files for this benchmark
            for (const auto& entry : std::filesystem::directory_iterator(results_subdir)) {
                if (entry.path().extension() == ".csv") {
                    std::string filename = entry.path().filename().string();
                    // Check if filename starts with benchmark_name
                    if (filename.find(benchmark_name) == 0) {
                        std::filesystem::remove(entry.path());
                    }
                }
            }
        }

        // Write per-benchmark CSV files (enables incremental updates)
        for (const auto& result : all_results) {
            write_benchmark_result_csv(results_subdir, result);
        }

        // Read all benchmark results from individual files and aggregate
        std::vector<BenchmarkResult> aggregated_results = read_all_benchmark_results(results_subdir);

        // Write aggregated files for backwards compatibility and convenience
        write_summary_csv(commit_dir, git_info, timestamp, aggregated_results);
        write_detailed_csv(commit_dir, aggregated_results);
        write_raw_timings_csv(commit_dir, all_results);  // Keep using all_results for raw timings (full data)
        write_samples_convergence_csv(commit_dir, all_results);  // Keep using all_results for convergence
        write_config_txt(commit_dir, git_info, timestamp);

        // Final summary (GTest-style)
        printf("[==========] %zu benchmarks completed\n", all_results.size());
        printf("[   SAVED  ] Results written to: %s/\n", commit_dir.c_str());
        printf("[   FILES  ] - summary.csv (aggregated metrics)\n");
        printf("[   FILES  ] - detailed.csv (per-benchmark stats)\n");
        printf("[   FILES  ] - raw_timings.csv (all raw samples)\n");
        printf("[   FILES  ] - samples_convergence.csv (CV convergence tracking)\n");
        printf("[   FILES  ] - benchmark_config.txt (configuration)\n");
    }

    void add_result(const BenchmarkResult& result) {
        current_results_.push_back(result);
        last_result_ = result;
    }

    const BenchmarkResult& get_last_result() const {
        return last_result_;
    }

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
                                       const std::string& metric) {
        std::ostringstream oss;
        oss << benchmark_name;
        for (const auto& [key, value] : params) {
            oss << "_" << value;
        }
        oss << "_" << metric;
        return oss.str();
    }

    static std::vector<BenchmarkResult> read_all_benchmark_results(const std::string& results_dir) {
        std::vector<BenchmarkResult> all_results;

        // Iterate through all CSV files in results directory
        for (const auto& entry : std::filesystem::directory_iterator(results_dir)) {
            if (entry.path().extension() == ".csv") {
                std::ifstream ifs(entry.path());
                if (!ifs) continue;

                std::string header_line;
                std::getline(ifs, header_line);  // Skip header

                std::string data_line;
                if (std::getline(ifs, data_line)) {
                    BenchmarkResult result;

                    // Parse CSV line (simplified - assumes proper format)
                    std::istringstream iss(data_line);
                    std::string token;

                    // benchmark_name
                    std::getline(iss, result.benchmark_name, ',');

                    // params (quoted string)
                    std::getline(iss, token, '"');  // Skip opening quote
                    std::getline(iss, token, '"');  // Get params
                    // Parse params into map
                    std::istringstream params_stream(token);
                    std::string param_pair;
                    while (std::getline(params_stream, param_pair, ',')) {
                        size_t eq_pos = param_pair.find('=');
                        if (eq_pos != std::string::npos) {
                            std::string key = param_pair.substr(0, eq_pos);
                            std::string value = param_pair.substr(eq_pos + 1);
                            // Trim whitespace
                            key.erase(0, key.find_first_not_of(" \t"));
                            key.erase(key.find_last_not_of(" \t") + 1);
                            value.erase(0, value.find_first_not_of(" \t"));
                            value.erase(value.find_last_not_of(" \t") + 1);
                            result.params[key] = value;
                        }
                    }

                    std::getline(iss, token, ',');  // Skip comma after closing quote

                    // metadata (quoted string)
                    std::getline(iss, token, '"');  // Skip opening quote
                    std::getline(iss, token, '"');  // Get metadata
                    // Parse metadata into map
                    std::istringstream metadata_stream(token);
                    std::string meta_pair;
                    while (std::getline(metadata_stream, meta_pair, ',')) {
                        size_t eq_pos = meta_pair.find('=');
                        if (eq_pos != std::string::npos) {
                            std::string key = meta_pair.substr(0, eq_pos);
                            std::string value = meta_pair.substr(eq_pos + 1);
                            // Trim whitespace
                            key.erase(0, key.find_first_not_of(" \t"));
                            key.erase(key.find_last_not_of(" \t") + 1);
                            value.erase(0, value.find_first_not_of(" \t"));
                            value.erase(value.find_last_not_of(" \t") + 1);
                            result.metadata[key] = value;
                        }
                    }

                    std::getline(iss, token, ',');  // Skip comma after closing quote

                    // result_type
                    std::getline(iss, token, ',');
                    if (token == "TIME") result.result_type = ResultType::TIME;
                    else if (token == "RATIO") result.result_type = ResultType::RATIO;
                    else if (token == "CUSTOM") result.result_type = ResultType::CUSTOM;

                    // Parse data columns using centralized schema
                    std::vector<std::string> data_tokens;
                    auto column_names = CSVSchema::get_column_names();
                    for (size_t i = 0; i < column_names.size(); ++i) {
                        std::getline(iss, token, ',');
                        data_tokens.push_back(token);
                    }
                    CSVSchema::parse_column_values(result, data_tokens);

                    // Sub-timings (if any) - parse remaining columns in groups of 4
                    std::vector<std::string> remaining_tokens;
                    while (std::getline(iss, token, ',')) {
                        remaining_tokens.push_back(token);
                    }

                    // Parse sub-timing stats from remaining columns
                    // Format: name_avg_us, name_stddev_us, name_min_us, name_max_us
                    // We need to extract names from header
                    std::istringstream header_stream(header_line);
                    std::vector<std::string> headers;
                    while (std::getline(header_stream, token, ',')) {
                        headers.push_back(token);
                    }

                    size_t base_columns = CSVSchema::base_column_count();
                    for (size_t i = base_columns; i + 3 < headers.size(); i += 4) {
                        std::string header = headers[i];
                        if (header.find("_avg_us") != std::string::npos) {
                            std::string name = header.substr(0, header.find("_avg_us"));
                            BenchmarkResult::SubTimingStats stats;
                            size_t token_idx = i - base_columns;
                            if (token_idx + 3 < remaining_tokens.size()) {
                                stats.avg_us = std::stod(remaining_tokens[token_idx]);
                                stats.stddev_us = std::stod(remaining_tokens[token_idx + 1]);
                                stats.min_us = std::stod(remaining_tokens[token_idx + 2]);
                                stats.max_us = std::stod(remaining_tokens[token_idx + 3]);
                                result.sub_timing_stats[name] = stats;
                            }
                        }
                    }

                    all_results.push_back(result);
                }
            }
        }

        return all_results;
    }

    static void write_benchmark_result_csv(const std::string& results_dir, const BenchmarkResult& result) {
        // Create filename: benchmark_name__params.csv
        std::ostringstream filename;
        filename << result.benchmark_name;
        if (!result.params.empty()) {
            filename << "__" << params_to_string(result.params);
        }

        // Sanitize filename: replace problematic characters
        std::string name = sanitize_for_filename(filename.str());
        std::replace(name.begin(), name.end(), '=', '_');
        std::replace(name.begin(), name.end(), ',', '_');
        std::replace(name.begin(), name.end(), ' ', '_');

        name += ".csv";

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6);

        // Convert result type to string
        std::string result_type_str;
        switch (result.result_type) {
            case ResultType::TIME: result_type_str = "TIME"; break;
            case ResultType::RATIO: result_type_str = "RATIO"; break;
            case ResultType::CUSTOM: result_type_str = "CUSTOM"; break;
        }

        // Header - use centralized schema
        oss << "benchmark_name,params,metadata,result_type";
        for (const auto& col_name : CSVSchema::get_column_names()) {
            oss << "," << col_name;
        }
        for (const auto& [sub_name, _] : result.sub_timing_stats) {
            oss << "," << sub_name << "_avg_us," << sub_name << "_stddev_us," << sub_name << "_min_us," << sub_name << "_max_us";
        }
        oss << "\n";

        // Data row - use centralized schema
        oss << result.benchmark_name << ",\""
            << params_to_string(result.params) << "\",\""
            << params_to_string(result.metadata) << "\","
            << result_type_str;
        CSVSchema::write_column_values(oss, result);

        for (const auto& [sub_name, stats] : result.sub_timing_stats) {
            oss << "," << stats.avg_us
                << "," << stats.stddev_us
                << "," << stats.min_us
                << "," << stats.max_us;
        }
        oss << "\n";

        CSVWriter::write_file(results_dir + "/" + name, oss.str());
    }

    static void write_summary_csv(const std::string& dir, const GitInfo& git_info,
                                  const std::string& timestamp,
                                  const std::vector<BenchmarkResult>& results) {
        std::ostringstream header, row;
        row << std::fixed << std::setprecision(6);

        header << "hash_type,hash,commit_date,timestamp";
        row << git_info.hash_type << "," << git_info.hash << "," << git_info.commit_date << "," << timestamp;

        for (const auto& result : results) {
            for (const char* metric : {"avg_us", "stddev_us", "samples", "min_us", "max_us"}) {
                std::string col_name = make_column_name(result.benchmark_name, result.params, metric);
                header << "," << col_name;

                if (std::string(metric) == "avg_us") row << "," << result.avg_us;
                else if (std::string(metric) == "stddev_us") row << "," << result.stddev_us;
                else if (std::string(metric) == "samples") row << "," << result.samples;
                else if (std::string(metric) == "min_us") row << "," << result.min_us;
                else if (std::string(metric) == "max_us") row << "," << result.max_us;
            }
        }

        std::string content = header.str() + "\n" + row.str() + "\n";
        CSVWriter::write_file(dir + "/summary.csv", content);
    }

    static void write_detailed_csv(const std::string& dir, const std::vector<BenchmarkResult>& results) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6);

        // Header - use centralized schema
        oss << "benchmark_name,params,metadata,result_type";
        for (const auto& col_name : CSVSchema::get_column_names()) {
            oss << "," << col_name;
        }

        // Collect all unique sub-timing names across all results
        std::set<std::string> all_sub_timing_names;
        for (const auto& result : results) {
            for (const auto& [name, _] : result.sub_timing_stats) {
                all_sub_timing_names.insert(name);
            }
        }

        // Add sub-timing columns to header
        for (const auto& name : all_sub_timing_names) {
            oss << "," << name << "_avg_us," << name << "_stddev_us," << name << "_min_us," << name << "_max_us";
        }
        oss << "\n";

        for (const auto& result : results) {
            // Convert result type to string
            std::string result_type_str;
            switch (result.result_type) {
                case ResultType::TIME: result_type_str = "TIME"; break;
                case ResultType::RATIO: result_type_str = "RATIO"; break;
                case ResultType::CUSTOM: result_type_str = "CUSTOM"; break;
            }

            // Data row - use centralized schema
            oss << result.benchmark_name << ",\""
                << params_to_string(result.params) << "\",\""
                << params_to_string(result.metadata) << "\","
                << result_type_str;
            CSVSchema::write_column_values(oss, result);

            // Add sub-timing data
            for (const auto& name : all_sub_timing_names) {
                auto it = result.sub_timing_stats.find(name);
                if (it != result.sub_timing_stats.end()) {
                    oss << "," << it->second.avg_us
                        << "," << it->second.stddev_us
                        << "," << it->second.min_us
                        << "," << it->second.max_us;
                } else {
                    oss << ",,,,";  // Empty columns if this result doesn't have this sub-timing
                }
            }
            oss << "\n";
        }

        CSVWriter::write_file(dir + "/detailed.csv", oss.str());
    }

    static void write_raw_timings_csv(const std::string& dir, const std::vector<BenchmarkResult>& results) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6);
        oss << "benchmark_name,params,sample_num,timing_us,sub_timing_name,sub_timing_us\n";

        for (const auto& result : results) {
            for (size_t i = 0; i < result.raw_timings_us.size(); ++i) {
                // Main timing
                oss << result.benchmark_name << ","
                    << params_to_string(result.params) << ","
                    << i << ","
                    << result.raw_timings_us[i] << ",,\n";

                // Sub-timings for this sample
                for (const auto& [name, timings] : result.sub_timings_us) {
                    if (i < timings.size()) {
                        oss << result.benchmark_name << ","
                            << params_to_string(result.params) << ","
                            << i << ",,"
                            << name << ","
                            << timings[i] << "\n";
                    }
                }
            }
        }

        CSVWriter::write_file(dir + "/raw_timings.csv", oss.str());
    }

    static void write_samples_convergence_csv(const std::string& dir, const std::vector<BenchmarkResult>& results) {
        std::ostringstream oss;
        oss << "benchmark_name,params,sample_num,cumulative_avg_us,cumulative_stddev_us,ci_width_percent\n";

        for (const auto& result : results) {
            for (const auto& point : result.convergence_history) {
                oss << result.benchmark_name << ","
                    << params_to_string(result.params) << ","
                    << point.sample_num << ","
                    << std::fixed << std::setprecision(6)
                    << point.cumulative_avg_us << ","
                    << point.cumulative_stddev_us << ","
                    << point.ci_width_percent
                    << std::defaultfloat << "\n";
            }
        }

        CSVWriter::write_file(dir + "/samples_convergence.csv", oss.str());
    }

    static void write_config_txt(const std::string& dir, const GitInfo& git_info, const std::string& timestamp) {
        std::ostringstream oss;
        oss << "BENCHMARK_MIN_SAMPLES=" << BENCHMARK_MIN_SAMPLES << "\n";
        oss << "BENCHMARK_MAX_SAMPLES=" << BENCHMARK_MAX_SAMPLES << "\n";
        oss << "BENCHMARK_CI_THRESHOLD=" << BENCHMARK_CI_THRESHOLD << "\n";
        oss << "HASH_TYPE=" << git_info.hash_type << "\n";
        oss << "HASH=" << git_info.hash << "\n";
        oss << "COMMIT_DATE=" << git_info.commit_date << "\n";
        oss << "TIMESTAMP=" << timestamp << "\n";
        oss << "COMPILER=" << BENCHMARK_COMPILER_INFO << "\n";

        CSVWriter::write_file(dir + "/benchmark_config.txt", oss.str());
    }
};

/**
 * Helper for benchmark registration
 */
struct BenchmarkRegistration {
    BenchmarkRegistration(const std::string& name, BenchmarkRegistry::BenchmarkFunc func, const std::string& description = "", bool is_reference = false) {
        BenchmarkRegistry::instance().register_benchmark(name, func, description, is_reference);
    }
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
inline void BENCHMARK_BEGIN() {
    auto& ctx = benchmark::get_context();
    ctx.manual_timing_start = std::chrono::high_resolution_clock::now();
}

/**
 * Manual timing control - end benchmark timing block and record sample
 */
inline void BENCHMARK_END() {
    auto& ctx = benchmark::get_context();
    auto end = std::chrono::high_resolution_clock::now();
    double timing_us = std::chrono::duration<double, std::micro>(end - ctx.manual_timing_start).count();
    ctx.manual_timings.push_back(timing_us);
}

/**
 * Submit manually collected timings (from BENCHMARK_BEGIN/END)
 */
inline void BENCHMARK_SUBMIT_MANUAL() {
    auto& ctx = benchmark::get_context();
    BenchmarkResult result;
    result.benchmark_name = ctx.current_benchmark;
    result.params = ctx.current_params;
    result.raw_timings_us = std::move(ctx.manual_timings);
    ctx.manual_timings.clear();
    calculate_statistics(result);
    benchmark::BenchmarkRegistry::instance().add_result(result);
}

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
inline void BENCHMARK_SUBMIT(const std::vector<double>& timings_us, ResultType type = ResultType::TIME) {
    auto& ctx = benchmark::get_context();

    if (ctx.inside_benchmark_code) {
        // Inside BENCHMARK_CODE: append to manual_timings for adaptive sampling
        for (double t : timings_us) {
            ctx.manual_timings.push_back(t);
        }
        ctx.current_result_type = type;
    } else {
        // Outside BENCHMARK_CODE: create complete result immediately
        BenchmarkResult result;
        result.benchmark_name = ctx.current_benchmark;
        result.params = ctx.current_params;
        result.metadata = ctx.current_metadata;
        result.result_type = type;
        result.raw_timings_us = timings_us;
        calculate_statistics(result);
        benchmark::BenchmarkRegistry::instance().add_result(result);

        // Clear params and metadata after submission
        ctx.current_params.clear();
        ctx.current_metadata.clear();
        ctx.current_result_type = ResultType::TIME;
    }
}

// Overload for single value submission (convenience)
inline void BENCHMARK_SUBMIT(double timing_us, ResultType type = ResultType::TIME) {
    BENCHMARK_SUBMIT(std::vector<double>{timing_us}, type);
}

/**
 * Helper to submit results to a different benchmark name
 * Useful for submitting related metrics with different units (e.g., timings vs speedup)
 */
inline void BENCHMARK_SUBMIT_AS(const std::string& benchmark_name, const std::vector<double>& timings_us, ResultType type = ResultType::TIME) {
    auto& ctx = benchmark::get_context();
    std::string original = ctx.current_benchmark;
    ctx.current_benchmark = benchmark_name;
    BENCHMARK_SUBMIT(timings_us, type);
    ctx.current_benchmark = original;
}

// Overload for single value submission
inline void BENCHMARK_SUBMIT_AS(const std::string& benchmark_name, double timing_us, ResultType type = ResultType::TIME) {
    BENCHMARK_SUBMIT_AS(benchmark_name, std::vector<double>{timing_us}, type);
}

/**
 * Helper to set benchmark metadata (for plotting customization)
 * Common keys: "y_label" (e.g., "Speedup (x)", "Time (μs)"), "y_scale" (e.g., "linear", "log")
 */
inline void BENCHMARK_META(const std::string& key, const std::string& value) {
    auto& ctx = benchmark::get_context();
    ctx.current_metadata[key] = value;
}

/**
 * Sub-timing macros for measuring portions of a benchmark
 */
#define BENCHMARK_TIMING_START(name) \
    benchmark::get_context().start_timing(name)

#define BENCHMARK_TIMING_STOP(name) \
    benchmark::get_context().stop_timing(name)

} // namespace benchmark

#endif // BENCHMARK_FRAMEWORK_HPP
