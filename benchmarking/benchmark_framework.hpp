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
constexpr size_t BENCHMARK_MIN_SAMPLES = 5;
constexpr size_t BENCHMARK_MAX_SAMPLES = 100;
constexpr double BENCHMARK_VARIANCE_THRESHOLD = 0.05;  // 5% coefficient of variation
constexpr size_t BENCHMARK_WARMUP_RUNS = 2;

/**
 * Statistical results for a benchmark run
 */
struct BenchmarkResult {
    std::string benchmark_name;
    std::map<std::string, std::string> params;  // param_name -> param_value
    size_t samples = 0;
    double min_us = 0.0;
    double max_us = 0.0;
    double avg_us = 0.0;
    double stddev_us = 0.0;
    double cv_percent = 0.0;  // coefficient of variation (stddev/mean * 100)
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
        double cv_percent;
    };
    std::vector<ConvergencePoint> convergence_history;
};

/**
 * Calculate statistics from timing samples
 */
inline void calculate_statistics(BenchmarkResult& result) {
    if (result.raw_timings_us.empty()) return;

    result.samples = result.raw_timings_us.size();

    // Min/Max
    result.min_us = *std::min_element(result.raw_timings_us.begin(), result.raw_timings_us.end());
    result.max_us = *std::max_element(result.raw_timings_us.begin(), result.raw_timings_us.end());

    // Average
    double sum = 0.0;
    for (double t : result.raw_timings_us) sum += t;
    result.avg_us = sum / result.samples;

    // Standard deviation
    double sq_sum = 0.0;
    for (double t : result.raw_timings_us) {
        double diff = t - result.avg_us;
        sq_sum += diff * diff;
    }
    result.stddev_us = std::sqrt(sq_sum / result.samples);

    // Coefficient of variation
    result.cv_percent = (result.avg_us > 0.0) ? (result.stddev_us / result.avg_us * 100.0) : 0.0;

    // Calculate sub-timing statistics
    for (auto& [name, timings] : result.sub_timings_us) {
        if (timings.empty()) continue;

        BenchmarkResult::SubTimingStats stats;
        stats.min_us = *std::min_element(timings.begin(), timings.end());
        stats.max_us = *std::max_element(timings.begin(), timings.end());

        double sub_sum = 0.0;
        for (double t : timings) sub_sum += t;
        stats.avg_us = sub_sum / timings.size();

        double sub_sq_sum = 0.0;
        for (double t : timings) {
            double diff = t - stats.avg_us;
            sub_sq_sum += diff * diff;
        }
        stats.stddev_us = std::sqrt(sub_sq_sum / timings.size());

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

        // Always get the last commit date (even for tree hashes)
        info.commit_date = run_git_cmd("git log -1 --format=%cd --date=short 2>/dev/null");

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

    // Sub-timing tracking
    std::map<std::string, std::chrono::high_resolution_clock::time_point> timing_starts;
    std::map<std::string, std::vector<double>> current_sub_timings;

    void reset() {
        current_benchmark.clear();
        current_params.clear();
        timing_starts.clear();
        current_sub_timings.clear();
    }

    void set_benchmark(const std::string& name) {
        current_benchmark = name;
        current_params.clear();
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
    template<typename Func>
    static BenchmarkResult run(const std::string& benchmark_name,
                              const std::map<std::string, std::string>& params,
                              Func&& benchmark_code,
                              size_t calibrated_samples = 0) {
        BenchmarkResult result;
        result.benchmark_name = benchmark_name;
        result.params = params;

        // If calibrated sample count is provided, use it
        if (calibrated_samples > 0) {
            std::string params_str = params_to_string(params);
            if (!params_str.empty()) {
                printf("[   TIMING ] Running %zu calibrated samples for (%s)...\n", calibrated_samples, params_str.c_str());
            } else {
                printf("[   TIMING ] Running %zu calibrated samples...\n", calibrated_samples);
            }
            for (size_t i = 0; i < calibrated_samples; ++i) {
                get_context().reset_sub_timings();
                double timing = time_single_run(benchmark_code);
                result.raw_timings_us.push_back(timing);

                // Collect sub-timings from this sample
                auto sub_timings = get_context().get_sub_timings();
                for (const auto& [name, timings] : sub_timings) {
                    for (double t : timings) {
                        result.sub_timings_us[name].push_back(t);
                    }
                }

                calculate_statistics(result);

                // Record convergence point
                BenchmarkResult::ConvergencePoint point;
                point.sample_num = result.raw_timings_us.size();
                point.cumulative_avg_us = result.avg_us;
                point.cumulative_stddev_us = result.stddev_us;
                point.cv_percent = result.cv_percent;
                result.convergence_history.push_back(point);

                printf("[   TIMING ]   Sample %zu/%zu: %.2f μs (avg: %.2f μs, CV: %.2f%%)\n",
                       i+1, calibrated_samples, timing, result.avg_us, result.cv_percent);
            }
            calculate_statistics(result);
            return result;
        }

        // Otherwise, use adaptive sampling
        printf("[   TIMING ] Adaptive sampling (target CV < %.1f%%)...\n", BENCHMARK_VARIANCE_THRESHOLD * 100.0);
        for (size_t i = 0; i < BENCHMARK_MIN_SAMPLES; ++i) {
            get_context().reset_sub_timings();
            double timing = time_single_run(benchmark_code);
            result.raw_timings_us.push_back(timing);

            // Collect sub-timings from this sample
            auto sub_timings = get_context().get_sub_timings();
            for (const auto& [name, timings] : sub_timings) {
                for (double t : timings) {
                    result.sub_timings_us[name].push_back(t);
                }
            }

            calculate_statistics(result);

            // Record convergence point
            BenchmarkResult::ConvergencePoint point;
            point.sample_num = result.raw_timings_us.size();
            point.cumulative_avg_us = result.avg_us;
            point.cumulative_stddev_us = result.stddev_us;
            point.cv_percent = result.cv_percent;
            result.convergence_history.push_back(point);
        }

        // Keep adding samples until variance converges or max samples reached
        while (result.raw_timings_us.size() < BENCHMARK_MAX_SAMPLES) {
            calculate_statistics(result);

            if (result.cv_percent < BENCHMARK_VARIANCE_THRESHOLD * 100.0) {
                printf("[   TIMING ]   Converged at %zu samples (CV: %.2f%%)\n",
                       result.raw_timings_us.size(), result.cv_percent);
                break;  // Variance converged
            }

            // Add 5 more samples
            printf("[   TIMING ]   CV %.2f%% > %.1f%%, adding more samples...\n",
                   result.cv_percent, BENCHMARK_VARIANCE_THRESHOLD * 100.0);
            for (size_t i = 0; i < 5 && result.raw_timings_us.size() < BENCHMARK_MAX_SAMPLES; ++i) {
                get_context().reset_sub_timings();
                double timing = time_single_run(benchmark_code);
                result.raw_timings_us.push_back(timing);

                // Collect sub-timings from this sample
                auto sub_timings = get_context().get_sub_timings();
                for (const auto& [name, timings] : sub_timings) {
                    for (double t : timings) {
                        result.sub_timings_us[name].push_back(t);
                    }
                }

                calculate_statistics(result);

                // Record convergence point
                BenchmarkResult::ConvergencePoint point;
                point.sample_num = result.raw_timings_us.size();
                point.cumulative_avg_us = result.avg_us;
                point.cumulative_stddev_us = result.stddev_us;
                point.cv_percent = result.cv_percent;
                result.convergence_history.push_back(point);
            }
        }

        calculate_statistics(result);
        return result;
    }

private:
    template<typename Func>
    static double time_single_run(Func&& benchmark_code) {
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
 * Load calibration data from file
 */
class CalibrationLoader {
public:
    static std::map<std::string, size_t> load(const std::string& filepath) {
        std::map<std::string, size_t> calibration;

        std::ifstream ifs(filepath);
        if (!ifs) {
            return calibration;  // File doesn't exist, return empty map
        }

        std::string line;
        // Skip header
        if (std::getline(ifs, line)) {
            // Read data lines
            while (std::getline(ifs, line)) {
                auto parts = split(line, ',');
                if (parts.size() >= 2) {
                    std::string key = parts[0];  // benchmark_name with params
                    size_t samples = std::stoul(parts[parts.size() - 2]);  // optimal_samples column
                    calibration[key] = samples;
                }
            }
        }

        return calibration;
    }

private:
    static std::vector<std::string> split(const std::string& s, char delimiter) {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(s);
        while (std::getline(tokenStream, token, delimiter)) {
            tokens.push_back(token);
        }
        return tokens;
    }
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

    void register_benchmark(const std::string& name, BenchmarkFunc func, const std::string& description = "") {
        benchmarks_[name] = func;
        descriptions_[name] = description;
    }

    const std::map<std::string, BenchmarkFunc>& get_benchmarks() const {
        return benchmarks_;
    }

    std::string get_description(const std::string& name) const {
        auto it = descriptions_.find(name);
        return (it != descriptions_.end()) ? it->second : "";
    }

    void list_benchmarks(const std::string& filter = "") const {
        printf("Available benchmarks:\n");
        std::regex filter_regex;
        bool use_regex = !filter.empty();
        if (use_regex) {
            try {
                // Implicit start-of-line anchor - user must match from beginning
                filter_regex = std::regex("^" + filter);
            } catch (const std::regex_error& e) {
                printf("Error: Invalid regex pattern '%s': %s\n", filter.c_str(), e.what());
                return;
            }
        }

        for (const auto& [name, func] : benchmarks_) {
            bool matches = !use_regex || std::regex_search(name, filter_regex);
            if (matches) {
                printf("  %s", name.c_str());
                auto desc = get_description(name);
                if (!desc.empty()) {
                    printf(" - %s", desc.c_str());
                }
                printf("\n");
            }
        }
        printf("\nTotal: %zu benchmark(s)\n", filter.empty() ? benchmarks_.size() :
               std::count_if(benchmarks_.begin(), benchmarks_.end(),
                   [&](const auto& p) { return std::regex_search(p.first, filter_regex); }));
    }

    void run_all(const std::string& output_dir, const std::map<std::string, size_t>& calibration = {}, const std::string& filter = "") {
        GitInfo git_info = GitInfo::get();
        std::string timestamp = get_timestamp();

        // Validate git info before creating any folders
        if (git_info.hash.empty()) {
            fprintf(stderr, "FATAL: Git hash is empty - refusing to create folders. Exiting.\n");
            exit(1);
        }

        // Create output directory: benchmark_results/<hash_type>-<hash>/
        std::string commit_dir = output_dir + "/" + git_info.hash_type + "-" + git_info.hash;
        CSVWriter::ensure_directory(commit_dir);

        std::vector<BenchmarkResult> all_results;

        // Compile regex filter with implicit start-of-line anchor
        std::regex filter_regex;
        bool use_regex = !filter.empty();
        if (use_regex) {
            try {
                filter_regex = std::regex("^" + filter);
            } catch (const std::regex_error& e) {
                printf("Error: Invalid regex pattern '%s': %s\n", filter.c_str(), e.what());
                return;
            }
        }

        // Count matching benchmarks
        size_t matching = 0;
        for (const auto& [name, func] : benchmarks_) {
            if (!use_regex || std::regex_search(name, filter_regex)) {
                matching++;
            }
        }

        printf("Running %zu benchmarks", matching);
        if (!filter.empty()) {
            printf(" (filter: ^%s)", filter.c_str());
        }
        printf("...\n");
        printf("%s: %s (%s)\n",
               git_info.hash_type == "commit" ? "Commit" : "Tree",
               git_info.hash.c_str(),
               git_info.commit_date.c_str());
        if (git_info.hash_type == "tree") {
            printf("⚠ WARNING: Benchmarking with staged changes - results are from uncommitted code\n");
        }
        printf("\n");

        for (const auto& [name, func] : benchmarks_) {
            // Apply regex filter
            if (use_regex && !std::regex_search(name, filter_regex)) {
                continue;
            }
            printf("[ RUN      ] %s\n", name.c_str());
            get_context().set_benchmark(name);

            // Run the benchmark function (will collect results internally)
            func();

            // Results are stored in current_results_ during benchmark execution
            for (const auto& result : current_results_) {
                all_results.push_back(result);
                printf("[       OK ] %s (%s) - %.2f ± %.2f μs (n=%zu, CV=%.2f%%)\n",
                       name.c_str(),
                       params_to_string(result.params).c_str(),
                       result.avg_us, result.stddev_us, result.samples, result.cv_percent);
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
    }

private:
    std::map<std::string, BenchmarkFunc> benchmarks_;
    std::map<std::string, std::string> descriptions_;
    std::vector<BenchmarkResult> current_results_;

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

                    // samples, min_us, max_us, avg_us, stddev_us, cv_percent
                    std::getline(iss, token, ','); result.samples = std::stoul(token);
                    std::getline(iss, token, ','); result.min_us = std::stod(token);
                    std::getline(iss, token, ','); result.max_us = std::stod(token);
                    std::getline(iss, token, ','); result.avg_us = std::stod(token);
                    std::getline(iss, token, ','); result.stddev_us = std::stod(token);
                    std::getline(iss, token, ','); result.cv_percent = std::stod(token);

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

                    size_t base_columns = 8;  // benchmark_name, params, samples, min, max, avg, stddev, cv
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
        std::string name = filename.str();
        std::replace(name.begin(), name.end(), '=', '_');
        std::replace(name.begin(), name.end(), ',', '_');
        std::replace(name.begin(), name.end(), ' ', '_');

        name += ".csv";

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6);

        // Header
        oss << "benchmark_name,params,samples,min_us,max_us,avg_us,stddev_us,cv_percent";
        for (const auto& [sub_name, _] : result.sub_timing_stats) {
            oss << "," << sub_name << "_avg_us," << sub_name << "_stddev_us," << sub_name << "_min_us," << sub_name << "_max_us";
        }
        oss << "\n";

        // Data row
        oss << result.benchmark_name << ",\""
            << params_to_string(result.params) << "\","
            << result.samples << ","
            << result.min_us << ","
            << result.max_us << ","
            << result.avg_us << ","
            << result.stddev_us << ","
            << result.cv_percent;

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
        oss << "benchmark_name,params,samples,min_us,max_us,avg_us,stddev_us,cv_percent";

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
            oss << result.benchmark_name << ",\""
                << params_to_string(result.params) << "\","
                << result.samples << ","
                << result.min_us << ","
                << result.max_us << ","
                << result.avg_us << ","
                << result.stddev_us << ","
                << result.cv_percent;

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
        oss << std::fixed << std::setprecision(6);
        oss << "benchmark_name,params,sample_num,cumulative_avg_us,cumulative_stddev_us,cv_percent\n";

        for (const auto& result : results) {
            for (const auto& point : result.convergence_history) {
                oss << result.benchmark_name << ","
                    << params_to_string(result.params) << ","
                    << point.sample_num << ","
                    << point.cumulative_avg_us << ","
                    << point.cumulative_stddev_us << ","
                    << point.cv_percent << "\n";
            }
        }

        CSVWriter::write_file(dir + "/samples_convergence.csv", oss.str());
    }

    static void write_config_txt(const std::string& dir, const GitInfo& git_info, const std::string& timestamp) {
        std::ostringstream oss;
        oss << "BENCHMARK_MIN_SAMPLES=" << BENCHMARK_MIN_SAMPLES << "\n";
        oss << "BENCHMARK_MAX_SAMPLES=" << BENCHMARK_MAX_SAMPLES << "\n";
        oss << "BENCHMARK_VARIANCE_THRESHOLD=" << BENCHMARK_VARIANCE_THRESHOLD << "\n";
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
    BenchmarkRegistration(const std::string& name, BenchmarkRegistry::BenchmarkFunc func, const std::string& description = "") {
        BenchmarkRegistry::instance().register_benchmark(name, func, description);
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

/**
 * Helper to set parameter in current context
 */
template<typename T>
inline void BENCHMARK_PARAM(const std::string& name, const T& value) {
    benchmark::get_context().set_param(name, value);
}

/**
 * Helper to run benchmark code with timing
 */
template<typename Func>
inline void BENCHMARK_CODE(Func&& code, size_t calibrated_samples = 0) {
    auto& ctx = benchmark::get_context();
    auto result = benchmark::AdaptiveSampler::run(
        ctx.current_benchmark,
        ctx.current_params,
        std::forward<Func>(code),
        calibrated_samples
    );
    benchmark::BenchmarkRegistry::instance().add_result(result);
}

/**
 * Helper to submit pre-collected timings (for external benchmarks like WolframScript)
 */
inline void BENCHMARK_SUBMIT(const std::vector<double>& timings_us) {
    auto& ctx = benchmark::get_context();
    BenchmarkResult result;
    result.benchmark_name = ctx.current_benchmark;
    result.params = ctx.current_params;
    result.raw_timings_us = timings_us;
    calculate_statistics(result);
    benchmark::BenchmarkRegistry::instance().add_result(result);
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
