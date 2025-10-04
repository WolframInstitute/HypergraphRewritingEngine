#include "benchmark_framework.hpp"
#include <cstdio>
#include <map>

#ifdef _WIN32
#include <windows.h>
#endif

int main(int argc, char** argv) {
#ifdef _WIN32
    // Set console to UTF-8 mode on Windows
    SetConsoleOutputCP(CP_UTF8);
#endif

    printf("Comprehensive Hypergraph Benchmark Suite\n");
    printf("==========================================\n\n");

    std::string output_dir = "benchmark_results";  // Default relative path
    std::string filter = "";
    bool list_only = false;
    bool output_dir_set = false;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--list") {
            list_only = true;
        } else if (arg.find("--filter=") == 0) {
            filter = arg.substr(9);
        } else if (arg.find("--output=") == 0) {
            output_dir = arg.substr(9);
            output_dir_set = true;
        } else if (!output_dir_set && !arg.empty() && arg[0] != '-') {
            // First positional argument is output directory
            output_dir = arg;
            output_dir_set = true;
        } else if (filter.empty() && !arg.empty() && arg[0] != '-') {
            // Second positional argument is filter
            filter = arg;
        }
    }

    // List benchmarks and exit if --list specified
    if (list_only) {
        benchmark::BenchmarkRegistry::instance().list_benchmarks(filter);
        return 0;
    }

    // Load calibration data if available
    std::map<std::string, size_t> calibration = benchmark::CalibrationLoader::load("benchmark_calibration.txt");

    if (!calibration.empty()) {
        printf("Loaded %zu calibrated sample counts\n\n", calibration.size());
    }

    benchmark::BenchmarkRegistry::instance().run_all(output_dir, calibration, filter);

    return 0;
}
