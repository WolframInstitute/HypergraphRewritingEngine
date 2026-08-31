#include "benchmark_framework.hpp"
#include "hgcommon/build_stamp.hpp"
#include <cstdio>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#endif

int main(int argc, char** argv) {
#ifdef _WIN32
    // Set console to UTF-8 mode on Windows
    SetConsoleOutputCP(CP_UTF8);
#endif

    // The configuration this binary was built with, first, so every log that carries a number
    // carries the record a reader needs to know what the number is evidence for; and alone on
    // `--build-info`, which is how tools/dev/perf_gate.sh gates before it times anything
    // (hgcommon/build_stamp.hpp).
    static const char kBuildStamp[] = HG_BUILD_STAMP_LITERAL;
    if (argc > 1 && std::strcmp(argv[1], "--build-info") == 0) { printf("%s\n", kBuildStamp); return 0; }
    printf("%s\n", kBuildStamp);

    printf("Comprehensive Hypergraph Benchmark Suite\n");
    printf("==========================================\n\n");

    std::string output_dir = "benchmark_results";  // Default relative path
    std::string filter = "";
    bool list_only = false;
    bool output_dir_set = false;
    bool include_reference = false;
    bool only_reference = false;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--list") {
            list_only = true;
        } else if (arg == "--reference") {
            only_reference = true;
        } else if (arg == "--include-reference") {
            include_reference = true;
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

    benchmark::BenchmarkRegistry::instance().run_all(output_dir, filter, include_reference, only_reference);

    return 0;
}
