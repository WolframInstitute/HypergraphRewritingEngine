cmake_minimum_required(VERSION 3.14)
# CMP0091 (CMake 3.15+) makes CMAKE_MSVC_RUNTIME_LIBRARY take effect, so a Windows/MSVC build
# can select the static C runtime (/MT). Must be set before project(). Without NEW the property
# is silently ignored and the binary keeps a VCRUNTIME140 / MSVCP140 (VC++ redist) dependency.
if(POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW)
endif()
project(HypergraphEngine VERSION 0.0.1 LANGUAGES CXX)

# Shared CPU/GPU definitions (hgcommon)
include_directories(${CMAKE_SOURCE_DIR}/common/include)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release build if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "Build type not specified, defaulting to Release")
endif()

# Detect WSL environment
# WSL (Windows Subsystem for Linux) requires special handling for:
# - File locking (Java/Lucene operations fail on WSL filesystem)
# - Path conversions between Linux and Windows
# Note: Use CMAKE_HOST_SYSTEM_NAME to detect the build host, not the target
set(IS_WSL FALSE)
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    if(EXISTS "/proc/sys/fs/binfmt_misc/WSLInterop" OR DEFINED ENV{WSL_DISTRO_NAME})
        set(IS_WSL TRUE)
        message(STATUS "WSL environment detected")
    endif()
endif()

# The commit every shipped artifact is stamped with (paclet_source/build_stamp.hpp).
#
# A release ships binaries for six platforms and the host that assembles them can run at most
# one, so "is this artifact current?" has to be answerable by READING the file. The stamp is how,
# and this is where the value it carries comes from.
#
# CMAKE_CONFIGURE_DEPENDS on .git/HEAD makes a branch switch or a checkout re-configure, so the
# stamp cannot outlive the commit it names. A commit ON the same branch does not touch HEAD's
# contents, which is why sign-off compares the stamp to `git rev-parse HEAD` rather than assuming
# a configured tree is up to date.
# PASSED IN WINS. build_windows_gpu.sh resolves the commit on the Linux side and hands it over
# with -D, because its configure runs under Windows cmake.exe against a \\wsl.localhost path where
# git finds no repository. An unconditional `set()` here would define a NORMAL variable that
# shadows that cache entry -- which is what left the one artifact the stamp exists for reading
# `commit=unknown` after the -D was already being passed.
find_package(Git QUIET)
if(NOT HG_BUILD_COMMIT)
    set(HG_BUILD_COMMIT "unknown")
endif()
if(HG_BUILD_COMMIT STREQUAL "unknown" AND GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
    execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
                    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                    OUTPUT_VARIABLE HG_BUILD_COMMIT
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
    if(HG_BUILD_COMMIT STREQUAL "")
        set(HG_BUILD_COMMIT "unknown")
    endif()
    if(EXISTS "${CMAKE_SOURCE_DIR}/.git/HEAD")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
                     "${CMAKE_SOURCE_DIR}/.git/HEAD")
    endif()
endif()
message(STATUS "Build stamp commit: ${HG_BUILD_COMMIT}")

# Export compile commands for clangd and other tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Create symlink to compile_commands.json in project root for clangd
# This runs at configure time, not build time
execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_BINARY_DIR}/compile_commands.json
        ${CMAKE_SOURCE_DIR}/compile_commands.json
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${CONFIG} CONFIG_UPPER)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_BINARY_DIR})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_BINARY_DIR})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONFIG_UPPER} ${CMAKE_BINARY_DIR})
endforeach()

# WebAssembly. The engine runs there in SERIAL mode -- no threads exist, which is why
# ParallelEvolutionEngine::ExecutionMode::Serial is the mode a wasm consumer selects.
#
# The link settings are not tuning. A wasm module gets a fixed 16 MB heap and a 64 KB stack
# unless told otherwise, and this engine outgrows both: the arena grows past 16 MB on any
# non-trivial evolution, and the matcher's per-frame scratch outgrows 64 KB. Without these an
# out-of-bounds trap is the FIRST thing a run produces.
if(EMSCRIPTEN)
    add_link_options(
        -sALLOW_MEMORY_GROWTH=1     # the arena grows; a fixed heap caps the reachable evolution
        -sINITIAL_MEMORY=64MB
        -sSTACK_SIZE=4MB
        -sEXIT_RUNTIME=1            # main()'s exit code is the gate's verdict
    )
endif()

# Benchmarking option for event relationship processing
option(SINGLE_THREADED_EVENT_RELATION_PROCESSING "Use single-threaded event relation processing for benchmarking" OFF)
if(SINGLE_THREADED_EVENT_RELATION_PROCESSING)
    add_compile_definitions(SINGLE_THREADED_EVENT_RELATION_PROCESSING)
endif()

# Fuzzing tests option (disabled by default for faster builds)
option(ENABLE_FUZZING_TESTS "Enable determinism fuzzing tests (slow but comprehensive)" OFF)
if(ENABLE_FUZZING_TESTS)
    add_compile_definitions(ENABLE_FUZZING_TESTS)
endif()

# Sanitizer options for detecting memory errors and race conditions
option(ENABLE_ASAN "Enable AddressSanitizer for memory error detection" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer for race condition detection" OFF)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)

# Sanitizer configuration (only one can be enabled at a time for ASAN/TSAN)
if(ENABLE_ASAN AND ENABLE_TSAN)
    message(FATAL_ERROR "Cannot enable both AddressSanitizer and ThreadSanitizer simultaneously")
endif()

if(ENABLE_ASAN)
    message(STATUS "AddressSanitizer enabled")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g -O1)
    add_link_options(-fsanitize=address)
endif()

if(ENABLE_TSAN)
    message(STATUS "ThreadSanitizer enabled")
    add_compile_options(-fsanitize=thread -fno-omit-frame-pointer -g -O1)
    add_link_options(-fsanitize=thread)
endif()

if(ENABLE_UBSAN)
    message(STATUS "UndefinedBehaviorSanitizer enabled")
    add_compile_options(-fsanitize=undefined -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=undefined)
endif()

# A name that shadows another is how two different things end up called one thing in a scope
# where both are live. Two were found in hypergraph_ffi.cpp by running this by hand -- `engine`
# re-bound to the same object it already named, and `key` meaning the OPTION name outside and a
# signature key inside, in the parse that decides event identity. Neither changed behaviour;
# both cost a reader the work of proving that.
#
# AN ERROR, NOT A WARNING. Both of those compiled and passed their tests, and a warning in a
# build that prints thousands of lines is a diagnostic nobody is required to read. The class is
# closed only if it cannot build. The whole host tree compiles clean under it -- engine, FFI,
# paclet and every test source -- so keeping it that way costs nothing, while a sweep run once
# protects only the day it was run.
#
# GNU/Clang only: MSVC spells its shadow diagnostics differently and the Windows leg has no
# equivalent to enable here.
#
# CXX ONLY, via the generator expression: add_compile_options applies to EVERY language, so an
# unscoped flag lands on the nvcc command line, changes it, and invalidates every .cu object --
# including persistent.cu, which needs about 12.5 GB to compile (#123) and cannot be rebuilt on a
# shared box. Scoped this way the CUDA compile lines are unchanged and nothing device-side
# rebuilds, while every .cpp the device targets carry is still covered.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wshadow>
                        $<$<COMPILE_LANGUAGE:CXX>:-Werror=shadow>)
endif()

add_subdirectory(lockfree_deque)
add_subdirectory(job_system)

# When visualization is enabled, create viz_events library BEFORE hypergraph
# so hypergraph can link to it for the VizEventSink static variables
if(BUILD_VISUALIZATION)
    add_library(viz_events STATIC
        visualisation/events/src/viz_event_sink.cpp
    )
    target_include_directories(viz_events PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/visualisation/events/include
    )
endif()

add_subdirectory(hypergraph)
add_subdirectory(wxf)

# Unified test suite
option(BUILD_TESTS "Build unified test suite" ON)
if(BUILD_TESTS)
    add_subdirectory(testing)
endif()

# Examples - API documentation and usage examples
option(BUILD_EXAMPLES "Build examples" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Regenerates reference/golden_matrix.txt, which the identity-matrix tests read. A build target
# rather than an ad-hoc compile: the file it writes is the expected output of 216 runs, so
# reproducing it has to be one command with the project's own include paths and flags.
# The exactness-and-cost harness. It proves output EXACT against the brute-force oracle while
# reporting arena bytes_allocated(), which is durable and noise-free where RSS is not -- so a
# change can be shown to preserve correctness and to move memory in a reproducible direction.
add_executable(event_key_decomposition_probe tools/event_key_decomposition_probe.cpp)
target_include_directories(event_key_decomposition_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(event_key_decomposition_probe PRIVATE hypergraph)

add_executable(thinning_mode_probe tools/thinning_mode_probe.cpp)
target_include_directories(thinning_mode_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(thinning_mode_probe PRIVATE hypergraph)

add_executable(cost_matrix tools/cost_matrix.cpp)
target_include_directories(cost_matrix PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cost_matrix PRIVATE hypergraph)

add_executable(generate_golden_matrix tools/generate_golden_matrix.cpp)
target_include_directories(generate_golden_matrix PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(generate_golden_matrix PRIVATE hypergraph)

# Repeat-count instrument for event identity stability. The gate compares one worker count
# against one other; this asks the same question many times over a spread of worker counts, which
# is what makes a "stable" reading evidence rather than a single agreement.
add_executable(event_signature_stability_probe tools/event_signature_stability_probe.cpp)
target_include_directories(event_signature_stability_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(event_signature_stability_probe PRIVATE hypergraph)

# Failure RATE of quotient causal determinism, over the sweep the pass/fail gate runs once.
# A gate that answers pass/fail cannot tell "fixed" from "did not fire this time", which is the
# whole question for a defect observed once in ~90 runs. --load runs background threads, since
# the one observation came from inside a full suite run.
add_executable(quotient_determinism_rate_probe tools/quotient_determinism_rate_probe.cpp)
target_include_directories(quotient_determinism_rate_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quotient_determinism_rate_probe PRIVATE hypergraph)

# Host counterpart of gpu_rank_order_probe: separates a stable presentation order from an
# order-independent rank rule as the reason the CPU's event identities do not move.
add_executable(cpu_rank_order_probe tools/cpu_rank_order_probe.cpp)
target_include_directories(cpu_rank_order_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cpu_rank_order_probe PRIVATE hypergraph)

# Benchmarks - Performance measurement tools
option(BUILD_BENCHMARKS "Build benchmarks" ON)
if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarking)
    add_subdirectory(benchmarks)
endif()

# Optional: Build GPU acceleration (requires CUDA). Added before the paclet so
# the paclet's optional GPU binary (hg_evolve_gpu) can link the hg_gpu target.
option(BUILD_GPU "Build GPU-accelerated evolution engine" OFF)
if(BUILD_GPU)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER}")
        message(STATUS "CUDA version: ${CMAKE_CUDA_COMPILER_VERSION}")
        add_subdirectory(gpu)
    else()
        message(WARNING "CUDA not found - GPU support disabled")
    endif()
endif()

# Optional: Build Wolfram Language paclet if the Wolfram Language is found
option(BUILD_WOLFRAM_LANGUAGE_PACLET "Build Wolfram Language paclet library" ON)
if(BUILD_WOLFRAM_LANGUAGE_PACLET)
    add_subdirectory(paclet_source)
endif()

# Viz-event emission. The interactive renderer lives in its own repo
# (../hypergraph_viz, split with history 2026-08-01); this switch builds the
# event-stream side it consumes: viz_events (above) plus the engine's
# VIZ_EMIT_* instrumentation (hypergraph/CMakeLists.txt). visualisation/ here
# retains only that interface (events/) and the blackhole analysis library the
# paclet compiles directly.
option(BUILD_VISUALIZATION "Emit visualization events for the external renderer" OFF)

# Every host tool gets a build target, automatically.
#
# 51 of 57 files under tools/ had none. `make <tool>` answered "No rule to make target", so each
# was compiled by hand once, produced its number, and then rotted -- including
# forwarding_validator_probe, the sole caller of the match-completeness validator, and cost_matrix,
# which describes itself as the harness the paper's ablation uses. A measurement whose instrument
# cannot be rebuilt cannot be repeated, and a number that cannot be repeated is not evidence.
#
# EVERY tool now has one, and `make host_tools` compiles them all, so a tool that falls behind
# the API it measures is found on purpose rather than when someone next needs it.
#
# EXCLUDE_FROM_ALL so a probe that has fallen behind the API it measures does not break the
# default build or CI, while `make <tool>` still works and the rot is discoverable rather than
# invisible. `make host_tools` builds them all, which is what turns "this probe no longer
# compiles" into something a person finds on purpose instead of by needing it.
#
# Tools declared explicitly above keep their own targets; this skips those names.
file(GLOB HG_TOOL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/tools/*.cpp)
add_custom_target(host_tools)
foreach(tool_src ${HG_TOOL_SOURCES})
    get_filename_component(tool_name ${tool_src} NAME_WE)
    # A tool that includes the CUDA headers belongs to the GPU build, not this one, and would
    # fail here for a reason that says nothing about the tool.
    file(READ ${tool_src} tool_text)
    string(FIND "${tool_text}" "hg_gpu/" tool_needs_gpu)
    # A tool that is not part of the repository. ir_incremental_probe.cpp is one clone's local
    # scratch (.git/info/exclude), and it reads IRCanonicalizer::last_stats, which the class
    # does not have -- so the glob, which sees the working tree rather than the index, would
    # break host_tools for everyone who happens to have it. Named rather than silently skipped:
    # a skip by pattern is how an unbuilt file stops being noticed.
    set(HG_UNTRACKED_TOOLS ir_incremental_probe)
    list(FIND HG_UNTRACKED_TOOLS ${tool_name} tool_untracked)
    if(NOT TARGET ${tool_name} AND tool_needs_gpu EQUAL -1 AND tool_untracked EQUAL -1)
        add_executable(${tool_name} EXCLUDE_FROM_ALL ${tool_src})
        target_include_directories(${tool_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(${tool_name} PRIVATE hypergraph)
        add_dependencies(host_tools ${tool_name})
    endif()
endforeach()

# Probes that already decide a verdict and exit non-zero become CTest gates.
#
# quotient_reconstruction_probe verifies the KEYSTONE claim of the quotient optimisation --
# that the full-expansion causal multiset is recoverable from the quotient skeleton -- across 24
# workloads, and returns 1 if any of them disagrees with the oracle. It ran in 0.04 s and nothing
# in the suite asserted it: the claim was checked by hand, when someone remembered.
#
# It stays EXCLUDE_FROM_ALL as a tool; `ctest` builds and runs it. A probe that already computes
# a pass/fail is a gate that was never wired up, not a tool.
enable_testing()
#
# causal_tr_exactness_probe is the second: it checks that online transitive reduction emits
# exactly the minimal edge set, at every thread count, and likewise returns non-zero on any
# disagreement. Also 0.06 s, also previously unasserted.
#
# The other reconstruction probes (events, matrix, observables) PRINT rather than decide, so they
# stay tools -- a probe with no verdict cannot be a gate without first being given one.
add_test(NAME quotient_reconstruction
         COMMAND $<TARGET_FILE:quotient_reconstruction_probe>)
add_test(NAME causal_tr_exactness
         COMMAND $<TARGET_FILE:causal_tr_exactness_probe>)
set_tests_properties(quotient_reconstruction causal_tr_exactness PROPERTIES TIMEOUT 300)
add_dependencies(host_tools quotient_reconstruction_probe causal_tr_exactness_probe)

# The GenMC harnesses, for the same reason as the two probes above: they already decide, and a
# check nobody runs is a check that stops being true. Two defects reached the tree while the
# stress tests covering those structures were green, so these run with the rest of the suite.
#
# The harnesses transcribe or include memory orders from the engine headers, and one of them
# (job_system_no_lost_wakeup) proved it can go stale the moment the header it mirrors is fixed
# without it. Running on every `ctest` is what surfaces that.
#
# SKIPPED, not failed, where genmc is absent: it is a separate build against LLVM 18 and is not a
# dependency of this project. run.sh exits 2 in that case, which is mapped to a skip so a machine
# without it reports honestly rather than either lying green or failing red.
# SPEC 5.4 as a standing gate: every quotient golden row must equal its full-capture twin on
# every count column AND the state fingerprint. Pure python over the checked-in golden file, so
# it costs milliseconds and cannot drift from what the golden gate itself compares.
add_test(NAME quotient_twin_check
         COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/golden_quotient_twin_check.py
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set_tests_properties(quotient_twin_check PROPERTIES TIMEOUT 60)

add_test(NAME genmc_harnesses
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/verification/genmc/run.sh all)
set_tests_properties(genmc_harnesses PROPERTIES
                     TIMEOUT 900
                     SKIP_RETURN_CODE 2)
