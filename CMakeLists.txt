cmake_minimum_required(VERSION 3.14)
# CMP0091 (CMake 3.15+) makes CMAKE_MSVC_RUNTIME_LIBRARY take effect, so a Windows/MSVC build
# can select the static C runtime (/MT). Must be set before project(). Without NEW the property

# Per-phase cycle counters on the host (hgcommon/phase_timing.hpp).
#
# OFF for anything shipped. The counters are one cache line per worker per phase and the timer
# reads the cycle counter twice per phase entry, so a release build must not carry them -- with
# this off the header defines PhaseTimer as an empty object and no storage exists at all.
# Turn it on only to answer "where do this run's cycles go", which is a question the wall clock
# and the instruction count both answer wrongly here.
option(HG_ENGINE_STATS "Diagnostic counters the determinism gates and instruments read; ON for test and development builds, OFF for shipped and verified ones" ON)
add_compile_definitions(HG_ENGINE_STATS=$<BOOL:${HG_ENGINE_STATS}>)

option(HG_PHASE_TIMING "Host per-phase cycle counters (measurement builds only)" OFF)
if(HG_PHASE_TIMING)
    add_compile_definitions(HG_PHASE_TIMING)
    message(STATUS "Host phase timing ENABLED -- this build is for measurement, not for shipping")
endif()

option(HG_BITSET_STATS "SparseBitset search-depth histogram (measurement builds only)" OFF)
if(HG_BITSET_STATS)
    add_compile_definitions(HG_BITSET_STATS)
endif()

option(HG_MATCH_BRANCH_STATS "Matcher candidate-branch coverage counters (measurement builds only)" OFF)
if(HG_MATCH_BRANCH_STATS)
    add_compile_definitions(HG_MATCH_BRANCH_STATS)
endif()
# is silently ignored and the binary keeps a VCRUNTIME140 / MSVCP140 (VC++ redist) dependency.
if(POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW)
endif()
project(HypergraphEngine VERSION 1.0.0 LANGUAGES CXX)

# ENABLE_TESTING() MUST PRECEDE EVERY add_subdirectory(), AND DID NOT.
#
# A directory processed before this call gets no CTestTestfile.cmake, and its add_test() and
# gtest_discover_tests() registrations are dropped WITHOUT A WARNING. It used to sit near the
# end of this file, after all twenty-odd subdirectories, so `ctest -N` listed 49 tests -- the
# lockfree_deque and job_system suites, which call enable_testing() in their own scopes, plus
# four root tests -- and NONE of the engine suites. all_tests (296 tests), hg_gpu_tests and
# gpu_differential_tests were registered into files nothing included.
#
# That is not cosmetic: .github/workflows/windows.yml gates on `ctest -E genmc` and nothing
# else, so the Windows lane ran 49 concurrency tests and never once ran the engine suite. The
# Linux, macOS and GPU lanes invoke their binaries directly and were unaffected, which is why
# this stayed invisible.
enable_testing()

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
# GIT WINS WHEN IT CAN ANSWER. A -D HG_BUILD_COMMIT (or a value a previous configure left in
# this build directory's cache) is used only where git cannot resolve HEAD -- the Windows
# configure against a \\wsl.localhost path is that case. A cached value taken over a live git
# answer stamped a whole build directory with the commit of an earlier release configure.
set(HG_GIT_COMMIT "")
if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
    execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
                    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                    OUTPUT_VARIABLE HG_GIT_COMMIT
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
    # RECONFIGURE WHEN HEAD MOVES. A commit rewrites the branch ref and appends to the reflog;
    # HEAD itself changes only on a checkout. Depending on HEAD alone left a warm build stamped
    # with the commit of an earlier configure (measured: a build_release bench reading
    # 9e8f39aa two commits later). The paths come from git so a worktree, whose .git is a
    # file pointing elsewhere, resolves them too.
    foreach(rel HEAD logs/HEAD)
        execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse --git-path "${rel}"
                        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                        OUTPUT_VARIABLE HG_GIT_DEP
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        ERROR_QUIET)
        if(HG_GIT_DEP AND NOT IS_ABSOLUTE "${HG_GIT_DEP}")
            set(HG_GIT_DEP "${CMAKE_SOURCE_DIR}/${HG_GIT_DEP}")
        endif()
        if(HG_GIT_DEP AND EXISTS "${HG_GIT_DEP}")
            set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${HG_GIT_DEP}")
        endif()
    endforeach()
endif()
if(NOT HG_GIT_COMMIT STREQUAL "")
    set(HG_BUILD_COMMIT "${HG_GIT_COMMIT}")
elseif(NOT HG_BUILD_COMMIT)
    set(HG_BUILD_COMMIT "unknown")
endif()
message(STATUS "Build stamp commit: ${HG_BUILD_COMMIT}")

# The build stamp (common/include/hgcommon/build_stamp.hpp) carries the configuration of every
# artifact and bench: the commit, the build type and the flags come from here; the option
# macros the stamp reads (HG_ENGINE_STATS, HG_PHASE_TIMING, NDEBUG, the sanitizers) are the
# ones the options themselves set, so the record cannot disagree with the code it sits in.
string(TOUPPER "${CMAKE_BUILD_TYPE}" HG_BUILD_TYPE_UPPER)
set(HG_BUILD_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${HG_BUILD_TYPE_UPPER}}")
string(STRIP "${HG_BUILD_FLAGS}" HG_BUILD_FLAGS)
# A generated header, not a definition on every target: only the translation units that
# define a stamp recompile when the commit changes.
configure_file("${CMAKE_SOURCE_DIR}/common/include/hgcommon/build_commit.hpp.in"
               "${CMAKE_BINARY_DIR}/generated/hgcommon/build_commit.hpp" @ONLY)
include_directories("${CMAKE_BINARY_DIR}/generated")

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
    add_compile_definitions(HG_BUILD_ASAN=1)
    add_link_options(-fsanitize=address)
endif()

if(ENABLE_TSAN)
    message(STATUS "ThreadSanitizer enabled")
    add_compile_options(-fsanitize=thread -fno-omit-frame-pointer -g -O1)
    add_compile_definitions(HG_BUILD_TSAN=1)
    add_link_options(-fsanitize=thread)
endif()

if(ENABLE_UBSAN)
    message(STATUS "UndefinedBehaviorSanitizer enabled")
    add_compile_options(-fsanitize=undefined -fno-omit-frame-pointer -g)
    add_compile_definitions(HG_BUILD_UBSAN=1)
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

    # A brace initializer that runs out before the members do is silently zero-filled, and the
    # members it leaves behind are the ones added last -- so a struct that grows acquires
    # zeroed fields at every existing construction site and nothing says so. The determinism
    # test's failure diagnostic was built that way: four counters naming which drop mechanism
    # lost a match were computed, never copied into the record that prints them, and reported
    # clean whatever had happened.
    #
    # Not part of -Wall, and -Wextra is not enabled here, so nothing was checking. Measured
    # across all 151 C++ translation units at zero, and over the engine sources at zero on
    # clang, aarch64-linux-gnu-g++ and x86_64-w64-mingw32-g++ as well, so -Werror costs nothing
    # on any toolchain this tree is built with.
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wmissing-field-initializers>
                        $<$<COMPILE_LANGUAGE:CXX>:-Werror=missing-field-initializers>)

    # A switch over an enum that does not name every value falls to whatever the default does,
    # and the default is usually a summary that is now wrong for the case it swallowed:
    # JobSystem::get_error_description() described a CapacityExhausted run as "Unknown error",
    # which is the opposite of what classifying that error separately achieves -- it is the one
    # kind that is NOT a defect, and the whole point of the type is that a caller can tell.
    #
    # -Wswitch is in -Wall and was firing on every build of this tree; nothing acted on it.
    # Measured at zero after that one case was added: 151 TUs in build_linux, 162 in build_gpu,
    # and zero on clang, aarch64-linux-gnu-g++ and x86_64-w64-mingw32-g++ over the engine
    # sources, so -Werror costs nothing on any toolchain this tree is built with.
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Werror=switch>)
endif()

# LINK-TIME OPTIMISATION, AND WHY IT IS NOT A MICRO-OPTIMISATION HERE.
#
# Without it the compiler can only inline what it can SEE, which is whatever a translation unit
# includes. That single fact is why an engine like this drifts into being written in headers:
# move a hot accessor into a .cpp and every other TU gets a real call plus the loss of constant
# propagation and register allocation across the boundary, so the header is the only place the
# optimiser can do its job. Structure then follows the optimiser instead of the other way round.
#
# With IPO the linker inlines across TUs, so where a body LIVES stops being a performance
# decision and becomes an organisational one -- which is the point.
#
# Release only: a debug build gains nothing and pays the link cost. CUDA targets are excluded
# below, because device code is linked by nvcc's own device-link step and IPO does not apply to
# it. Guarded by check_ipo_supported so a toolchain without it configures rather than fails.
include(CheckIPOSupported)
check_ipo_supported(RESULT HG_IPO_SUPPORTED OUTPUT HG_IPO_WHY)
if(HG_IPO_SUPPORTED)
    # A PLAIN set() SHADOWS THE CACHE, so -DCMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE=FALSE
    # was accepted and then overridden: the caller got LTO anyway and nothing said so. That is
    # the same defect class as HG_GPU_ARCHS forcing CUDA_ARCHITECTURES, and it invalidated a
    # measurement -- a Windows crash was tested with LTO "off" and the flag never took.
    # Defaulted, not forced, and the choice is printed.
    if(NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()
    if(NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
    endif()
    message(STATUS "Link-time optimisation (Release): ${CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE}")
else()
    message(STATUS "Link-time optimisation: unavailable (${HG_IPO_WHY})")
endif()

# POPCNT, on the x86-64 hosts that have had it since 2008 (SSE4.2 / x86-64-v2). The baseline
# x86-64 ISA does not include the instruction, so __builtin_popcountll compiles to a libgcc CALL
# -- __popcountdi2 shows up in the engine's own profile, and the bitsets that call it are on the
# hot path. Scoped to CXX for the same reason as above: an unscoped flag reaches nvcc and
# invalidates every .cu object.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND
   CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-mpopcnt>)
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

# S2's instrument. Measures the individualization-refinement search's SIZE (leaves, nodes,
# depth) rather than its wall time, and calibrates itself on C_6 -- a 6-cycle's automorphism
# group is dihedral of order 12 and its initial refinement is one cell, so the search must
# branch. If that case reports no search the counters are dead and the rest is meaningless.
add_executable(ir_core_phase_probe tools/ir_core_phase_probe.cpp)
target_include_directories(ir_core_phase_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ir_core_phase_probe PRIVATE hypergraph)

# S2's second instrument: IR against WL, per CALL. Its own header states the scope that makes
# the ratio safe to read -- a per-call ratio does not license an engine-level route choice, and
# the engine-level number runs the other way (Full/IR 59,028,301 instructions against
# non-Full/WL's 68,004,245, because the engine pays the allocation discipline around WL).
add_executable(ir_vs_wl tools/ir_vs_wl.cpp)
target_include_directories(ir_vs_wl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ir_vs_wl PRIVATE hypergraph)


# ---------------------------------------------------------------------------------------------
# EVERY REMAINING tools/ PROGRAM, REGISTERED.
#
# 55 of 66 files here were built by nothing. That is not a tidiness point: a probe nothing
# compiles rots invisibly and, worse, its FINDINGS go missing. Registering two of them on
# 2026-08-18 (ir_core_phase_probe, ir_vs_wl) is what closed the "IR 1100x high-symmetry
# pathology" as unreproducible AND found a per-call memset worth 1.09% of engine instructions --
# a real cost that survived precisely because its instrument was unregistered.
#
# All 53 below compile against the current tree. The one that does not, ir_incremental_probe.cpp,
# is deleted in this commit; the project's own CLAUDE.md already named it as broken.
#
# Registered with EXCLUDE_FROM_ALL: they build under `cmake --build . --target <name>` and on
# `--target probes`, but are not on the default build, so the ordinary path pays nothing while
# CI can compile them all and catch rot at the commit that causes it.
# ---------------------------------------------------------------------------------------------
add_custom_target(probes)

add_executable(park_cost_bench EXCLUDE_FROM_ALL tools/park_cost_bench.cpp)
target_include_directories(park_cost_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(park_cost_bench PRIVATE job_system)
add_dependencies(probes park_cost_bench)
add_executable(arena_bump_bench EXCLUDE_FROM_ALL tools/arena_bump_bench.cpp)
target_include_directories(arena_bump_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(arena_bump_bench PRIVATE hypergraph)
add_dependencies(probes arena_bump_bench)
add_executable(arena_reset_test EXCLUDE_FROM_ALL tools/arena_reset_test.cpp)
target_include_directories(arena_reset_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(arena_reset_test PRIVATE hypergraph)
add_dependencies(probes arena_reset_test)
add_executable(branchial_flux_probe EXCLUDE_FROM_ALL tools/branchial_flux_probe.cpp)
target_include_directories(branchial_flux_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(branchial_flux_probe PRIVATE hypergraph)
add_dependencies(probes branchial_flux_probe)
add_executable(branchial_nondeterminism_probe EXCLUDE_FROM_ALL tools/branchial_nondeterminism_probe.cpp)
target_include_directories(branchial_nondeterminism_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(branchial_nondeterminism_probe PRIVATE hypergraph)
add_dependencies(probes branchial_nondeterminism_probe)
add_executable(budget_collapse_probe EXCLUDE_FROM_ALL tools/budget_collapse_probe.cpp)
target_include_directories(budget_collapse_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(budget_collapse_probe PRIVATE hypergraph)
add_dependencies(probes budget_collapse_probe)
add_executable(canonical_causal_oracle EXCLUDE_FROM_ALL tools/canonical_causal_oracle.cpp)
target_include_directories(canonical_causal_oracle PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(canonical_causal_oracle PRIVATE hypergraph)
add_dependencies(probes canonical_causal_oracle)
add_executable(causal_tr_determinism_probe EXCLUDE_FROM_ALL tools/causal_tr_determinism_probe.cpp)
target_include_directories(causal_tr_determinism_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(causal_tr_determinism_probe PRIVATE hypergraph)
add_dependencies(probes causal_tr_determinism_probe)
add_executable(causal_tr_exactness_probe EXCLUDE_FROM_ALL tools/causal_tr_exactness_probe.cpp)
target_include_directories(causal_tr_exactness_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(causal_tr_exactness_probe PRIVATE hypergraph)
add_dependencies(probes causal_tr_exactness_probe)
add_executable(deque_aba_headroom EXCLUDE_FROM_ALL tools/deque_aba_headroom.cpp)
target_include_directories(deque_aba_headroom PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(deque_aba_headroom PRIVATE hypergraph)
add_dependencies(probes deque_aba_headroom)
add_executable(determinism_forwarding_repro EXCLUDE_FROM_ALL tools/determinism_forwarding_repro.cpp)
target_include_directories(determinism_forwarding_repro PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(determinism_forwarding_repro PRIVATE hypergraph)
add_dependencies(probes determinism_forwarding_repro)
add_executable(forwarding_validator_probe EXCLUDE_FROM_ALL tools/forwarding_validator_probe.cpp)
target_include_directories(forwarding_validator_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(forwarding_validator_probe PRIVATE hypergraph)
add_dependencies(probes forwarding_validator_probe)
add_executable(higgs_shadow_probe EXCLUDE_FROM_ALL tools/higgs_shadow_probe.cpp)
target_include_directories(higgs_shadow_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(higgs_shadow_probe PRIVATE hypergraph)
add_dependencies(probes higgs_shadow_probe)
add_executable(incremental_probe EXCLUDE_FROM_ALL tools/incremental_probe.cpp)
target_include_directories(incremental_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(incremental_probe PRIVATE hypergraph)
add_dependencies(probes incremental_probe)
add_executable(incremental_wl_probe EXCLUDE_FROM_ALL tools/incremental_wl_probe.cpp)
target_include_directories(incremental_wl_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(incremental_wl_probe PRIVATE hypergraph)
add_dependencies(probes incremental_wl_probe)
add_executable(ir_core_bench EXCLUDE_FROM_ALL tools/ir_core_bench.cpp)
target_include_directories(ir_core_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ir_core_bench PRIVATE hypergraph)
add_dependencies(probes ir_core_bench)
add_executable(ir_core_equivalence_probe EXCLUDE_FROM_ALL tools/ir_core_equivalence_probe.cpp)
target_include_directories(ir_core_equivalence_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ir_core_equivalence_probe PRIVATE hypergraph)
add_dependencies(probes ir_core_equivalence_probe)
add_executable(ir_edge_map_probe EXCLUDE_FROM_ALL tools/ir_edge_map_probe.cpp)
target_include_directories(ir_edge_map_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ir_edge_map_probe PRIVATE hypergraph)
add_dependencies(probes ir_edge_map_probe)
add_executable(ir_edge_orbit_probe EXCLUDE_FROM_ALL tools/ir_edge_orbit_probe.cpp)
target_include_directories(ir_edge_orbit_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ir_edge_orbit_probe PRIVATE hypergraph)
add_dependencies(probes ir_edge_orbit_probe)
add_executable(ir_malloc_bench EXCLUDE_FROM_ALL tools/ir_malloc_bench.cpp)
target_include_directories(ir_malloc_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ir_malloc_bench PRIVATE hypergraph)
add_dependencies(probes ir_malloc_bench)
add_executable(mode_matrix_probe EXCLUDE_FROM_ALL tools/mode_matrix_probe.cpp)
target_include_directories(mode_matrix_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mode_matrix_probe PRIVATE hypergraph)
add_dependencies(probes mode_matrix_probe)
add_executable(multi_init_rule_2x2_probe EXCLUDE_FROM_ALL tools/multi_init_rule_2x2_probe.cpp)
target_include_directories(multi_init_rule_2x2_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(multi_init_rule_2x2_probe PRIVATE hypergraph)
add_dependencies(probes multi_init_rule_2x2_probe)
add_executable(multiplicity_propagation_probe EXCLUDE_FROM_ALL tools/multiplicity_propagation_probe.cpp)
target_include_directories(multiplicity_propagation_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(multiplicity_propagation_probe PRIVATE hypergraph)
add_dependencies(probes multiplicity_propagation_probe)
add_executable(profile_evolve EXCLUDE_FROM_ALL tools/profile_evolve.cpp)
target_include_directories(profile_evolve PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(profile_evolve PRIVATE hypergraph)
add_dependencies(probes profile_evolve)
add_executable(push_site_probe EXCLUDE_FROM_ALL tools/push_site_probe.cpp)
target_include_directories(push_site_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(push_site_probe PRIVATE hypergraph)
add_dependencies(probes push_site_probe)
add_executable(quotient_branchial_diff EXCLUDE_FROM_ALL tools/quotient_branchial_diff.cpp)
target_include_directories(quotient_branchial_diff PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quotient_branchial_diff PRIVATE hypergraph)
add_dependencies(probes quotient_branchial_diff)
add_executable(quotient_branchial_gap_probe EXCLUDE_FROM_ALL tools/quotient_branchial_gap_probe.cpp)
target_include_directories(quotient_branchial_gap_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quotient_branchial_gap_probe PRIVATE hypergraph)
add_dependencies(probes quotient_branchial_gap_probe)
add_executable(quotient_causal_support_probe EXCLUDE_FROM_ALL tools/quotient_causal_support_probe.cpp)
target_include_directories(quotient_causal_support_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quotient_causal_support_probe PRIVATE hypergraph)
add_dependencies(probes quotient_causal_support_probe)
add_executable(quotient_causal_tr_deadend_probe EXCLUDE_FROM_ALL tools/quotient_causal_tr_deadend_probe.cpp)
target_include_directories(quotient_causal_tr_deadend_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quotient_causal_tr_deadend_probe PRIVATE hypergraph)
add_dependencies(probes quotient_causal_tr_deadend_probe)
add_executable(quotient_causal_tr_isolate_probe EXCLUDE_FROM_ALL tools/quotient_causal_tr_isolate_probe.cpp)
target_include_directories(quotient_causal_tr_isolate_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quotient_causal_tr_isolate_probe PRIVATE hypergraph)
add_dependencies(probes quotient_causal_tr_isolate_probe)
add_executable(quotient_causal_tr_recon_probe EXCLUDE_FROM_ALL tools/quotient_causal_tr_recon_probe.cpp)
target_include_directories(quotient_causal_tr_recon_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quotient_causal_tr_recon_probe PRIVATE hypergraph)
add_dependencies(probes quotient_causal_tr_recon_probe)
add_executable(quotient_expansion_capture_probe EXCLUDE_FROM_ALL tools/quotient_expansion_capture_probe.cpp)
target_include_directories(quotient_expansion_capture_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quotient_expansion_capture_probe PRIVATE hypergraph)
add_dependencies(probes quotient_expansion_capture_probe)
add_executable(quotient_observable_gap_probe EXCLUDE_FROM_ALL tools/quotient_observable_gap_probe.cpp)
target_include_directories(quotient_observable_gap_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quotient_observable_gap_probe PRIVATE hypergraph)
add_dependencies(probes quotient_observable_gap_probe)
add_executable(quotient_pairs_reconstruction_probe EXCLUDE_FROM_ALL tools/quotient_pairs_reconstruction_probe.cpp)
target_include_directories(quotient_pairs_reconstruction_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quotient_pairs_reconstruction_probe PRIVATE hypergraph)
add_dependencies(probes quotient_pairs_reconstruction_probe)
add_executable(quotient_raw_causal_reconstruction_probe EXCLUDE_FROM_ALL tools/quotient_raw_causal_reconstruction_probe.cpp)
target_include_directories(quotient_raw_causal_reconstruction_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quotient_raw_causal_reconstruction_probe PRIVATE hypergraph)
add_dependencies(probes quotient_raw_causal_reconstruction_probe)
add_executable(quotient_reconstruction_cost_probe EXCLUDE_FROM_ALL tools/quotient_reconstruction_cost_probe.cpp)
target_include_directories(quotient_reconstruction_cost_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quotient_reconstruction_cost_probe PRIVATE hypergraph)
add_dependencies(probes quotient_reconstruction_cost_probe)
add_executable(quotient_reconstruction_events_probe EXCLUDE_FROM_ALL tools/quotient_reconstruction_events_probe.cpp)
target_include_directories(quotient_reconstruction_events_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quotient_reconstruction_events_probe PRIVATE hypergraph)
add_dependencies(probes quotient_reconstruction_events_probe)
add_executable(quotient_reconstruction_matrix_probe EXCLUDE_FROM_ALL tools/quotient_reconstruction_matrix_probe.cpp)
target_include_directories(quotient_reconstruction_matrix_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quotient_reconstruction_matrix_probe PRIVATE hypergraph)
add_dependencies(probes quotient_reconstruction_matrix_probe)
add_executable(quotient_reconstruction_observables_probe EXCLUDE_FROM_ALL tools/quotient_reconstruction_observables_probe.cpp)
target_include_directories(quotient_reconstruction_observables_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quotient_reconstruction_observables_probe PRIVATE hypergraph)
add_dependencies(probes quotient_reconstruction_observables_probe)
add_executable(quotient_reconstruction_probe EXCLUDE_FROM_ALL tools/quotient_reconstruction_probe.cpp)
target_include_directories(quotient_reconstruction_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quotient_reconstruction_probe PRIVATE hypergraph)
add_dependencies(probes quotient_reconstruction_probe)
add_executable(quotient_routing_probe EXCLUDE_FROM_ALL tools/quotient_routing_probe.cpp)
target_include_directories(quotient_routing_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(quotient_routing_probe PRIVATE hypergraph)
add_dependencies(probes quotient_routing_probe)
add_executable(representation_prototype EXCLUDE_FROM_ALL tools/representation_prototype.cpp)
target_include_directories(representation_prototype PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(representation_prototype PRIVATE hypergraph)
add_dependencies(probes representation_prototype)
add_executable(sampling_cost_probe EXCLUDE_FROM_ALL tools/sampling_cost_probe.cpp)
target_include_directories(sampling_cost_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sampling_cost_probe PRIVATE hypergraph)
add_dependencies(probes sampling_cost_probe)
add_executable(sampling_cost_smoke EXCLUDE_FROM_ALL tools/sampling_cost_smoke.cpp)
target_include_directories(sampling_cost_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sampling_cost_smoke PRIVATE hypergraph)
add_dependencies(probes sampling_cost_smoke)
add_executable(segmented_array_stress EXCLUDE_FROM_ALL tools/segmented_array_stress.cpp)
target_include_directories(segmented_array_stress PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(segmented_array_stress PRIVATE hypergraph)
add_dependencies(probes segmented_array_stress)
add_executable(spine_probe EXCLUDE_FROM_ALL tools/spine_probe.cpp)
target_include_directories(spine_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(spine_probe PRIVATE hypergraph)
add_dependencies(probes spine_probe)
add_executable(thinning_recovery_probe EXCLUDE_FROM_ALL tools/thinning_recovery_probe.cpp)
target_include_directories(thinning_recovery_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(thinning_recovery_probe PRIVATE hypergraph)
add_dependencies(probes thinning_recovery_probe)
add_executable(timing_harness EXCLUDE_FROM_ALL tools/timing_harness.cpp)
target_include_directories(timing_harness PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(timing_harness PRIVATE hypergraph)
add_dependencies(probes timing_harness)
add_executable(unfolding_ratio_probe EXCLUDE_FROM_ALL tools/unfolding_ratio_probe.cpp)
target_include_directories(unfolding_ratio_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(unfolding_ratio_probe PRIVATE hypergraph)
add_dependencies(probes unfolding_ratio_probe)
add_executable(wl_core_bitid_check EXCLUDE_FROM_ALL tools/wl_core_bitid_check.cpp)
target_include_directories(wl_core_bitid_check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(wl_core_bitid_check PRIVATE hypergraph)
add_dependencies(probes wl_core_bitid_check)
add_executable(wl_engine_incremental EXCLUDE_FROM_ALL tools/wl_engine_incremental.cpp)
target_include_directories(wl_engine_incremental PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(wl_engine_incremental PRIVATE hypergraph)
add_dependencies(probes wl_engine_incremental)
add_executable(wl_iso_invariance_probe EXCLUDE_FROM_ALL tools/wl_iso_invariance_probe.cpp)
target_include_directories(wl_iso_invariance_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(wl_iso_invariance_probe PRIVATE hypergraph)
add_dependencies(probes wl_iso_invariance_probe)
add_executable(wl_sparse_prototype EXCLUDE_FROM_ALL tools/wl_sparse_prototype.cpp)
target_include_directories(wl_sparse_prototype PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(wl_sparse_prototype PRIVATE hypergraph)
add_dependencies(probes wl_sparse_prototype)

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
# enable_testing() is called immediately after project(); see the note there.
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

# THE TIMEOUT IS MEASURED, NOT GUESSED, and it has to be: a model-checking gate that is cut off
# mid-enumeration reports a failure indistinguishable from a real violation, and one set below
# what the suite costs is a gate that can only ever fail. Per-harness wall clock, measured
# 2026-08-23 on this box at each harness's own recorded bound, box load 4-8:
#
#   concurrent_map_double_growth_3t   140.58 s   (26,573 executions, --sc, four-context bound)
#   concurrent_map_repeated_offer      88.57 s
#   key_set_exactly_once_3t            80.86 s
#   concurrent_map_double_growth_2t    68.14 s   (34,485 executions, RC11, no bound)
#   concurrent_map_resize              37.76 s
#   the remaining fourteen             ~15 s combined
#
# About 430 s of enumeration plus a clang compile per harness. The expensive ones are the
# three-worker shapes across a table growth, which is where the defects are: every double-claim
# this project has found needed a growth concurrent with two claimants.
#
# REDUCING THEM IS NOT AVAILABLE, tested rather than assumed. Dropping a three-worker harness
# from --bound=4 to --bound=3 makes it finish sooner and STOP DETECTING its defect: with the
# guard it exists for deleted, it reports no violation in 21,041 executions at three, where at
# four it reports one. Shrinking the table instead (working_capacity 16 -> 4) does not help,
# since the cost is the three-thread interleaving space and not the table width. So the budget
# is stated from the measurement, with room for a loaded box.
# NOT REGISTERED ON WINDOWS. All three of these gates are driven by a POSIX shell script, and
# ctest on a Windows runner cannot launch one: it reports BAD_COMMAND, which is a FAILURE and not
# a skip, so SKIP_RETURN_CODE never gets a chance to apply. That is not a hypothetical -- it is
# what the Windows leg reported for tla_cells and mingw_tls_teardown, two failures out of 348.
#
# Nothing is lost. Each of the three is a host-side gate whose toolchain is Linux: genmc needs a
# local LLVM/GenMC build, the TLA cells need a JVM and tla2tools.jar, and the mingw cells need the
# mingw-w64 CROSS compiler, which is by definition not what a native Windows runner has. They run
# on Linux CI and locally; registering them here only manufactures red.
if(NOT WIN32)
add_test(NAME genmc_harnesses
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/verification/genmc/run.sh all)
set_tests_properties(genmc_harnesses PROPERTIES
                     TIMEOUT 1500
                     SKIP_RETURN_CODE 2)

# THE TLC CELLS RUN TOO, and for the same reason the GenMC harnesses do. Each .cfg declares its
# own verdict on line 1 -- `Expected: PASS` or `Expected: VIOLATION` -- and run.sh checks the
# verdict TLC produces against it. Two of the seven cells are CALIBRATIONS that must VIOLATE:
# MCSegmentedArrayBroken and MCMatchForwardingEagerBroken. A spec edit that quietly turns one of
# those into a pass has disabled the calibration, and only a runner that knows the expected
# verdict can see that; a table of results in a README cannot.
#
# --quick drops the deep SegmentedArray cell, which is about a minute on its own against seconds
# for the other six (measured here at -workers 6: 2s, 3s, 3s, 8s, 1s, 2s -- 19s combined). The
# deep cell is run from the full `run.sh` when a minute is available, and its numbers live in
# verification/tla/README.md.
#
# SKIPPED where the toolchain is absent, as with genmc: TLC needs a JVM and tla2tools.jar, and
# neither is a dependency of this project, so run.sh exits 2 and that is mapped to a skip.
add_test(NAME tla_cells
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/verification/tla/run.sh --quick)
set_tests_properties(tla_cells PROPERTIES
                     TIMEOUT 600
                     SKIP_RETURN_CODE 2)

# THE MINGW TEARDOWN CELLS RUN TOO, for the same reason the other two suites do: the decision to
# build the shipped Windows x86-64 artifacts natively with MSVC rests on mingw-w64 corrupting the
# heap at worker-thread exit, and until these cells existed the only account of that was a comment
# in build_all_platforms.sh that nothing checked. Each cell declares CLEAN or CORRUPT and the
# runner checks it, so a toolchain that fixes the defect is noticed instead of assumed. SKIPS
# (exit 2) where the mingw cross-compiler is absent or a Windows binary cannot be executed, which
# is every Linux CI runner.
add_test(NAME mingw_tls_teardown
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/verification/mingw/run.sh)
set_tests_properties(mingw_tls_teardown PROPERTIES
                     TIMEOUT 600
                     SKIP_RETURN_CODE 2)
endif()

# Each bench names itself in its build stamp, as the shipped artifacts do, so a measuring script
# can read the configuration it is about to time from the binary itself.
foreach(_hg_bench bench_cpu_evolve bench_gpu_evolve sampling_cost_smoke cost_matrix)
    if(TARGET ${_hg_bench})
        target_compile_definitions(${_hg_bench} PRIVATE HG_BUILD_VARIANT="${_hg_bench}")
    endif()
endforeach()
