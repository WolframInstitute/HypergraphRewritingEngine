# hgevolve for JavaScript: the paclet's job code (hypergraph_ffi.cpp, paclet_support.cpp)
# compiled to WebAssembly with Emscripten, with the embind binding of js/src/binding.cpp. The
# module is an ES module, hgevolve.mjs (no threads) or hgevolve-threads.mjs (-pthread), and its
# .wasm beside it. js/build.sh configures one build tree per variant and copies both into js/dist.
#
# Every object of the build compiles with -fwasm-exceptions (js/build.sh sets it in
# CMAKE_CXX_FLAGS): the engine reports a refused job by throwing, and Emscripten's default build
# aborts on a throw.

if(NOT EMSCRIPTEN)
    return()
endif()

set(_hg_js_threads OFF)
if(CMAKE_CXX_FLAGS MATCHES "-pthread")
    set(_hg_js_threads ON)
endif()

add_executable(hgevolve_js
    src/binding.cpp
    ${CMAKE_SOURCE_DIR}/paclet_source/hypergraph_ffi.cpp
    ${CMAKE_SOURCE_DIR}/paclet_source/paclet_support.cpp
)
target_compile_definitions(hgevolve_js PRIVATE HG_STANDALONE_BINARY HG_BUILD_VARIANT="js")
target_include_directories(hgevolve_js PRIVATE
    ${CMAKE_SOURCE_DIR}/paclet_source
    ${CMAKE_SOURCE_DIR}/hypergraph/include
    ${CMAKE_SOURCE_DIR}/job_system/include
    ${CMAKE_SOURCE_DIR}/lockfree_deque/include
)
target_link_libraries(hgevolve_js PRIVATE hypergraph wxf)
target_compile_options(hgevolve_js PRIVATE -Wno-unused-function)

# The module is a library: it has no main(), and its runtime stays up between calls.
target_link_options(hgevolve_js PRIVATE
    -lembind
    -fwasm-exceptions
    -sMODULARIZE=1
    -sEXPORT_ES6=1
    -sEXPORT_NAME=createHgEvolve
    -sENVIRONMENT=web,worker,node
    -sEXIT_RUNTIME=0
    -sALLOW_MEMORY_GROWTH=1
    -sMAXIMUM_MEMORY=4GB
    -sWASM_BIGINT=1
)
if(_hg_js_threads)
    # The pool holds every thread the engine starts: the evolution's workers and the reply's
    # workers (HostJobs), each the hardware concurrency. A thread created beyond the pool waits for
    # the event loop of the thread that created it, and run() holds that loop, so the pool is made
    # when the module loads. PTHREAD_POOL_SIZE_STRICT=2 makes a thread beyond it an error.
    target_link_options(hgevolve_js PRIVATE
        -pthread
        "-sPTHREAD_POOL_SIZE=Math.min(globalThis.navigator?.hardwareConcurrency ?? 4, 64) * 2 + 2"
        -sPTHREAD_POOL_SIZE_STRICT=2
        -Wno-pthreads-mem-growth
    )
    set_target_properties(hgevolve_js PROPERTIES OUTPUT_NAME "hgevolve-threads" SUFFIX ".mjs")
else()
    set_target_properties(hgevolve_js PROPERTIES OUTPUT_NAME "hgevolve" SUFFIX ".mjs")
endif()
