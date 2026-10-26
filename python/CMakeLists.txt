# hgevolve._core: the extension module of the hgevolve package. It compiles the paclet's job
# code (hypergraph_ffi.cpp, paclet_support.cpp) as hg_evolve does and replies in arrays
# (paclet_source/reply_out.hpp, ArrayReply). The module is written into python/hgevolve/, beside
# the package's Python files, so a build tree can be imported with PYTHONPATH=<repo>/python; a
# scikit-build-core build (pyproject.toml) installs it into the wheel instead.

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
if(NOT nanobind_DIR)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE nanobind_ROOT
        RESULT_VARIABLE _hg_nanobind_found)
    if(NOT _hg_nanobind_found EQUAL 0)
        message(FATAL_ERROR "HG_BUILD_PYTHON needs nanobind for ${Python_EXECUTABLE} "
                            "(pip install nanobind)")
    endif()
endif()
find_package(nanobind CONFIG REQUIRED)

# Release builds strip the module's symbols; a profile (callgrind) needs them.
option(HG_PYTHON_KEEP_SYMBOLS "Keep the symbols of the hgevolve extension module" OFF)
set(_hg_nostrip)
if(HG_PYTHON_KEEP_SYMBOLS)
    set(_hg_nostrip NOSTRIP)
endif()

# NOMINSIZE: the engine sources compile at -O3, as in hg_evolve; nanobind would add -Os.
nanobind_add_module(_core NB_STATIC NOMINSIZE ${_hg_nostrip}
    src/module.cpp
    ${CMAKE_SOURCE_DIR}/paclet_source/hypergraph_ffi.cpp
    ${CMAKE_SOURCE_DIR}/paclet_source/paclet_support.cpp
)
target_compile_definitions(_core PRIVATE HG_STANDALONE_BINARY HG_BUILD_VARIANT="python")
target_include_directories(_core PRIVATE
    ${CMAKE_SOURCE_DIR}/paclet_source
    ${CMAKE_SOURCE_DIR}/hypergraph/include
    ${CMAKE_SOURCE_DIR}/job_system/include
    ${CMAKE_SOURCE_DIR}/lockfree_deque/include
)
target_link_libraries(_core PRIVATE hypergraph wxf)
# nanobind is third-party code: its headers are system headers to _core, and its own sources
# build without this tree's warnings-as-errors (-Werror=shadow and the others).
set_target_properties(nanobind-static PROPERTIES SYSTEM ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(nanobind-static PRIVATE -w)
    # The LibraryLink helper functions compile out under HG_STANDALONE_BINARY.
    target_compile_options(_core PRIVATE -Wno-unused-function)
endif()

if(SKBUILD)
    install(TARGETS _core LIBRARY DESTINATION hgevolve COMPONENT python)
else()
    set_target_properties(_core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/hgevolve")
endif()
