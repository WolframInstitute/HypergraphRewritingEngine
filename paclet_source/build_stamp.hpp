#pragma once
#include "hgcommon/namespace.hpp"
//
// The commit a shipped artifact was built from, written into the artifact as a byte string.
//
// A release ships binaries for six platforms plus two CUDA executables, and the machine that
// assembles them can EXECUTE at most one platform's worth. So `--version` cannot answer "is this
// artifact current?" for the others: an ARM64 macOS dylib on a Linux box is a file, not a program.
// A literal in .rodata can be read from any of them without running anything, which is why the
// stamp is a scannable string and not a function.
//
// FORMAT. This is a contract with tools/dev/artifact_stamp_check.py, which greps for it:
//
//     HGBUILDSTAMP/1 commit=<40-hex-or-unknown> variant=<name> :HGBUILDSTAMP
//
// The trailing sentinel bounds the match, so a scanner reads a fixed field rather than "whatever
// followed until the next NUL" -- a linker is free to place another string immediately after.
//
// COMMIT ONLY, NEVER "dirty". CMake computes HG_BUILD_COMMIT at CONFIGURE time; whether the tree
// was modified is a property of BUILD time, and a configure-time answer to it would be a claim
// the build cannot support. The two questions are checked where each is answerable: the stamp
// says which commit, and sign-off separately requires a clean tree.
//
// HG_BUILD_VARIANT names which artifact this is (paclet-library / hg_evolve / hg_evolve_gpu), so
// one scan over a platform directory can tell which of the three files it is reading.

#ifndef HG_BUILD_COMMIT
#define HG_BUILD_COMMIT "unknown"
#endif
#ifndef HG_BUILD_VARIANT
#define HG_BUILD_VARIANT "unknown"
#endif

namespace HG_NAMESPACE {
namespace ffi {

// External linkage, and referenced from each artifact's entry point (WolframLibrary_initialize
// for the library, main for the executables). An unreferenced object with internal linkage is
// exactly what a linker is entitled to drop, and a stamp that can be dropped proves nothing.
extern const char kBuildStamp[];

}  // namespace ffi
}  // namespace HG_NAMESPACE
