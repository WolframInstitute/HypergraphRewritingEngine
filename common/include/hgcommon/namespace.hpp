#pragma once
//
// THE ONE NAMESPACE ROOT, and the knob that renames it.
//
// Every namespace this project defines sits under a single root so that linking the engine into
// someone else's program adds exactly ONE name to the global namespace. The engine used to
// contribute nine -- hgcommon, hypergraph, hg_gpu, hgffi, hgmarshal, job_system, lockfree, wxf,
// hg -- each of which is a name a host program cannot then use.
//
// THE ROOT IS A MACRO because a library cannot know it has not collided. A program that already
// has its own `hg` can build this one with -DHG_NAMESPACE=whatever and every symbol moves,
// without editing a line of it. That is only possible if the root is written as a macro at every
// point a namespace is opened, which is why the declarations read
//
//     namespace HG_NAMESPACE { namespace common { ... } }
//
// rather than naming the root directly.
//
// THE SHORT ALIASES ARE THE MIGRATION SEAM. `hgcommon` and its siblings are declared here as
// namespace aliases so the thousands of existing call sites keep resolving while the
// declarations move one subsystem at a time. They are the one thing that still occupies global
// scope, and they are what a later pass removes -- an alias is cheap to delete, a scattered
// call site is not.

#ifndef HG_NAMESPACE
#define HG_NAMESPACE hg
#endif

// Declared here rather than at each subsystem's first header, so the alias below always has
// something to bind to no matter which header a translation unit reaches first.
namespace HG_NAMESPACE {
namespace common {}
namespace engine {}
namespace gpu {}
namespace ffi {}
namespace marshal {}
namespace jobs {}
namespace deque {}
namespace wxf {}
}  // namespace HG_NAMESPACE

// The subsystem's own short name, kept so existing call sites resolve. `wxf` maps to itself:
// the sub-namespace is already the name callers use, and only its ENCLOSING scope moved.
namespace hgcommon   = HG_NAMESPACE::common;
namespace hypergraph = HG_NAMESPACE::engine;
namespace hg_gpu     = HG_NAMESPACE::gpu;
namespace hgffi      = HG_NAMESPACE::ffi;
namespace hgmarshal  = HG_NAMESPACE::marshal;
namespace job_system = HG_NAMESPACE::jobs;
namespace lockfree   = HG_NAMESPACE::deque;
namespace wxf        = HG_NAMESPACE::wxf;
