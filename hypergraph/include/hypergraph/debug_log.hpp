#pragma once
#include "hgcommon/namespace.hpp"

// DEBUG_LOG: the engine's trace lines, compiled in only with ENABLE_DEBUG_OUTPUT. Each line goes
// through the engine's one diagnostic channel, hgcommon::diag_printf (hgcommon/diag.hpp): stderr,
// or the sink a host installs (the hg_evolve worker's per-thread rings). Off, the macro is a
// no-op and names nothing, so translation units that include this header for it pull in neither
// the channel nor <cstdarg>; a model checker that compiles engine sources without job_system
// links (verification/genmc/README.md).
#ifdef ENABLE_DEBUG_OUTPUT
#include "hgcommon/diag.hpp"
    #define DEBUG_LOG(fmt, ...) ::hgcommon::diag_printf("[DEBUG] " fmt "\n", ##__VA_ARGS__)
#else
    #define DEBUG_LOG(fmt, ...) ((void)0)
#endif
