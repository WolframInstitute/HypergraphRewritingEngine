#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <functional>

#include "wxf.hpp"   // wxf::Pieces

// The bridge through which the host-agnostic evolution core reports progress.
// The callback may be empty. There is no cooperative abort: the standalone
// binary is aborted by the parent killing the process.
//
//   - Standalone binary: progress -> stderr.
//   - LibraryLink fallback: progress empty (no in-process progress).
struct HostBridge {
    std::function<void(const std::string&)> progress;
};

// A GPU engine binary that cannot use its device refuses every job with a message that starts
// with this prefix and continues with the reason (hg_gpu::DeviceProbe::reason). A worker sends
// the message as an error frame; the one-shot mode writes it to stderr and exits with
// kNoUsableDeviceExit. The paclet kernel matches the same prefix (hgNoDevicePrefix in
// HypergraphRewriting.wl) and runs the job on the CPU.
inline constexpr char kNoUsableDevicePrefix[] = "no usable GPU: ";
inline constexpr int kNoUsableDeviceExit = 3;

// Parse a WXF input association, run the multiway evolution, and serialize the
// result association to WXF. Input and output byte streams are identical to the
// LibraryLink performRewriting contract. Throws std::exception on error.
std::vector<uint8_t> run_rewriting_core(const std::vector<uint8_t>& wxf_bytes,
                                        const HostBridge& host);
// The same reply as consecutive byte runs (wxf::Pieces): each large section is a run of its
// own, written by the transport without being joined first.
HG_NAMESPACE::wxf::Pieces run_rewriting_core_pieces(const std::vector<uint8_t>& wxf_bytes,
                                                    const HostBridge& host);

// Returns a finished job's memory in a process that outlives the job: hg_evolve --serve, the
// paclet's library in the Wolfram kernel, and the Python module. A finished job's arena blocks
// sit in the process-wide pool, mapped and resident, until an arena takes them again
// (hypergraph/arena.hpp); this unmaps the pooled blocks beyond `keep_pooled_bytes`. hg_evolve
// --serve keeps kPooledBytesKeptBetweenJobs (64 MB) for its next job. The Wolfram kernel and a
// Python process keep none, since their memory and address-space limits also cover the host's
// own work (Python, 6-step job then three jobs filling a 64 MB "MemoryLimit": resident set 41 ->
// 79 MB without this call, 30.6 -> 30.8 MB with it). Unmapping needs the pool quiescent: the
// job's engine threads have exited, or wait idle for a held session's next verb. The exited
// threads' job-slot pools keep chunks for as many jobs as each thread ever held at once, and are
// freed (after the A3 job at 32 workers, the idle worker's resident set 101 MB -> 80 MB). glibc
// keeps what those threads freed in their malloc arenas until malloc_trim returns it.
void release_job_memory(size_t keep_pooled_bytes);
