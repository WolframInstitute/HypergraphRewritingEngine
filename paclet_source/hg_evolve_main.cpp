// Standalone process-isolation front end for the multiway engine.
//
// Two modes:
//   one-shot (default): read a WXF job from stdin, run, write the WXF result to
//     stdout. Progress on stderr. Exit 0 on success, 1 on a caught exception.
//   worker  (--serve):  stay alive and process a stream of length-prefixed jobs.
//     Each request/response frame is [8-byte little-endian length][payload]. A
//     zero-length response frame signals that job errored (detail on stderr).
//     The loop ends on stdin EOF. This amortises expensive per-process setup —
//     for the GPU backend, the CUDA context is created once and reused across
//     jobs instead of ~700 ms per invocation.
//
// hg_evolve_gpu probes its device at startup (hg_gpu::probe_device). Without a usable device it
// refuses every job with kNoUsableDevicePrefix + the reason (hg_core.hpp): an error frame in the
// worker modes, a stderr line and exit code kNoUsableDeviceExit in the one-shot mode.
//
// `--version` prints the build stamp (hgcommon/build_stamp.hpp) and exits, ahead of any transport
// setup, so it answers on a machine with no GPU and no peer.
//
// Abort is a process kill by the parent (no cooperative abort). Compiled with
// -DHG_STANDALONE_BINARY, so it links no Wolfram SDK.

#include "hg_core.hpp"
#include "frame_writer.hpp"
#include "hgcommon/diag.hpp"
#include "hgcommon/build_stamp.hpp"
#include "hgcommon/affinity.hpp"
#include "hgcommon/tuning.hpp"
#include "hgcommon/memory_limit.hpp"
#include "hypergraph/arena.hpp"
#include "hypergraph/parallel_evolution.hpp"   // derived_causal_set_capacity
#include "job_system/job_pool.hpp"
#ifdef HG_GPU_BACKEND
#include "hg_gpu/device_probe.hpp"
#include "hg_gpu/evolve.hpp"   // lane_backlog_of
#endif

namespace HG_NAMESPACE { namespace ffi { extern const char kBuildStamp[]; } }  // hypergraph_ffi.cpp
namespace hgffi = HG_NAMESPACE::ffi;

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <exception>

#if defined(_WIN32)
#include <io.h>
#include <fcntl.h>
#else
#include <unistd.h>
#if defined(__GLIBC__)
#include <malloc.h>
#endif
#endif

namespace {

// Ceiling on a single job frame. The length prefix is 8 bytes read straight off the wire, so
// without a bound a corrupt or truncated header sizes an allocation from garbage -- all-0xFF
// reads as ~1.8e19 -- and the resize throws from outside the per-job try, terminating a
// worker the client is still talking to. 1 GiB is far above any real job and far below a
// length that cannot be allocated.
constexpr uint64_t kMaxJobBytes = 1ull << 30;

// THE REPLY STREAM. The parent reads replies from fd 1, and libraries the engine links write
// diagnostics to the process's standard output: the CUDA runtime prints device printf there. So
// take_reply_stream() keeps a duplicate of fd 1 for the replies and points fd 1 at stderr; every
// reply byte goes through reply().
FILE* g_reply = nullptr;

FILE* reply() { return g_reply ? g_reply : stdout; }

void take_reply_stream() {
    std::fflush(stdout);
#if defined(_WIN32)
    const int fd = _dup(_fileno(stdout));
    if (fd < 0) return;
    _setmode(fd, _O_BINARY);
    g_reply = _fdopen(fd, "wb");
    if (g_reply) _dup2(_fileno(stderr), _fileno(stdout));
#else
    const int fd = ::dup(1);
    if (fd < 0) return;
    g_reply = ::fdopen(fd, "wb");
    if (g_reply) ::dup2(2, 1);
#endif
}

// Empty, or kNoUsableDevicePrefix + reason when this is the GPU binary and its device probe failed
// at startup. When set, every job is refused with this message.
std::string g_no_device;

// Read exactly n bytes from stdin into out. Returns false on EOF before any
// byte of a new frame (clean end) or on a short read mid-frame (caller decides).
bool read_exact(size_t n, std::vector<uint8_t>& out) {
    out.resize(n);
    size_t got = 0;
    while (got < n) {
        size_t r = std::fread(out.data() + got, 1, n - got, stdin);
        if (r == 0) return false;
        got += r;
    }
    return true;
}

int run_one_shot(const HostBridge& host) {
    std::vector<uint8_t> input;
    {
        unsigned char buf[65536];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof(buf), stdin)) > 0) {
            input.insert(input.end(), buf, buf + n);
        }
    }
    if (!g_no_device.empty()) {
        std::fprintf(stderr, "%s\n", g_no_device.c_str());
        return kNoUsableDeviceExit;
    }
    try {
        const HG_NAMESPACE::wxf::Pieces out = run_rewriting_core_pieces(input, host);
        for (const auto& part : out.parts)
            if (!part.empty()) std::fwrite(part.data(), 1, part.size(), reply());
        std::fflush(reply());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "HGEvolve fatal: %s\n", e.what());
        return 1;
    }
}

// A finished job's arena blocks sit in the process-wide pool, mapped and resident, until an
// arena takes them again (hypergraph/arena.hpp). A worker stays alive between jobs, so after each
// job it unmaps the pooled blocks beyond 64 MB, which needs the pool quiescent: the job's engine
// threads have exited, or wait idle for a held session's next verb. The exited threads' job-slot
// pools keep chunks for as many jobs as each thread ever held at once, and are freed (after the
// A3 job at 32 workers, the idle worker's resident set 101 MB -> 80 MB). glibc keeps what those
// threads freed in their malloc arenas until malloc_trim returns it.
void release_job_memory() {
    hypergraph::arena_trim_block_pool(hypergraph::kPooledBytesKeptBetweenJobs);
    HG_NAMESPACE::jobs::JobSlotPool::trim_retired_pools();
#if defined(__GLIBC__)
    malloc_trim(0);
#endif
}

// Reads and drops n bytes of stdin in pieces; false on EOF first.
bool skip_exact(uint64_t n) {
    char buf[65536];
    while (n > 0) {
        const size_t want = n < sizeof buf ? static_cast<size_t>(n) : sizeof buf;
        const size_t r = std::fread(buf, 1, want, stdin);
        if (r == 0) return false;
        n -= r;
    }
    return true;
}

// The worker: length-prefixed jobs on stdin until EOF, every frame written by one FrameWriter
// (frame_writer.hpp). Engine diagnostics (hgcommon::diag_printf) and progress go to the reporting
// thread's ring; main has pointed fd 1 and fd 2 at the capture pipe.
// In the test workers (HG_WORKER_TEST_HOOKS, paclet_source/CMakeLists.txt) HG_DEBUG_OUTPUT_BYTES=n
// writes n bytes to stdout (the C runtime stream device printf reaches) and n bytes to stderr at
// the start of every job, which tests the capture.
int run_serve(const HostBridge& host_in) {
    hgframe::FrameWriter writer(reply());
    hgcommon::set_diag_sink([](const char* text, std::size_t length) {
        hgframe::post_message(hgframe::MessageKind::Log, text, length);
    });
    HostBridge host = host_in;
    host.progress = [](const std::string& m) {
        hgframe::post_message(hgframe::MessageKind::Progress, m.data(), m.size());
    };
#if defined(HG_WORKER_TEST_HOOKS)
    const char* debug_env = std::getenv("HG_DEBUG_OUTPUT_BYTES");
    const size_t debug_bytes = debug_env ? std::strtoull(debug_env, nullptr, 10) : 0;
#endif
    std::vector<uint8_t> lenbuf, job;
    for (;;) {
        if (!read_exact(8, lenbuf)) break;  // clean EOF between frames
        uint64_t len = 0;
        for (int i = 0; i < 8; ++i) len |= static_cast<uint64_t>(lenbuf[i]) << (8 * i);
        writer.begin_job();
        // A frame over the cap is read and dropped, so the stream stays in step, and refused.
        if (len > kMaxJobBytes) {
            if (!skip_exact(len)) break;
            writer.write_error("job frame of " + std::to_string(len) + " bytes exceeds the " +
                               std::to_string(kMaxJobBytes) + "-byte cap");
            continue;
        }
        if (!read_exact(static_cast<size_t>(len), job)) {
            std::fprintf(stderr, "HGEvolve: truncated job frame\n");
            return 1;
        }
#if defined(HG_WORKER_TEST_HOOKS)
        if (debug_bytes) {
            const std::string out(99, 'o'), err(99, 'x');
            for (size_t done = 0; done < debug_bytes; done += 100) {
                std::fprintf(stdout, "%s\n", out.c_str());
                std::fprintf(stderr, "%s\n", err.c_str());
            }
        }
#endif
        if (!g_no_device.empty()) {
            writer.write_error(g_no_device);
            continue;
        }
        try {
            const HG_NAMESPACE::wxf::Pieces out = run_rewriting_core_pieces(job, host);
            writer.write_reply(out);
        } catch (const std::exception& e) {
            writer.write_error(e.what());
        }
        release_job_memory();
    }
    hgcommon::set_diag_sink(nullptr);
    return 0;
}

}  // namespace

// The command line this binary actually accepts. Printed for --help and alongside the error for
// an unrecognised argument, so a caller who guessed wrong is told what the choices are rather
// than left to infer them from a WXF parse failure.
static const char* const kUsage =
    "usage: hg_evolve [--serve] [--version] [--print-tuning] [--help]\n"
    "\n"
    "  (no arguments)          one-shot: read one WXF job from stdin, write the WXF result to\n"
    "                          stdout. Progress on stderr.\n"
    "  --serve                 worker: process a stream of length-prefixed WXF jobs on stdin,\n"
    "                          each framed [8-byte little-endian length][payload], until EOF.\n"
    "  --version               print the build stamp (commit and variant) and exit.\n"
    "  --print-tuning          print the host (and device) probe and the values derived from\n"
    "                          it, with the source of each, and exit.\n"
    "  --help, -h              print this and exit.\n";

namespace {

std::string mib(uint64_t bytes) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.1f MB", static_cast<double>(bytes) / (1024.0 * 1024.0));
    return buf;
}

// The probes and the tuning values derived from them. Every line names the value's source.
void print_tuning() {
    const hgcommon::HostTopology& t = hgcommon::host_topology();
    std::printf("host.model              %s\n", t.model.empty() ? "(unknown)" : t.model.c_str());
    std::printf("host.logical_cpus       %u\n", t.logical_cpus);
    std::printf("host.physical_cores     %u\n", t.physical_cores);
    std::printf("host.smt_width          %u\n", t.smt_width);
    std::printf("host.core_classes       %u\n", t.core_classes);
    std::printf("host.performance_cores  %u\n", t.performance_cores);
    std::printf("host.llc_domains        %zu%s\n", t.llc_bytes.size(),
                t.cache_known ? "" : " (cache topology not published)");
    for (size_t d = 0; d < t.llc_bytes.size(); ++d) {
        unsigned n = 0;
        for (const auto& e : t.cpus) n += e.llc == d ? 1u : 0u;
        std::printf("host.llc[%zu]             %s, %u logical CPUs\n", d, mib(t.llc_bytes[d]).c_str(), n);
    }
    std::printf("host.l2_per_core        %s\n", mib(t.l2_bytes).c_str());
    std::printf("host.memory_total       %s\n", mib(t.total_bytes).c_str());
    std::printf("host.memory_available   %s\n", mib(t.available_bytes).c_str());
    std::printf("host.huge_page          %s\n", mib(t.huge_page_bytes).c_str());
    std::printf("host.virtualised        %s\n", t.virtualised ? "yes (WSL: the layout is the hypervisor's)" : "no");
    std::string classes;
    for (const auto& e : t.cpus) classes += e.core_class == 0 ? 'P' : 'E';
    if (t.core_classes > 1) std::printf("host.cpu_classes        %s\n", classes.c_str());

    std::printf("workers                 %u  (%s)\n", hgcommon::default_worker_count(HG_NAMESPACE::engine::MAX_ARENA_WORKERS),
                hgcommon::tuning_env().get(hgcommon::Tune::Threads)
                    ? "HG_TUNE_THREADS"
                    : "logical CPUs this process may use; \"Threads\" per job");
    {
        hgcommon::TuneSource src;
        const int64_t cap = hgcommon::tuned(hgcommon::TuningOverrides{},
                                            hgcommon::Tune::CausalSetCapacity, [&] {
            return HG_NAMESPACE::engine::ParallelEvolutionEngine::derived_causal_set_capacity(
                0, 0, 1, t.available_bytes);
        }, &src);
        std::printf("causal_set_capacity     %lld per shard  (%s)\n", static_cast<long long>(cap),
                    src == hgcommon::TuneSource::Environment
                        ? "HG_TUNE_CAUSAL_SET_CAPACITY"
                        : "run without a step cap: 8192 above 4 GB available, else 1024; a step "
                          "cap sizes it from the key bound");
    }
    if (const auto fixed = hgcommon::tuning_env().get(hgcommon::Tune::ExpandChunk))
        std::printf("expand_chunk            %lld matches  (HG_TUNE_EXPAND_CHUNK)\n",
                    static_cast<long long>(*fixed));
    else
        std::printf("expand_chunk            ceil(matches / workers), 4 to 64; %zu for a 64-match "
                    "state at the default workers\n",
                    HG_NAMESPACE::engine::ParallelEvolutionEngine::derived_expand_chunk(
                        64, hgcommon::default_worker_count(HG_NAMESPACE::engine::MAX_ARENA_WORKERS)));
    {
        hgcommon::TuneSource src;
        const uint64_t limit = hgcommon::resolved_memory_limit(hgcommon::TuningOverrides{}, &src);
        std::printf("memory_limit            %s  (%s)\n",
                    limit == hgcommon::kNoMemoryLimit ? "none" : mib(limit).c_str(),
                    src == hgcommon::TuneSource::Environment
                        ? "HG_TUNE_MEMORY_LIMIT"
                        : "half of memory_total; \"Tuning\" -> MemoryLimit per job, at most "
                          "memory_total");
        std::printf("memory_limit.os         %s  (the arenas' check applies as well)\n",
                    hgcommon::memory_limit_backend_name(hgcommon::memory_limit_backend()));
    }
    const std::vector<unsigned> placed = hgcommon::default_worker_cpus();
    if (placed.empty()) {
        std::printf("worker_placement        operating system  (default_worker_cpus: %s)\n",
                    !hgcommon::affinity_supported() ? "no thread binding on this platform"
                    : !t.cache_known                ? "cache topology not published"
                                                    : "one cache domain");
    } else {
        std::string list;
        for (unsigned c : placed) list += (list.empty() ? "" : ",") + std::to_string(c);
        std::printf("worker_placement        %s  (default_worker_cpus, domain-major)\n", list.c_str());
    }
#ifdef HG_GPU_BACKEND
    namespace hgg = HG_NAMESPACE::gpu;
    const hgg::DeviceProbe& p = hgg::probe_device();
    std::printf("gpu.usable              %s\n", p.usable ? "yes" : ("no: " + p.reason).c_str());
    std::printf("gpu.name                %s\n", p.name.c_str());
    std::printf("gpu.compute_capability  %d.%d\n", p.major, p.minor);
    std::printf("gpu.driver_version      %d.%d\n", p.driver_version / 1000, (p.driver_version % 1000) / 10);
    std::printf("gpu.runtime_version     %d.%d\n", p.runtime_version / 1000, (p.runtime_version % 1000) / 10);
    std::printf("gpu.sm_count            %d\n", p.sm_count);
    std::printf("gpu.max_threads_per_sm  %d\n", p.max_threads_per_sm);
    std::printf("gpu.max_blocks_per_sm   %d\n", p.max_blocks_per_sm);
    std::printf("gpu.regs_per_sm         %d\n", p.regs_per_sm);
    std::printf("gpu.shared_per_sm       %llu B\n", static_cast<unsigned long long>(p.shared_per_sm));
    std::printf("gpu.shared_per_block    %llu B (opt-in)\n",
                static_cast<unsigned long long>(p.shared_per_block_optin));
    std::printf("gpu.l2                  %s\n", mib(p.l2_bytes).c_str());
    std::printf("gpu.persisting_l2_max   %s\n", mib(p.persisting_l2_max).c_str());
    std::printf("gpu.memory_total        %s\n", mib(p.total_bytes).c_str());
    if (p.usable) {
        std::printf("gpu.blocks_per_sm       %u  (k_persistent_evolve occupancy at 32 threads)\n",
                    hgg::persistent_blocks_per_sm());
        std::printf("gpu.plateau_per_sm      %u  (grid sweep of cc %d.%d)\n",
                    hgg::persistent_blocks_per_sm_plateau(p.major, p.minor), p.major, p.minor);
        std::printf("gpu.persistent_grid     %u  (%s)\n", hgg::default_persistent_grid(),
                    hgcommon::tuning_env().get(hgcommon::Tune::GpuBlocks)
                        ? "HG_TUNE_GPU_BLOCKS / HG_GPU_PERSISTENT_BLOCKS"
                        : "sm_count x min(blocks_per_sm, plateau_per_sm); \"Tuning\" -> GPUBlocks per job");
        {
            const hgg::EngineConfig cfg{};
            std::printf("gpu.lane_backlog        %u  (%s)\n", hgg::lane_backlog_of(cfg),
                        hgcommon::tuning_env().get(hgcommon::Tune::GpuBacklog)
                            ? "HG_TUNE_GPU_BACKLOG"
                            : "max(8, persistent_grid / 16); \"Tuning\" -> GPUBacklog per job");
        }
        std::printf("gpu.l2_persist_limit    %s  (%s)\n", mib(hgg::l2_persist_limit_of()).c_str(),
                    hgcommon::tuning_env().get(hgcommon::Tune::GpuL2Persist)
                        ? "HG_TUNE_GPU_L2_PERSIST"
                        : "persisting_l2_max on cc >= 8.0: the state claim map is kept in L2 "
                          "when it is at most this; \"Tuning\" -> GPUL2Persist per job");
        std::printf("gpu.grid_stride_cap     %u blocks of 256  (resident_threads / 256)\n",
                    hgg::grid_stride_blocks(~uint64_t{0} >> 1, 256));
        std::printf("gpu.resident_threads    %llu  (sm_count x max_threads_per_sm)\n",
                    static_cast<unsigned long long>(hgg::device_resident_threads()));
        const uint64_t cap = hgg::resolved_memory_cap(0);
        std::printf("gpu.memory_cap          %s  (%s)\n", mib(cap).c_str(),
                    hgcommon::tuning_env().get(hgcommon::Tune::GpuMemoryLimit)
                        ? "HG_TUNE_GPU_MEMORY_LIMIT"
                        : "min(90% of memory_total, free - 256 MB); \"Tuning\" -> GPUMemoryLimit per job");
        std::printf("gpu.initial_budget      %s  (memory_cap / 4: the first engine of a run)\n",
                    mib(cap / 4).c_str());
    }
#endif
}

}  // namespace

int main(int argc, char** argv) {
#if defined(__GLIBC__)
    // Fixed thresholds instead of glibc's sliding ones. glibc raises the mmap threshold after a
    // large block is freed (to 32 MB at most) and the trim threshold with it (to twice that), and
    // a worker thread's malloc arena gives its top back only above the trim threshold, which
    // malloc_trim does not lower. Measured after the A3 job through --serve at 32 workers: 586
    // MB free in the threads' arenas stayed resident.
    mallopt(M_MMAP_THRESHOLD, 1 << 20);
    mallopt(M_TRIM_THRESHOLD, 1 << 20);
    // Two malloc arenas, not up to eight per core. glibc gives a thread a new arena when the ones
    // it tries are locked, so the count follows the schedule, and each reserves 64 MB of address
    // space. The engine allocates its work from its own arenas (hypergraph/arena.hpp), so malloc
    // is off the hot path. Measured on wpp depth 8 under quotient exploration at 32 workers:
    // VmPeak 6024 MB -> 4039 MB, VmHWM unchanged (2574 -> 2561 MB median of 6).
    mallopt(M_ARENA_MAX, 2);
#endif
    // This thread reads jobs and builds replies, and a reply can be most of a job's time (the
    // reconstructed counts and relations). The engine's workers exempt themselves; this is the
    // one other thread that computes.
    hgcommon::exempt_this_thread_from_power_throttling();
#if defined(_WIN32)
    // stdin/stdout carry raw WXF bytes; keep them out of text (CRLF) mode.
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    bool serve = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--version") == 0) {
            // Prints the same literal a scanner finds in the file, so the runnable answer and the
            // scannable one cannot disagree. On the ONE platform a release host can execute, this
            // is how the stamp itself is ground-truthed against the scanner.
            std::fputs(hgffi::kBuildStamp, stdout);
            std::fputc('\n', stdout);
            return 0;
        }
        if (std::strcmp(argv[i], "--print-tuning") == 0) {
            print_tuning();
            return 0;
        }
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            std::fputs(kUsage, stdout);
            return 0;
        }
        if (std::strcmp(argv[i], "--serve") == 0) serve = true;
        else {
            // AN ARGUMENT THIS BINARY DOES NOT KNOW IS AN ERROR, NOT A NO-OP. Every mode here
            // reads WXF from stdin, so an unrecognised argument that falls through produces
            // "Parse error: Unexpected end of WXF data" from an empty stdin -- a message about
            // the transport, for a mistake in the command line. A mistyped `--serve` would
            // silently run one-shot instead of serving.
            std::fprintf(stderr, "HGEvolve fatal: unknown argument '%s'\n\n", argv[i]);
            std::fputs(kUsage, stderr);
            return 2;
        }
    }

    take_reply_stream();
    // The worker's start-up messages (the memory limit below, the device probe) go through the
    // capture; the paclet reports them with the first call's HGEvolve::enginelog (hgWorkerTry).
    if (serve) hgframe::start_output_capture();

    // THE HOST MEMORY LIMIT (hgcommon/memory_limit.hpp), installed before any job, so an
    // allocation beyond it fails in this process and the run returns a partial result. Each job
    // sets it to its own "MemoryLimit" while it runs (hypergraph::JobMemoryLimit).
    if (hgcommon::memory_limit_backend() != hgcommon::MemoryLimitBackend::None &&
        !hgcommon::install_process_memory_limit(
            hgcommon::resolved_memory_limit(hgcommon::TuningOverrides{}))) {
        std::fprintf(stderr, "HGEvolve: the %s memory limit was not installed; the engine's "
                     "arenas apply the limit alone\n",
                     hgcommon::memory_limit_backend_name(hgcommon::memory_limit_backend()));
    }

#ifdef HG_GPU_BACKEND
    // Checked before any job, so a machine without a usable device gets one message naming the
    // reason instead of a CUDA error from the first kernel call.
    {
        const auto& probe = HG_NAMESPACE::gpu::probe_device();
        if (!probe.usable) g_no_device = std::string(kNoUsableDevicePrefix) + probe.reason;
    }
#endif

    HostBridge host;
    host.progress = [](const std::string& m) {
        std::fputs(m.c_str(), stderr);
        std::fputc('\n', stderr);
        std::fflush(stderr);
    };
    // No abort_query: the parent aborts by killing this process.

    if (serve) return run_serve(host);
    return run_one_shot(host);
}
