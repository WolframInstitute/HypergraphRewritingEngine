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
// Abort is a process kill by the parent (no cooperative abort). Compiled with
// -DHG_STANDALONE_BINARY, so it links no Wolfram SDK.

#include "hg_core.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <exception>

#if defined(_WIN32)
#include <io.h>
#include <fcntl.h>
#endif

namespace {

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

void write_frame(const std::vector<uint8_t>& payload) {
    uint64_t len = static_cast<uint64_t>(payload.size());
    uint8_t hdr[8];
    for (int i = 0; i < 8; ++i) hdr[i] = static_cast<uint8_t>(len >> (8 * i));
    std::fwrite(hdr, 1, 8, stdout);
    if (len) std::fwrite(payload.data(), 1, len, stdout);
    std::fflush(stdout);
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
    try {
        std::vector<uint8_t> out = run_rewriting_core(input, host);
        if (!out.empty()) std::fwrite(out.data(), 1, out.size(), stdout);
        std::fflush(stdout);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "HGEvolve fatal: %s\n", e.what());
        return 1;
    }
}

int run_serve(const HostBridge& host) {
    std::vector<uint8_t> lenbuf, job;
    for (;;) {
        if (!read_exact(8, lenbuf)) break;  // clean EOF between frames
        uint64_t len = 0;
        for (int i = 0; i < 8; ++i) len |= static_cast<uint64_t>(lenbuf[i]) << (8 * i);
        if (!read_exact(len, job)) {
            std::fprintf(stderr, "HGEvolve: truncated job frame\n");
            return 1;
        }
        try {
            std::vector<uint8_t> out = run_rewriting_core(job, host);
            write_frame(out);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "HGEvolve job error: %s\n", e.what());
            write_frame({});  // zero-length frame = this job errored
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
#if defined(_WIN32)
    // stdin/stdout carry raw WXF bytes; keep them out of text (CRLF) mode.
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    bool serve = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--serve") == 0) serve = true;
    }

    HostBridge host;
    host.progress = [](const std::string& m) {
        std::fputs(m.c_str(), stderr);
        std::fputc('\n', stderr);
        std::fflush(stderr);
    };
    // No abort_query: the parent aborts by killing this process.

    return serve ? run_serve(host) : run_one_shot(host);
}
