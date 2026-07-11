// Standalone process-isolation front end for the multiway engine.
//
// Reads a WXF input association from stdin, runs run_rewriting_core, writes the
// WXF result association to stdout. Progress lines go to stderr. Abort is a
// process kill by the parent (no cooperative abort). A caught exception prints
// "HGEvolve fatal: ..." to stderr and exits 1.
//
// Compiled with -DHG_STANDALONE_BINARY, so it links no Wolfram SDK.

#include "hg_core.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>
#include <string>
#include <exception>

#if defined(_WIN32)
#include <io.h>
#include <fcntl.h>
#endif

int main() {
#if defined(_WIN32)
    // stdin/stdout carry raw WXF bytes; keep them out of text (CRLF) mode.
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    std::vector<uint8_t> input;
    {
        unsigned char buf[65536];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof(buf), stdin)) > 0) {
            input.insert(input.end(), buf, buf + n);
        }
    }

    HostBridge host;
    host.progress = [](const std::string& m) {
        std::fputs(m.c_str(), stderr);
        std::fputc('\n', stderr);
        std::fflush(stderr);
    };
    // No abort_query: the parent aborts by killing this process.

    try {
        std::vector<uint8_t> out = run_rewriting_core(input, host);
        if (!out.empty()) {
            std::fwrite(out.data(), 1, out.size(), stdout);
        }
        std::fflush(stdout);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "HGEvolve fatal: %s\n", e.what());
        return 1;
    }
}
