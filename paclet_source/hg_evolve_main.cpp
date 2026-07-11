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
#include <winsock2.h>
#include <ws2tcpip.h>
using socket_t = SOCKET;
#define HG_CLOSESOCK closesocket
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
using socket_t = int;
#define INVALID_SOCKET (-1)
#define HG_CLOSESOCK ::close
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

bool sock_recv_exact(socket_t s, void* buf, size_t n) {
    char* p = static_cast<char*>(buf);
    size_t got = 0;
    while (got < n) {
        int r = recv(s, p + got, static_cast<int>(n - got), 0);
        if (r <= 0) return false;  // peer closed or error
        got += static_cast<size_t>(r);
    }
    return true;
}

bool sock_send_all(socket_t s, const void* buf, size_t n) {
    const char* p = static_cast<const char*>(buf);
    size_t sent = 0;
    while (sent < n) {
        int r = send(s, p + sent, static_cast<int>(n - sent), 0);
        if (r <= 0) return false;
        sent += static_cast<size_t>(r);
    }
    return true;
}

// Socket worker: bind a loopback TCP port, publish it (to `portfile` if given,
// else "HGPORT <port>\n" on stdout), accept one connection, and serve the same
// length-prefixed frames over the socket. Used where the front end cannot carry
// binary over a process pipe (Wolfram's StartProcess stdin drops BinaryWrite and
// truncates WriteString at NUL, and does not surface a running child's stdout)
// but can speak a TCP socket (SocketConnect + BinaryWrite/SocketReadMessage are
// NUL-safe). The port file is the race-free channel for Wolfram: it polls the
// file for the OS-assigned port, then connects. CUDA context and warm caches
// persist across jobs, as with --serve.
int run_serve_socket(const HostBridge& host, const char* portfile) {
#if defined(_WIN32)
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        std::fprintf(stderr, "HGEvolve: WSAStartup failed\n");
        return 1;
    }
#endif
    socket_t listener = socket(AF_INET, SOCK_STREAM, 0);
    if (listener == INVALID_SOCKET) { std::fprintf(stderr, "HGEvolve: socket() failed\n"); return 1; }

    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;  // OS-assigned free port
    if (bind(listener, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        std::fprintf(stderr, "HGEvolve: bind() failed\n");
        HG_CLOSESOCK(listener);
        return 1;
    }
    socklen_t alen = sizeof(addr);
    getsockname(listener, reinterpret_cast<sockaddr*>(&addr), &alen);
    unsigned port = ntohs(addr.sin_port);
    if (portfile) {
        // Write to a temp file, then atomically rename into place so the parent
        // never sees a half-written port.
        std::string tmp = std::string(portfile) + ".tmp";
        FILE* f = std::fopen(tmp.c_str(), "w");
        if (f) { std::fprintf(f, "%u\n", port); std::fclose(f); std::rename(tmp.c_str(), portfile); }
    } else {
        std::printf("HGPORT %u\n", port);
        std::fflush(stdout);
    }

    if (listen(listener, 1) != 0) { std::fprintf(stderr, "HGEvolve: listen() failed\n"); HG_CLOSESOCK(listener); return 1; }
    socket_t conn = accept(listener, nullptr, nullptr);
    if (conn == INVALID_SOCKET) { std::fprintf(stderr, "HGEvolve: accept() failed\n"); HG_CLOSESOCK(listener); return 1; }

    uint8_t lenbuf[8];
    std::vector<uint8_t> job;
    for (;;) {
        if (!sock_recv_exact(conn, lenbuf, 8)) break;  // client closed
        uint64_t len = 0;
        for (int i = 0; i < 8; ++i) len |= static_cast<uint64_t>(lenbuf[i]) << (8 * i);
        job.resize(static_cast<size_t>(len));
        if (len && !sock_recv_exact(conn, job.data(), static_cast<size_t>(len))) break;
        std::vector<uint8_t> out;
        try {
            out = run_rewriting_core(job, host);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "HGEvolve job error: %s\n", e.what());
            out.clear();  // zero-length reply frame = this job errored
        }
        uint8_t hdr[8];
        uint64_t olen = out.size();
        for (int i = 0; i < 8; ++i) hdr[i] = static_cast<uint8_t>(olen >> (8 * i));
        if (!sock_send_all(conn, hdr, 8)) break;
        if (olen && !sock_send_all(conn, out.data(), out.size())) break;
    }
    HG_CLOSESOCK(conn);
    HG_CLOSESOCK(listener);
#if defined(_WIN32)
    WSACleanup();
#endif
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
#if defined(_WIN32)
    // stdin/stdout carry raw WXF bytes; keep them out of text (CRLF) mode.
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    bool serve = false, serve_socket = false;
    const char* socket_portfile = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--serve") == 0) serve = true;
        else if (std::strcmp(argv[i], "--serve-socket") == 0) {
            serve_socket = true;
            // Optional next arg: a path to write the OS-assigned port into.
            if (i + 1 < argc && argv[i + 1][0] != '-') socket_portfile = argv[++i];
        }
    }

    HostBridge host;
    host.progress = [](const std::string& m) {
        std::fputs(m.c_str(), stderr);
        std::fputc('\n', stderr);
        std::fflush(stderr);
    };
    // No abort_query: the parent aborts by killing this process.

    if (serve_socket) return run_serve_socket(host, socket_portfile);
    if (serve) return run_serve(host);
    return run_one_shot(host);
}
