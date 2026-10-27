#include "frame_writer.hpp"

#include <cstring>
#include <new>

#include "hgcommon/utf8.hpp"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#else
#include <unistd.h>
#endif
#if defined(__linux__)
#include <pthread.h>
#endif

namespace hgframe {

namespace {

// The bytes of `text` that go in one slot: at most kSlotBytes, ending at a character boundary
// unless no boundary is found.
std::size_t slot_part(const char* text, std::size_t length) {
    if (length <= LogRing::kSlotBytes) return length;
    const std::size_t whole = hgcommon::utf8_complete_prefix(text, LogRing::kSlotBytes);
    return whole > 0 ? whole : LogRing::kSlotBytes;
}

}  // namespace

bool LogRing::push(MessageKind kind, const char* text, std::size_t length) {
    uint32_t parts = 0;
    for (std::size_t rest = length, at = 0; parts == 0 || rest > 0; ++parts) {
        if (parts > kSlots) break;  // more slots than the ring has: dropped below
        const std::size_t part = slot_part(text + at, rest);
        at += part;
        rest -= part;
    }
    const uint64_t t = tail.load(std::memory_order_relaxed);
    if (parts > kSlots - (t - head.load(std::memory_order_acquire))) {
        dropped.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    for (uint32_t i = 0; i < parts; ++i) {
        Slot& s = slots[(t + i) % kSlots];
        const std::size_t part = slot_part(text, length);
        s.kind = kind;
        s.parts = static_cast<uint16_t>(i == 0 ? parts : 0);
        s.length = static_cast<uint32_t>(part);
        std::memcpy(s.text, text, part);
        text += part;
        length -= part;
    }
    tail.store(t + parts, std::memory_order_release);
    return true;
}

namespace {

LogRing* make_ring() { return new (std::nothrow) LogRing(); }

// The writer's park word: one domain, one waiter (the writer), and producers that signal it.
hgcommon::ParkGate::Domain g_writer_domain;
hgcommon::ParkGate& writer_gate() {
    static hgcommon::ParkGate* gate = [] {
        auto* g = new hgcommon::ParkGate();
        g->seat(&g_writer_domain, 1, 1, false);
        return g;
    }();
    return *gate;
}

struct RingClaim {
    LogRing* ring = nullptr;
    ~RingClaim() {
        if (ring) ring->owned.store(0, std::memory_order_release);
    }
};
thread_local RingClaim t_claim;

}  // namespace

WorkerRings& worker_rings() {
    static WorkerRings* rings = new WorkerRings(make_ring);
    return *rings;
}

LogRing* this_thread_ring() {
    // A thread that got no ring tries again on a later message: a ring may have been handed on.
    if (!t_claim.ring) t_claim.ring = worker_rings().claim();
    return t_claim.ring;
}

void post_message(MessageKind kind, const char* text, std::size_t length) {
    LogRing* r = this_thread_ring();
    if (!r) worker_rings().count_unringed();
    else r->push(kind, text, length);
    writer_gate().wake_one(0);
}

// The thread starts after every member is constructed.
FrameWriter::FrameWriter(std::FILE* out) : out_(out) { thread_ = std::thread([this] { run(); }); }

FrameWriter::~FrameWriter() {
    stop_.store(true, std::memory_order_seq_cst);
    writer_gate().wake_one(0);
    thread_.join();
}

void FrameWriter::begin_job() { job_seq_.fetch_add(1, std::memory_order_release); }

void FrameWriter::write_header(uint64_t header) {
    uint8_t h[8];
    for (int i = 0; i < 8; ++i) h[i] = static_cast<uint8_t>(header >> (8 * i));
    std::fwrite(h, 1, 8, out_);
}

void FrameWriter::write_frame(uint64_t header, const char* data, std::size_t length) {
    write_header(header);
    if (length) std::fwrite(data, 1, length, out_);
}

// A count report as a log frame, formatted on the stack: the writer allocates nothing.
void FrameWriter::write_report(const char* format, unsigned long long count) {
    char text[200];
    const int n = std::snprintf(text, sizeof text, format, count);
    if (n > 0) write_frame(kLogFrameBit | static_cast<uint64_t>(n), text, static_cast<std::size_t>(n));
}

bool FrameWriter::drain_rings() {
    bool any = false;
    // A new job starts its own count of forwarded process output.
    const uint64_t job = job_seq_.load(std::memory_order_acquire);
    if (job != seen_job_) {
        seen_job_ = job;
        capture_forwarded_ = 0;
    }
    WorkerRings& rings = worker_rings();
    const uint32_t n = rings.size();
    for (uint32_t i = 0; i < n; ++i) {
        LogRing* r = rings.at(i);
        if (!r) continue;
        while (r->pop([&](MessageKind kind, uint32_t parts, auto part) {
            std::size_t length = 0, escaped = 0;
            for (uint32_t p = 0; p < parts; ++p) {
                length += part(p).length;
                escaped += hgcommon::utf8_escaped_length(part(p).text, part(p).length);
            }
            if (kind == MessageKind::Capture) {
                if (capture_forwarded_ + length > kCaptureBytesPerJob) {
                    capture_omitted_ += length;
                    return;
                }
                capture_forwarded_ += length;
            }
            // One frame for the whole message; each slot ends at a character boundary, so the
            // slots are escaped one at a time.
            write_header((kind == MessageKind::Progress ? kProgressFrameBit : kLogFrameBit) |
                         escaped);
            for (uint32_t p = 0; p < parts; ++p)
                hgcommon::utf8_escaped(part(p).text, part(p).length,
                                       [&](const char* s, std::size_t k) { std::fwrite(s, 1, k, out_); });
        }))
            any = true;
        if (const uint64_t lost = r->dropped.exchange(0, std::memory_order_relaxed)) {
            write_report("HGEvolve: %llu diagnostic messages were dropped (a full log ring)", lost);
            any = true;
        }
    }
    if (const uint64_t lost = rings.take_unringed()) {
        write_report("HGEvolve: %llu diagnostic messages were dropped (no log ring)", lost);
        any = true;
    }
    if (any) std::fflush(out_);
    return any;
}

void FrameWriter::finish_handoff() {
    // Everything posted before the handoff is in the rings by now; it goes first.
    drain_rings();
    if (capture_omitted_) {
        char text[200];
        const int n = std::snprintf(text, sizeof text,
                                    "HGEvolve: %zu bytes of process output were left out "
                                    "(%zu bytes forwarded per job)",
                                    capture_omitted_, kCaptureBytesPerJob);
        if (n > 0) write_frame(kLogFrameBit | static_cast<uint64_t>(n), text, static_cast<std::size_t>(n));
        capture_omitted_ = 0;
    }
    if (reply_) {
        write_header(reply_->size());
        for (const auto& part : reply_->parts)
            if (!part.empty()) std::fwrite(part.data(), 1, part.size(), out_);
    } else if (error_) {
        write_header(kErrorFrameBit |
                     hgcommon::utf8_escaped_length(error_->data(), error_->size()));
        hgcommon::utf8_escaped(error_->data(), error_->size(),
                               [&](const char* s, std::size_t k) { std::fwrite(s, 1, k, out_); });
    }
    std::fflush(out_);
    reply_ = nullptr;
    error_ = nullptr;
    pending_.store(0, std::memory_order_release);
    hgcommon::unpark_all(pending_);
}

void FrameWriter::run() {
#if defined(__linux__)
    pthread_setname_np(pthread_self(), "hg_frame_writer");
#endif
    hgcommon::ParkGate& gate = writer_gate();
    for (;;) {
        const bool any = drain_rings();
        if (pending_.load(std::memory_order_seq_cst)) {
            finish_handoff();
            continue;
        }
        if (stop_.load(std::memory_order_seq_cst)) {
            drain_rings();
            return;
        }
        if (any) continue;
        // The last look drains the rings itself; the writer parks only when it found nothing.
        gate.park_unless(0, false,
                         [this] {
                             return drain_rings() || pending_.load(std::memory_order_seq_cst) ||
                                    stop_.load(std::memory_order_seq_cst);
                         },
                         [] { return true; });
    }
}

void FrameWriter::hand_off() {
    pending_.store(1, std::memory_order_seq_cst);
    writer_gate().wake_one(0);
    while (pending_.load(std::memory_order_acquire)) hgcommon::park_if_equal(pending_, 1);
}

void FrameWriter::write_reply(const HG_NAMESPACE::wxf::Pieces& reply) {
    reply_ = &reply;
    hand_off();
}

void FrameWriter::write_error(const std::string& message) {
    error_ = &message;
    hand_off();
}

void forward_output(int fd) {
    char buf[LogRing::kSlotBytes];
    std::size_t held = 0;
    for (;;) {
#if defined(_WIN32)
        const int n = _read(fd, buf + held, static_cast<unsigned>(sizeof buf - held));
#else
        const ssize_t n = ::read(fd, buf + held, sizeof buf - held);
#endif
        if (n <= 0) break;
        const std::size_t total = held + static_cast<std::size_t>(n);
        const std::size_t whole = hgcommon::utf8_complete_prefix(buf, total);
        if (whole) post_message(MessageKind::Capture, buf, whole);
        held = total - whole;
        std::memmove(buf, buf + whole, held);
    }
    if (held) post_message(MessageKind::Capture, buf, held);
}

bool start_output_capture() {
    int fds[2];
#if defined(_WIN32)
    if (_pipe(fds, 64 * 1024, _O_BINARY) != 0) return false;
    std::fflush(stdout);
    std::fflush(stderr);
    if (_dup2(fds[1], 1) != 0 || _dup2(fds[1], 2) != 0) return false;
    // Libraries that write through the process's standard handles (not the C runtime's
    // descriptors) write to the pipe too.
    SetStdHandle(STD_OUTPUT_HANDLE, reinterpret_cast<HANDLE>(_get_osfhandle(1)));
    SetStdHandle(STD_ERROR_HANDLE, reinterpret_cast<HANDLE>(_get_osfhandle(2)));
    _close(fds[1]);
#else
    if (::pipe(fds) != 0) return false;
    std::fflush(stdout);
    std::fflush(stderr);
    if (::dup2(fds[1], 1) < 0 || ::dup2(fds[1], 2) < 0) return false;
    ::close(fds[1]);
#endif
    // Device printf reaches fd 1 through the C runtime's stdout buffer; unbuffered, it arrives
    // when written.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    const int read_fd = fds[0];
    std::thread([read_fd] { forward_output(read_fd); }).detach();
    return true;
}

}  // namespace hgframe
