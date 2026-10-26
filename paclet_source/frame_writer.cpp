#include "frame_writer.hpp"

#include <chrono>
#include <cstring>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#else
#include <unistd.h>
#endif

namespace hgframe {

bool LogRing::push(MessageKind kind, const char* text, std::size_t length) {
    const uint64_t t = tail.load(std::memory_order_relaxed);
    if (t - head.load(std::memory_order_acquire) >= kSlots) {
        dropped.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    Slot& s = slots[t % kSlots];
    s.kind = kind;
    s.length = static_cast<uint32_t>(length < kSlotBytes ? length : kSlotBytes);
    std::memcpy(s.text, text, s.length);
    tail.store(t + 1, std::memory_order_release);
    return true;
}

namespace {

// Every ring ever claimed, in claim order. A ring is never freed: a thread that exits hands its
// ring on (owned = 0) to the next thread that reports, which continues its tail.
constexpr uint32_t kMaxRings = 4096;
std::atomic<LogRing*> g_rings[kMaxRings];
std::atomic<uint32_t> g_ring_count{0};

struct RingClaim {
    LogRing* ring = nullptr;
    ~RingClaim() {
        if (ring) ring->owned.store(0, std::memory_order_release);
    }
};
thread_local RingClaim t_claim;

LogRing* claim_ring() {
    const uint32_t n = g_ring_count.load(std::memory_order_acquire);
    for (uint32_t i = 0; i < n; ++i) {
        LogRing* r = g_rings[i].load(std::memory_order_acquire);
        uint32_t free = 0;
        if (r && r->owned.compare_exchange_strong(free, 1, std::memory_order_acq_rel)) return r;
    }
    const uint32_t i = g_ring_count.fetch_add(1, std::memory_order_acq_rel);
    if (i >= kMaxRings) return nullptr;
    LogRing* r = new LogRing();
    r->owned.store(1, std::memory_order_relaxed);
    g_rings[i].store(r, std::memory_order_release);
    return r;
}

}  // namespace

LogRing* this_thread_ring() {
    if (!t_claim.ring) t_claim.ring = claim_ring();
    return t_claim.ring;
}

void post_message(MessageKind kind, const char* text, std::size_t length) {
    LogRing* r = this_thread_ring();
    if (!r) return;
    do {
        const std::size_t part = length < LogRing::kSlotBytes ? length : LogRing::kSlotBytes;
        if (!r->push(kind, text, part)) return;
        text += part;
        length -= part;
    } while (length > 0);
}

// The thread starts after every member is constructed.
FrameWriter::FrameWriter(std::FILE* out) : out_(out) { thread_ = std::thread([this] { run(); }); }

FrameWriter::~FrameWriter() {
    {
        std::lock_guard<std::mutex> lock(wake_mutex_);
        stop_.store(true, std::memory_order_release);
    }
    wake_.notify_one();
    thread_.join();
}

void FrameWriter::begin_job() { job_seq_.fetch_add(1, std::memory_order_release); }

void FrameWriter::write_frame(uint64_t header, const char* data, std::size_t length) {
    uint8_t h[8];
    for (int i = 0; i < 8; ++i) h[i] = static_cast<uint8_t>(header >> (8 * i));
    std::fwrite(h, 1, 8, out_);
    if (length) std::fwrite(data, 1, length, out_);
}

bool FrameWriter::drain_rings() {
    bool any = false;
    // A new job starts its own count of forwarded process output.
    const uint64_t job = job_seq_.load(std::memory_order_acquire);
    if (job != seen_job_) {
        seen_job_ = job;
        capture_forwarded_ = 0;
    }
    const uint32_t n = g_ring_count.load(std::memory_order_acquire);
    for (uint32_t i = 0; i < n && i < kMaxRings; ++i) {
        LogRing* r = g_rings[i].load(std::memory_order_acquire);
        if (!r) continue;
        while (r->pop([&](MessageKind kind, const char* text, uint32_t length) {
            if (kind == MessageKind::Capture) {
                if (capture_forwarded_ + length > kCaptureBytesPerJob) {
                    capture_omitted_ += length;
                    return;
                }
                capture_forwarded_ += length;
            }
            write_frame((kind == MessageKind::Progress ? kProgressFrameBit : kLogFrameBit) |
                            length, text, length);
        }))
            any = true;
        const uint64_t lost = r->dropped.exchange(0, std::memory_order_relaxed);
        if (lost) {
            const std::string m = "HGEvolve: " + std::to_string(lost) +
                                  " diagnostic messages were dropped (a full log ring)";
            write_frame(kLogFrameBit | m.size(), m.data(), m.size());
            any = true;
        }
    }
    if (any) std::fflush(out_);
    return any;
}

void FrameWriter::finish_handoff() {
    // Everything posted before the handoff is in the rings by now; it goes first.
    drain_rings();
    if (capture_omitted_) {
        const std::string m = "HGEvolve: " + std::to_string(capture_omitted_) +
                              " bytes of process output were left out (" +
                              std::to_string(kCaptureBytesPerJob) + " bytes forwarded per job)";
        write_frame(kLogFrameBit | m.size(), m.data(), m.size());
        capture_omitted_ = 0;
    }
    if (reply_) {
        uint8_t h[8];
        const uint64_t len = reply_->size();
        for (int i = 0; i < 8; ++i) h[i] = static_cast<uint8_t>(len >> (8 * i));
        std::fwrite(h, 1, 8, out_);
        for (const auto& part : reply_->parts)
            if (!part.empty()) std::fwrite(part.data(), 1, part.size(), out_);
    } else if (error_) {
        write_frame(kErrorFrameBit | error_->size(), error_->data(), error_->size());
    }
    std::fflush(out_);
    reply_ = nullptr;
    error_ = nullptr;
    pending_.store(0, std::memory_order_release);
    pending_.notify_all();
}

void FrameWriter::run() {
    for (;;) {
        const bool any = drain_rings();
        if (pending_.load(std::memory_order_acquire)) {
            finish_handoff();
            continue;
        }
        if (stop_.load(std::memory_order_acquire)) {
            drain_rings();
            return;
        }
        if (!any) {
            // Messages are picked up within 1 ms; a reply wakes the writer at once.
            std::unique_lock<std::mutex> lock(wake_mutex_);
            wake_.wait_for(lock, std::chrono::milliseconds(1), [this] {
                return pending_.load(std::memory_order_acquire) != 0 ||
                       stop_.load(std::memory_order_acquire);
            });
        }
    }
}

void FrameWriter::write_reply(const HG_NAMESPACE::wxf::Pieces& reply) {
    reply_ = &reply;
    {
        std::lock_guard<std::mutex> lock(wake_mutex_);
        pending_.store(1, std::memory_order_release);
    }
    wake_.notify_one();
    pending_.wait(1, std::memory_order_acquire);
}

void FrameWriter::write_error(const std::string& message) {
    error_ = &message;
    {
        std::lock_guard<std::mutex> lock(wake_mutex_);
        pending_.store(1, std::memory_order_release);
    }
    wake_.notify_one();
    pending_.wait(1, std::memory_order_acquire);
}

bool start_output_capture() {
    int fds[2];
#if defined(_WIN32)
    if (_pipe(fds, 64 * 1024, _O_BINARY) != 0) return false;
    std::fflush(stdout);
    std::fflush(stderr);
    if (_dup2(fds[1], 1) != 0 || _dup2(fds[1], 2) != 0) return false;
    // Libraries that write through the process's standard handles (not the C runtime's
    // descriptors) see the pipe too.
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
    std::thread([read_fd] {
        char buf[LogRing::kSlotBytes];
        for (;;) {
#if defined(_WIN32)
            const int n = _read(read_fd, buf, sizeof buf);
#else
            const ssize_t n = ::read(read_fd, buf, sizeof buf);
#endif
            if (n <= 0) return;
            post_message(MessageKind::Capture, buf, static_cast<std::size_t>(n));
        }
    }).detach();
    return true;
}

}  // namespace hgframe
