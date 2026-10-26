#pragma once
// THE WORKER'S OUTPUT. hg_evolve --serve sends the paclet a stream of frames on its standard
// output, each [8-byte little-endian length][payload]. The two top bits of the length and bit 61
// name the frame's kind:
//   bit 63  PROGRESS  one progress message; the job's reply still follows
//   bit 62  ERROR     the job failed; the payload is the message, and nothing follows
//   bit 61  LOG       one diagnostic message, or text a library wrote to fd 1 or fd 2
//   none    REPLY     the job's WXF reply
// One writer thread writes every frame, so frames never interleave.
//
// Producers never block. Each thread that reports (the job's calling thread, every engine worker
// through hgcommon::diag_printf, the capture thread) owns one bounded single-producer,
// single-consumer ring; the writer is the one consumer of all of them. A full ring drops the
// message and counts it, and the writer reports the count in a log frame. The rings are claimed
// per thread and handed on when the thread exits, so their number is the most threads that
// reported at once.
//
// THIRD-PARTY OUTPUT. The CUDA runtime prints device printf to fd 1, and drivers and libraries
// write to fd 1 and fd 2. OutputCapture points both at a pipe the worker owns; its thread reads
// the pipe continuously into its ring, so that output cannot fill a pipe and stop the process. The
// writer forwards at most kCaptureBytesPerJob of it per job and reports the bytes it left out.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>

#include "wxf.hpp"

namespace hgframe {

constexpr uint64_t kProgressFrameBit = 1ull << 63;
constexpr uint64_t kErrorFrameBit = 1ull << 62;
constexpr uint64_t kLogFrameBit = 1ull << 61;

// The bytes of process output (fd 1 and fd 2) forwarded per job; the rest is counted.
constexpr std::size_t kCaptureBytesPerJob = 256 * 1024;

enum class MessageKind : uint8_t { Log = 0, Progress = 1, Capture = 2 };

// One bounded ring: kSlots messages of at most kSlotBytes each (a longer message is cut into
// several). The producer writes tail, the consumer head; each is on its own cache line.
struct LogRing {
    static constexpr uint32_t kSlots = 64;
    static constexpr uint32_t kSlotBytes = 1000;
    struct Slot {
        uint32_t length;
        MessageKind kind;
        char text[kSlotBytes];
    };
    alignas(64) std::atomic<uint64_t> tail{0};
    alignas(64) std::atomic<uint64_t> head{0};
    alignas(64) std::atomic<uint64_t> dropped{0};
    std::atomic<uint32_t> owned{0};
    Slot slots[kSlots];

    // Producer side. False (and the message counted as dropped) when the ring is full.
    bool push(MessageKind kind, const char* text, std::size_t length);
    // Consumer side: the oldest message, or false when the ring is empty.
    template <class F>
    bool pop(F&& take) {
        const uint64_t h = head.load(std::memory_order_relaxed);
        if (h == tail.load(std::memory_order_acquire)) return false;
        const Slot& s = slots[h % kSlots];
        take(s.kind, s.text, s.length);
        head.store(h + 1, std::memory_order_release);
        return true;
    }
};

// The calling thread's ring, claimed on its first message.
LogRing* this_thread_ring();

// Puts a message in the calling thread's ring; never blocks. Messages longer than a slot take
// several slots.
void post_message(MessageKind kind, const char* text, std::size_t length);

class FrameWriter {
public:
    // Starts the writer thread on `out`, which only the writer writes from then on.
    explicit FrameWriter(std::FILE* out);
    ~FrameWriter();
    FrameWriter(const FrameWriter&) = delete;
    FrameWriter& operator=(const FrameWriter&) = delete;

    // Starts a job's count of forwarded process output.
    void begin_job();
    // Writes the job's reply, or its error frame, after every message posted before the call;
    // returns once the frame is written.
    void write_reply(const HG_NAMESPACE::wxf::Pieces& reply);
    void write_error(const std::string& message);

private:
    void run();
    bool drain_rings();
    void write_frame(uint64_t header, const char* data, std::size_t length);
    void finish_handoff();

    std::FILE* out_;
    std::thread thread_;
    std::mutex wake_mutex_;
    std::condition_variable wake_;
    std::atomic<bool> stop_{false};
    // The handoff: the calling thread sets one of these and pending_, the writer clears pending_.
    const HG_NAMESPACE::wxf::Pieces* reply_ = nullptr;
    const std::string* error_ = nullptr;
    std::atomic<uint32_t> pending_{0};
    std::atomic<uint64_t> job_seq_{0};
    // The writer's own: the job whose output it is counting, and that job's counts.
    uint64_t seen_job_ = 0;
    std::size_t capture_forwarded_ = 0;
    std::size_t capture_omitted_ = 0;
};

// Points fd 1 and fd 2 (and on Windows the process's standard output and error handles) at a
// pipe and starts a thread that reads it into its ring as Capture messages. Call after the
// reply stream has its own descriptor (take_reply_stream). False when the pipe cannot be made.
bool start_output_capture();

}  // namespace hgframe
