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
// reported at once, at most kMaxRings. A thread that gets no ring (kMaxRings in use, or no memory
// for a new one) drops its messages, and the writer reports that count too.
//
// The writer parks when every ring is empty and no reply is waiting. A producer wakes it through
// one hgcommon::ParkGate (park_gate.hpp): the producer publishes its message, passes the gate's
// barrier and signals only when the writer has announced itself idle.
//
// A message takes as many slots as its length needs, cut at UTF-8 character boundaries
// (hgcommon/utf8.hpp), and is published whole or dropped whole. The writer sends it as one frame.
//
// THIRD-PARTY OUTPUT. The CUDA runtime prints device printf to fd 1, and drivers and libraries
// write to fd 1 and fd 2. start_output_capture points both at a pipe the worker owns; its thread
// reads the pipe continuously into its ring (forward_output), so that output cannot fill a pipe
// and stop the process. The writer forwards at most kCaptureBytesPerJob of it per job and reports
// the bytes it left out.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>

#include "hgcommon/park_gate.hpp"
#include "wxf.hpp"

namespace hgframe {

constexpr uint64_t kProgressFrameBit = 1ull << 63;
constexpr uint64_t kErrorFrameBit = 1ull << 62;
constexpr uint64_t kLogFrameBit = 1ull << 61;

// The bytes of process output (fd 1 and fd 2) forwarded per job; the rest is counted.
constexpr std::size_t kCaptureBytesPerJob = 256 * 1024;

enum class MessageKind : uint8_t { Log = 0, Progress = 1, Capture = 2 };

// One bounded ring of kSlots slots of kSlotBytes each; a message takes one or more consecutive
// slots. The producer writes tail, the consumer head; each is on its own cache line.
struct LogRing {
    static constexpr uint32_t kSlots = 64;
    static constexpr uint32_t kSlotBytes = 1000;
    struct Slot {
        uint32_t length;
        uint16_t parts;  // the message's slot count in its first slot, 0 in the others
        MessageKind kind;
        char text[kSlotBytes];
    };
    alignas(64) std::atomic<uint64_t> tail{0};
    alignas(64) std::atomic<uint64_t> head{0};
    alignas(64) std::atomic<uint64_t> dropped{0};
    std::atomic<uint32_t> owned{0};
    Slot slots[kSlots];

    // Producer side: the message in the slots it needs, published together. False (and the
    // message counted as dropped) when the free slots are fewer.
    bool push(MessageKind kind, const char* text, std::size_t length);
    // Consumer side: the oldest message as take(kind, parts, part), where part(i) is the i-th
    // slot of the message; false when the ring is empty.
    template <class F>
    bool pop(F&& take) {
        const uint64_t h = head.load(std::memory_order_relaxed);
        if (h == tail.load(std::memory_order_acquire)) return false;
        const Slot& first = slots[h % kSlots];
        const uint32_t parts = first.parts ? first.parts : 1;
        take(first.kind, parts, [&](uint32_t i) -> const Slot& { return slots[(h + i) % kSlots]; });
        head.store(h + parts, std::memory_order_release);
        return true;
    }
};

// Every ring claimed, in claim order, at most N. A ring is never freed while the registry lives: a
// thread that exits hands its ring on (owned = 0) to the next thread that claims one, which
// continues its tail. `make` allocates a ring and returns nullptr when it cannot.
template <uint32_t N>
class RingRegistry {
public:
    using MakeRing = LogRing* (*)();
    explicit RingRegistry(MakeRing make) : make_(make) {}
    ~RingRegistry() {
        for (uint32_t i = 0; i < size(); ++i) delete rings_[i].load(std::memory_order_acquire);
    }
    RingRegistry(const RingRegistry&) = delete;
    RingRegistry& operator=(const RingRegistry&) = delete;

    // A ring handed on by an exited thread, else a new one; nullptr when N rings are in use or a
    // new ring cannot be allocated. The scan reads at most N entries.
    LogRing* claim() {
        const uint32_t n = size();
        for (uint32_t i = 0; i < n; ++i) {
            LogRing* r = rings_[i].load(std::memory_order_acquire);
            uint32_t free = 0;
            if (r && r->owned.compare_exchange_strong(free, 1, std::memory_order_acq_rel)) return r;
        }
        if (n >= N) return nullptr;
        LogRing* r = make_();
        if (!r) return nullptr;
        r->owned.store(1, std::memory_order_relaxed);
        uint32_t i = count_.load(std::memory_order_relaxed);
        do {
            if (i >= N) {
                delete r;
                return nullptr;
            }
        } while (!count_.compare_exchange_weak(i, i + 1, std::memory_order_acq_rel,
                                               std::memory_order_relaxed));
        rings_[i].store(r, std::memory_order_release);
        return r;
    }
    // The entries claimed so far, at most N; an entry reads nullptr until its ring is published.
    uint32_t size() const { return count_.load(std::memory_order_acquire); }
    LogRing* at(uint32_t i) const { return rings_[i].load(std::memory_order_acquire); }

    // Messages from threads that got no ring.
    void count_unringed() { unringed_.fetch_add(1, std::memory_order_relaxed); }
    uint64_t take_unringed() { return unringed_.exchange(0, std::memory_order_relaxed); }

private:
    MakeRing make_;
    std::atomic<LogRing*> rings_[N] = {};
    std::atomic<uint32_t> count_{0};
    std::atomic<uint64_t> unringed_{0};
};

constexpr uint32_t kMaxRings = 4096;
using WorkerRings = RingRegistry<kMaxRings>;

// The process's registry, created on first use and never destroyed (the capture thread may post
// until the process exits).
WorkerRings& worker_rings();

// The calling thread's ring, claimed on its first message; nullptr when it got none.
LogRing* this_thread_ring();

// Puts a message in the calling thread's ring and wakes the writer if it is parked; never blocks.
// A message longer than a slot takes several (LogRing::push).
void post_message(MessageKind kind, const char* text, std::size_t length);

class FrameWriter {
public:
    // Starts the writer thread on `out`, which only the writer writes from then on. One
    // FrameWriter at a time per process.
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
    void write_header(uint64_t header);
    void write_frame(uint64_t header, const char* data, std::size_t length);
    void write_report(const char* format, unsigned long long count);
    void finish_handoff();
    void hand_off();

    std::FILE* out_;
    std::thread thread_;
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

// The capture thread's loop: reads `fd` until end of file or an error, and posts what it reads as
// Capture messages. A character split across two reads is held back and posted whole with the
// next read.
void forward_output(int fd);

// Points fd 1 and fd 2 (and on Windows the process's standard output and error handles) at a
// pipe and starts a thread that runs forward_output on it. Call after the reply stream has its
// own descriptor (take_reply_stream). False when the pipe cannot be made.
bool start_output_capture();

}  // namespace hgframe
