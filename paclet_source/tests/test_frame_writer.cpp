// The worker's frame writer (paclet_source/frame_writer.hpp): one writer thread, per-thread
// rings, frames that never interleave, drops that are counted, text cut at UTF-8 character
// boundaries, a writer that sleeps while idle, and the capture of fd 1 and fd 2.
// Runs on POSIX and on Windows (the C runtime's _pipe, _read, _dup2, and SetStdHandle).
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#else
#include <unistd.h>
#endif
#if defined(__linux__)
#include <dirent.h>
#include <fstream>
#endif

#include "frame_writer.hpp"
#include "hgcommon/utf8.hpp"

namespace {

// The C runtime's pipe and descriptor calls under one name on both platforms.
bool make_pipe(int fds[2]) {
#if defined(_WIN32)
    return _pipe(fds, 1 << 16, _O_BINARY) == 0;
#else
    return ::pipe(fds) == 0;
#endif
}
long read_fd(int fd, void* buf, std::size_t n) {
#if defined(_WIN32)
    return _read(fd, buf, static_cast<unsigned>(n));
#else
    return static_cast<long>(::read(fd, buf, n));
#endif
}
long write_fd(int fd, const void* buf, std::size_t n) {
#if defined(_WIN32)
    return _write(fd, buf, static_cast<unsigned>(n));
#else
    return static_cast<long>(::write(fd, buf, n));
#endif
}
void close_fd(int fd) {
#if defined(_WIN32)
    _close(fd);
#else
    ::close(fd);
#endif
}
std::FILE* open_write_stream(int fd) {
#if defined(_WIN32)
    return _fdopen(fd, "wb");
#else
    return ::fdopen(fd, "wb");
#endif
}

struct Frame {
    uint64_t kind;   // the header's top three bits
    std::string payload;
};

// Reads every frame from `fd` until the end of the stream.
std::vector<Frame> read_frames(int fd) {
    std::vector<Frame> frames;
    auto read_n = [&](std::size_t n, std::string& out) {
        out.resize(n);
        std::size_t got = 0;
        while (got < n) {
            const long r = read_fd(fd, &out[got], n - got);
            if (r <= 0) return false;
            got += static_cast<std::size_t>(r);
        }
        return true;
    };
    for (;;) {
        std::string h;
        if (!read_n(8, h)) return frames;
        uint64_t len = 0;
        for (int i = 0; i < 8; ++i) len |= static_cast<uint64_t>(static_cast<uint8_t>(h[i])) << (8 * i);
        const uint64_t kind = len & (7ull << 61);
        Frame f{kind, {}};
        if (!read_n(static_cast<std::size_t>(len & ~(7ull << 61)), f.payload)) return frames;
        frames.push_back(std::move(f));
    }
}

// A FrameWriter on a pipe and a thread reading its frames. finish() writes a reply, stops the
// writer and returns every frame it wrote: those before the reply, the reply, and those it wrote
// after the reply before it stopped (a message posted while the reply was written).
struct WriterOnPipe {
    int fds[2] = {-1, -1};
    std::FILE* out = nullptr;
    hgframe::FrameWriter* writer = nullptr;
    std::vector<Frame> frames;
    std::thread reader;

    WriterOnPipe() {
        EXPECT_TRUE(make_pipe(fds));
        out = open_write_stream(fds[1]);
        EXPECT_NE(out, nullptr);
        reader = std::thread([this] { frames = read_frames(fds[0]); });
        writer = new hgframe::FrameWriter(out);
    }
    std::vector<Frame> finish(std::vector<uint8_t> reply_bytes = {1, 2, 3}) {
        HG_NAMESPACE::wxf::Pieces reply;
        reply.parts.push_back(std::move(reply_bytes));
        writer->write_reply(reply);
        delete writer;
        writer = nullptr;
        std::fclose(out);
        reader.join();
        close_fd(fds[0]);
        return frames;
    }
};

uint64_t dropped_in(const std::string& text) {
    const auto at = text.find(" diagnostic messages were dropped");
    if (text.rfind("HGEvolve: ", 0) != 0 || at == std::string::npos) return 0;
    return std::stoull(text.substr(10, at - 10));
}

// True when `s` is whole UTF-8 characters: it neither starts with a continuation byte nor ends
// inside a character.
bool whole_characters(const std::string& s) {
    if (s.empty()) return true;
    if ((static_cast<unsigned char>(s[0]) & 0xC0) == 0x80) return false;
    return hgcommon::utf8_complete_prefix(s.data(), s.size()) == s.size();
}

}  // namespace

// 16 threads each post 2000 log messages of distinct text and lengths up to 900 bytes while the
// writer runs; then the reply. Every frame the reader parses is a whole message, a drop report
// or the reply: received + reported drops = sent, each thread's messages arrive in its order, and
// the reply is last and whole. 10 rounds.
TEST(FrameWriter, FramesNeverInterleave) {
    constexpr int kThreads = 16, kMessages = 2000;
    for (int round = 0; round < 10; ++round) {
        WriterOnPipe w;
        std::vector<std::thread> producers;
        for (int t = 0; t < kThreads; ++t)
            producers.emplace_back([t] {
                for (int m = 0; m < kMessages; ++m) {
                    std::string text = "t" + std::to_string(t) + " m" + std::to_string(m) + " ";
                    text.append(static_cast<size_t>((m * 37 + t * 11) % 880), static_cast<char>('a' + t));
                    hgframe::post_message(hgframe::MessageKind::Log, text.data(), text.size());
                }
            });
        for (auto& p : producers) p.join();
        const std::vector<Frame> frames = w.finish(std::vector<uint8_t>(5000, 0x5A));
        ASSERT_FALSE(frames.empty());
        EXPECT_EQ(frames.back().kind, 0u);
        EXPECT_EQ(frames.back().payload, std::string(5000, '\x5A'));
        uint64_t received = 0, dropped = 0;
        std::map<int, int> last;
        for (size_t i = 0; i + 1 < frames.size(); ++i) {
            const Frame& f = frames[i];
            ASSERT_EQ(f.kind, hgframe::kLogFrameBit) << "round " << round << " frame " << i;
            if (const uint64_t d = dropped_in(f.payload)) {
                dropped += d;
                continue;
            }
            int t = -1, m = -1;
            ASSERT_EQ(std::sscanf(f.payload.c_str(), "t%d m%d ", &t, &m), 2) << f.payload.substr(0, 40);
            const std::string head = "t" + std::to_string(t) + " m" + std::to_string(m) + " ";
            ASSERT_EQ(f.payload.size(), head.size() + static_cast<size_t>((m * 37 + t * 11) % 880));
            EXPECT_EQ(f.payload.find_first_not_of(static_cast<char>('a' + t), head.size()),
                      std::string::npos);
            auto it = last.find(t);
            if (it != last.end()) EXPECT_GT(m, it->second) << "thread " << t << " out of order";
            last[t] = m;
            ++received;
        }
        EXPECT_EQ(received + dropped, uint64_t(kThreads) * kMessages) << "round " << round;
    }
}

// A thread posts 200 messages before any writer runs: its ring keeps the first 64 and counts 136
// dropped, and the writer sends the 64 and a report of 136.
TEST(FrameWriter, AFullRingDropsAndReportsTheCount) {
    std::thread([] {
        for (int m = 0; m < 200; ++m) {
            const std::string text = "early " + std::to_string(m);
            hgframe::post_message(hgframe::MessageKind::Log, text.data(), text.size());
        }
    }).join();
    WriterOnPipe w;
    uint64_t early = 0, dropped = 0;
    for (const Frame& f : w.finish()) {
        if (f.payload.rfind("early ", 0) == 0) ++early;
        dropped += dropped_in(f.payload);
    }
    EXPECT_EQ(early, uint64_t(hgframe::LogRing::kSlots));
    EXPECT_EQ(dropped, 200u - hgframe::LogRing::kSlots);
}

// 1, 4 and 16 threads post 200000 messages each while the writer runs. Every message reaches the
// reader or a drop report, and the threads that post at once hold distinct rings. The time per
// message is printed.
TEST(FrameWriter, ProducersEachWriteTheirOwnRing) {
    for (int threads : {1, 4, 16}) {
        constexpr int kMessages = 200000;
        WriterOnPipe w;
        std::vector<std::thread> producers;
        std::vector<double> ns(threads);
        std::vector<hgframe::LogRing*> rings(threads);
        std::atomic<int> started{0};
        for (int t = 0; t < threads; ++t)
            producers.emplace_back([&, t] {
                const char text[] = "progress: 123456 states, 654321 events";
                rings[t] = hgframe::this_thread_ring();
                started.fetch_add(1);
                while (started.load() < threads) std::this_thread::yield();
                const auto t0 = std::chrono::steady_clock::now();
                for (int m = 0; m < kMessages; ++m)
                    hgframe::post_message(hgframe::MessageKind::Log, text, sizeof text - 1);
                ns[t] = std::chrono::duration<double, std::nano>(
                            std::chrono::steady_clock::now() - t0).count() / kMessages;
            });
        for (auto& p : producers) p.join();
        uint64_t received = 0, dropped = 0;
        for (const Frame& f : w.finish()) {
            if (f.payload == "progress: 123456 states, 654321 events") ++received;
            dropped += dropped_in(f.payload);
        }
        EXPECT_EQ(received + dropped, uint64_t(threads) * kMessages) << threads << " threads";
        EXPECT_EQ(std::set<hgframe::LogRing*>(rings.begin(), rings.end()).size(),
                  static_cast<size_t>(threads));
        EXPECT_EQ(std::count(rings.begin(), rings.end(), nullptr), 0);
        double mean = 0;
        for (double v : ns) mean += v / threads;
        std::printf("[frame_writer] %d producer thread(s): %.1f ns per message, %llu dropped\n",
                    threads, mean, static_cast<unsigned long long>(dropped));
    }
}

// A message of 2600 bytes whose characters (2, 3 and 4 bytes) straddle the 1000-byte slot size
// arrives as frames of whole characters that join to the message.
TEST(FrameWriter, LogTextIsCutAtCharacterBoundaries) {
    std::string text(999, 'a');
    text += "\xC3\xA9";              // bytes 999-1000: straddles the first slot's end
    text += "\xE2\x82\xAC";          // bytes 1001-1003
    text.append(994, 'b');
    text += "\xF0\x9F\x98\x80";      // bytes 1998-2001: straddles the second slot's end (1999)
    while (text.size() < 2600) text += "\xC3\xA9";
    WriterOnPipe w;
    std::thread([&] { hgframe::post_message(hgframe::MessageKind::Log, text.data(), text.size()); }).join();
    std::string joined;
    for (const Frame& f : w.finish()) {
        if (f.kind != hgframe::kLogFrameBit) continue;
        EXPECT_TRUE(whole_characters(f.payload)) << "a frame of " << f.payload.size() << " bytes";
        EXPECT_LE(f.payload.size(), hgframe::LogRing::kSlotBytes);
        joined += f.payload;
    }
    EXPECT_EQ(joined, text);
}

// The capture loop reads a pipe 1000 bytes at a time. Output whose characters straddle the read
// size arrives as whole characters, all of it.
TEST(FrameWriter, CapturedOutputKeepsCharactersWhole) {
    std::string text(999, 'c');
    text += "\xE2\x82\xAC";   // bytes 999-1001: the first read ends inside it
    for (int i = 0; i < 700; ++i) text += "\xC3\xA9";
    int fds[2];
    ASSERT_TRUE(make_pipe(fds));
    WriterOnPipe w;
    std::thread capture([&] { hgframe::forward_output(fds[0]); });
    ASSERT_EQ(write_fd(fds[1], text.data(), text.size()), static_cast<long>(text.size()));
    close_fd(fds[1]);
    capture.join();
    close_fd(fds[0]);
    std::string joined;
    for (const Frame& f : w.finish()) {
        if (f.kind != hgframe::kLogFrameBit) continue;
        EXPECT_TRUE(whole_characters(f.payload)) << "a frame of " << f.payload.size() << " bytes";
        joined += f.payload;
    }
    EXPECT_EQ(joined, text);
}

// start_output_capture points fd 1 and fd 2, and on Windows the standard output and error handles,
// at the capture pipe. Text written each way arrives as log frames. The test restores the
// descriptors and handles afterwards, which closes the pipe and ends the capture thread.
TEST(FrameWriter, TheCaptureCarriesBothStreams) {
    std::fflush(stdout);
    std::fflush(stderr);
#if defined(_WIN32)
    const int saved_out = _dup(1), saved_err = _dup(2);
    const HANDLE saved_out_handle = GetStdHandle(STD_OUTPUT_HANDLE);
    const HANDLE saved_err_handle = GetStdHandle(STD_ERROR_HANDLE);
#else
    const int saved_out = ::dup(1), saved_err = ::dup(2);
#endif
    ASSERT_GE(saved_out, 0);
    ASSERT_GE(saved_err, 0);
    WriterOnPipe w;
    ASSERT_TRUE(hgframe::start_output_capture());
    std::fprintf(stdout, "capture-marker-stdout\n");
    std::fprintf(stderr, "capture-marker-stderr\n");
    std::vector<std::string> expected = {"capture-marker-stdout", "capture-marker-stderr"};
#if defined(_WIN32)
    DWORD written = 0;
    const char out_text[] = "capture-marker-out-handle\n";
    const char err_text[] = "capture-marker-err-handle\n";
    WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), out_text, sizeof out_text - 1, &written, nullptr);
    WriteFile(GetStdHandle(STD_ERROR_HANDLE), err_text, sizeof err_text - 1, &written, nullptr);
    expected.push_back("capture-marker-out-handle");
    expected.push_back("capture-marker-err-handle");
    _dup2(saved_out, 1);
    _dup2(saved_err, 2);
    SetStdHandle(STD_OUTPUT_HANDLE, saved_out_handle);
    SetStdHandle(STD_ERROR_HANDLE, saved_err_handle);
    _close(saved_out);
    _close(saved_err);
#else
    ::dup2(saved_out, 1);
    ::dup2(saved_err, 2);
    ::close(saved_out);
    ::close(saved_err);
#endif
    // The capture thread posts what it reads as it reads it; the replies below collect the frames
    // until every marker has arrived, for at most 10 s.
    std::string captured;
    HG_NAMESPACE::wxf::Pieces reply;
    reply.parts.push_back({7});
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    auto all_found = [&] {
        for (const auto& m : expected)
            if (captured.find(m) == std::string::npos) return false;
        return true;
    };
    std::vector<Frame> frames = w.finish();
    for (const Frame& f : frames)
        if (f.kind == hgframe::kLogFrameBit) captured += f.payload;
    while (!all_found() && std::chrono::steady_clock::now() < deadline) {
        WriterOnPipe again;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        for (const Frame& f : again.finish())
            if (f.kind == hgframe::kLogFrameBit) captured += f.payload;
    }
    for (const auto& m : expected) EXPECT_NE(captured.find(m), std::string::npos) << m;
}

#if defined(__linux__)
namespace {
// The voluntary context switches of the thread named `name`, or -1 when there is none.
long voluntary_switches(const std::string& name) {
    DIR* d = opendir("/proc/self/task");
    if (!d) return -1;
    long found = -1;
    while (dirent* e = readdir(d)) {
        if (e->d_name[0] == '.') continue;
        const std::string base = std::string("/proc/self/task/") + e->d_name;
        std::ifstream comm(base + "/comm");
        std::string c;
        std::getline(comm, c);
        if (c != name) continue;
        std::ifstream status(base + "/status");
        std::string line;
        while (std::getline(status, line))
            if (line.rfind("voluntary_ctxt_switches:", 0) == 0)
                found = std::stol(line.substr(line.find(':') + 1));
    }
    closedir(d);
    return found;
}
}  // namespace

// An idle writer sleeps: over one second with nothing posted it is switched in at most 5 times.
// A message posted afterwards still arrives.
TEST(FrameWriter, AnIdleWriterSleeps) {
    WriterOnPipe w;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const long before = voluntary_switches("hg_frame_writer");
    ASSERT_GE(before, 0) << "no thread named hg_frame_writer";
    std::this_thread::sleep_for(std::chrono::seconds(1));
    const long after = voluntary_switches("hg_frame_writer");
    std::printf("[frame_writer] idle writer: %ld context switches in 1 s\n", after - before);
    EXPECT_LE(after - before, 5);
    std::thread([] { hgframe::post_message(hgframe::MessageKind::Log, "late", 4); }).join();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    bool late = false;
    for (const Frame& f : w.finish()) late = late || f.payload == "late";
    EXPECT_TRUE(late);
}
#endif

// The registry claims at most N rings. Claims beyond N, and claims whose allocation fails, return
// nullptr; a ring handed on by an exited thread is claimed again.
namespace {
hgframe::LogRing* no_ring() { return nullptr; }
hgframe::LogRing* new_ring() { return new hgframe::LogRing(); }
}  // namespace

TEST(FrameWriter, TheRingRegistryIsBounded) {
    hgframe::RingRegistry<3> rings(new_ring);
    hgframe::LogRing* a = rings.claim();
    hgframe::LogRing* b = rings.claim();
    hgframe::LogRing* c = rings.claim();
    ASSERT_TRUE(a && b && c);
    for (int i = 0; i < 10; ++i) EXPECT_EQ(rings.claim(), nullptr);
    EXPECT_EQ(rings.size(), 3u);
    b->owned.store(0);
    EXPECT_EQ(rings.claim(), b);

    hgframe::RingRegistry<3> failing(no_ring);
    EXPECT_EQ(failing.claim(), nullptr);
    EXPECT_EQ(failing.size(), 0u);
    failing.count_unringed();
    failing.count_unringed();
    EXPECT_EQ(failing.take_unringed(), 2u);
    EXPECT_EQ(failing.take_unringed(), 0u);
}
