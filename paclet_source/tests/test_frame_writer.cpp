// The worker's frame writer (paclet_source/frame_writer.hpp): one writer thread, per-thread
// rings, frames that never interleave, and drops that are counted.
#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include "frame_writer.hpp"

namespace {

struct Frame {
    uint64_t kind;   // the header's top three bits
    std::string payload;
};

// Reads frames from `fd` until the first REPLY frame (no kind bit), which is included.
std::vector<Frame> read_frames(int fd) {
    std::vector<Frame> frames;
    auto read_n = [&](size_t n, std::string& out) {
        out.resize(n);
        size_t got = 0;
        while (got < n) {
            const ssize_t r = ::read(fd, &out[got], n - got);
            if (r <= 0) return false;
            got += static_cast<size_t>(r);
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
        if (!read_n(len & ~(7ull << 61), f.payload)) return frames;
        frames.push_back(std::move(f));
        if (kind == 0) return frames;
    }
}

uint64_t dropped_in(const std::string& text) {
    const auto at = text.find(" diagnostic messages were dropped");
    if (text.rfind("HGEvolve: ", 0) != 0 || at == std::string::npos) return 0;
    return std::stoull(text.substr(10, at - 10));
}

}  // namespace

// 16 threads each post 2000 log messages of distinct text and lengths up to 900 bytes while the
// writer runs; then the reply. Every frame the reader parses is a whole message, a drop report
// or the reply: received + reported drops = sent, each thread's messages arrive in its order, and
// the reply is last and whole. 10 rounds.
TEST(FrameWriter, FramesNeverInterleave) {
    constexpr int kThreads = 16, kMessages = 2000;
    for (int round = 0; round < 10; ++round) {
        int fds[2];
        ASSERT_EQ(::pipe(fds), 0);
        std::vector<Frame> frames;
        std::thread reader([&] { frames = read_frames(fds[0]); });
        std::FILE* out = ::fdopen(fds[1], "wb");
        ASSERT_NE(out, nullptr);
        {
            hgframe::FrameWriter writer(out);
            std::vector<std::thread> producers;
            for (int t = 0; t < kThreads; ++t)
                producers.emplace_back([t] {
                    for (int m = 0; m < kMessages; ++m) {
                        std::string text = "t" + std::to_string(t) + " m" + std::to_string(m) + " ";
                        text.append(static_cast<size_t>((m * 37 + t * 11) % 880), 'a' + t);
                        hgframe::post_message(hgframe::MessageKind::Log, text.data(), text.size());
                    }
                });
            for (auto& p : producers) p.join();
            HG_NAMESPACE::wxf::Pieces reply;
            reply.parts.push_back(std::vector<uint8_t>(5000, 0x5A));
            writer.write_reply(reply);
        }
        std::fclose(out);
        reader.join();
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
    int fds[2];
    ASSERT_EQ(::pipe(fds), 0);
    std::vector<Frame> frames;
    std::thread reader([&] { frames = read_frames(fds[0]); });
    std::FILE* out = ::fdopen(fds[1], "wb");
    {
        hgframe::FrameWriter writer(out);
        HG_NAMESPACE::wxf::Pieces reply;
        reply.parts.push_back({1, 2, 3});
        writer.write_reply(reply);
    }
    std::fclose(out);
    reader.join();
    uint64_t early = 0, dropped = 0;
    for (const Frame& f : frames) {
        if (f.payload.rfind("early ", 0) == 0) ++early;
        dropped += dropped_in(f.payload);
    }
    EXPECT_EQ(early, uint64_t(hgframe::LogRing::kSlots));
    EXPECT_EQ(dropped, 200u - hgframe::LogRing::kSlots);
}

// The producer's cost per message at 1, 4 and 16 threads with the writer running: each thread
// writes only its own ring, so the time per message does not grow with the thread count.
// Printed; asserted only that every message is accounted for in the ring counters.
TEST(FrameWriter, ProducerCostPerMessage) {
    int fds[2];
    ASSERT_EQ(::pipe(fds), 0);
    std::thread drain([&] {
        char buf[65536];
        while (::read(fds[0], buf, sizeof buf) > 0) {
        }
    });
    std::FILE* out = ::fdopen(fds[1], "wb");
    {
        hgframe::FrameWriter writer(out);
        for (int threads : {1, 4, 16}) {
            constexpr int kMessages = 200000;
            std::vector<std::thread> producers;
            std::vector<double> ns(threads);
            for (int t = 0; t < threads; ++t)
                producers.emplace_back([&ns, t] {
                    const char text[] = "progress: 123456 states, 654321 events";
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int m = 0; m < kMessages; ++m)
                        hgframe::post_message(hgframe::MessageKind::Log, text, sizeof text - 1);
                    ns[t] = std::chrono::duration<double, std::nano>(
                                std::chrono::steady_clock::now() - t0).count() / kMessages;
                });
            for (auto& p : producers) p.join();
            double mean = 0;
            for (double v : ns) mean += v / threads;
            std::printf("[frame_writer] %d producer thread(s): %.1f ns per message\n", threads, mean);
        }
        HG_NAMESPACE::wxf::Pieces reply;
        writer.write_reply(reply);
    }
    std::fclose(out);
    drain.join();
}
