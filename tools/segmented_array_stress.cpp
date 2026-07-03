// segmented_array_stress.cpp — concurrency regression guard for SegmentedArray::emplace.
//
// emplace() must publish a slot's segment (and, for the claiming thread, construct
// its element) before the reader-visible count_ passes that slot. If it does not, a
// reader iterating [0, size()) can dereference a not-yet-installed (null) segment and
// crash, or read an argument value that was never written back at its returned index.
//
// This test hammers a fresh array from many writer threads (small segments, so
// segment-boundary installs race constantly) while reader threads iterate by size(),
// then checks that every emplace-returned index holds exactly the value that emplace
// stored. A crash or any mismatch is a failure.
//
// Build:
//   g++ -O2 -std=c++20 -pthread -I hypergraph/include \
//       tools/segmented_array_stress.cpp -o /tmp/sa_stress && /tmp/sa_stress
#include "hypergraph/segmented_array.hpp"
#include "hypergraph/arena.hpp"
#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

using namespace hypergraph;

int main(int argc, char** argv) {
    // Small segments (64) make segment installs race on nearly every emplace; keep
    // total elements under MAX_SEGMENTS*segment_size (4096*64) so seg_idx stays in range.
    int writers = argc > 1 ? atoi(argv[1]) : 16;
    int per_writer = argc > 2 ? atoi(argv[2]) : 10000;
    int readers = argc > 3 ? atoi(argv[3]) : 4;
    int rounds = argc > 4 ? atoi(argv[4]) : 20;

    for (int r = 0; r < rounds; ++r) {
        ConcurrentHeterogeneousArena arena;
        // Small segment size so segment installs race on nearly every emplace.
        SegmentedArray<uint64_t> arr(64);

        // Each writer records (returned index -> value) so we can verify write-back.
        std::vector<std::vector<std::pair<uint32_t, uint64_t>>> recorded(writers);
        std::atomic<bool> go{false};
        std::atomic<bool> writing{true};

        std::vector<std::thread> ts;
        for (int w = 0; w < writers; ++w) {
            ts.emplace_back([&, w] {
                while (!go.load(std::memory_order_acquire)) {}
                recorded[w].reserve(per_writer);
                for (int i = 0; i < per_writer; ++i) {
                    uint64_t value = (uint64_t(w) << 40) | uint64_t(i);
                    uint32_t idx = arr.emplace(arena, value);
                    recorded[w].emplace_back(idx, value);
                }
            });
        }
        // Readers iterate by size() concurrently; must never fault. They may legally
        // observe a default (0) for an in-flight slot of another writer, so they only
        // assert the absence of a crash, not a value.
        std::atomic<uint64_t> sink{0};
        for (int rd = 0; rd < readers; ++rd) {
            ts.emplace_back([&] {
                while (!go.load(std::memory_order_acquire)) {}
                while (writing.load(std::memory_order_acquire)) {
                    uint32_t n = arr.size();
                    uint64_t acc = 0;
                    for (uint32_t i = 0; i < n; ++i) acc ^= arr[i];
                    sink.fetch_add(acc, std::memory_order_relaxed);
                }
            });
        }

        go.store(true, std::memory_order_release);
        for (int w = 0; w < writers; ++w) ts[w].join();
        writing.store(false, std::memory_order_release);
        for (int rd = writers; rd < writers + readers; ++rd) ts[rd].join();

        // Verify: every returned index holds exactly the value emplace stored, and the
        // published size accounts for every emplace.
        size_t total = 0, mismatches = 0;
        for (int w = 0; w < writers; ++w) {
            for (auto& [idx, value] : recorded[w]) {
                ++total;
                if (arr[idx] != value) ++mismatches;
            }
        }
        uint32_t expected_size = uint32_t(writers) * uint32_t(per_writer);
        if (mismatches != 0 || arr.size() != expected_size || total != expected_size) {
            std::printf("round %d FAIL: size=%u expected=%u mismatches=%zu (sink=%llu)\n",
                        r, arr.size(), expected_size, mismatches,
                        (unsigned long long)sink.load());
            return 1;
        }
    }
    std::printf("PASS: %d rounds, %d writers x %d emplaces, %d concurrent readers, all values correct, no fault\n",
                rounds, writers, per_writer, readers);
    return 0;
}
