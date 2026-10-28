#include "hypergraph/bitset.hpp"

// The bodies behind bitset.hpp. `contains` and `find_chunk` are NOT here: they are
// HG_INLINE in the header with the measurement that pins them there, recorded at
// `contains`. Everything the templates need (set, clear, derive, for_each, the arena
// helpers) is a template and stays in the header as well.

namespace HG_NAMESPACE {
namespace engine {

SparseBitset::Chunk::Chunk() {
    std::memset(words, 0, sizeof(words));
}

bool SparseBitset::Chunk::empty() const {
    for (size_t i = 0; i < WORDS_PER_CHUNK; ++i) {
        if (words[i] != 0) return false;
    }
    return true;
}

size_t SparseBitset::Chunk::popcount() const {
    size_t count = 0;
    for (size_t i = 0; i < WORDS_PER_CHUNK; ++i) {
        count += hgcommon::popcount64(words[i]);
    }
    return count;
}

SparseBitset::SparseBitset()
    : entries_(nullptr)
    , num_entries_(0)
    , capacity_(0)
    , count_cached_(0)
    , count_valid_(true)
{}

SparseBitset::SparseBitset(SparseBitset&& other) noexcept
    : entries_(other.entries_)
    , num_entries_(other.num_entries_)
    , capacity_(other.capacity_)
    , count_cached_(other.count_cached_.load(std::memory_order_relaxed))
    , count_valid_(other.count_valid_.load(std::memory_order_relaxed))
{
    other.entries_ = nullptr;
    other.num_entries_ = 0;
    other.capacity_ = 0;
    other.count_cached_.store(0, std::memory_order_relaxed);
    other.count_valid_.store(true, std::memory_order_relaxed);
}

SparseBitset& SparseBitset::operator=(SparseBitset&& other) noexcept {
    if (this != &other) {
        entries_ = other.entries_;
        num_entries_ = other.num_entries_;
        capacity_ = other.capacity_;
        count_cached_.store(other.count_cached_.load(std::memory_order_relaxed),
                            std::memory_order_relaxed);
        count_valid_.store(other.count_valid_.load(std::memory_order_relaxed),
                           std::memory_order_relaxed);
        other.entries_ = nullptr;
        other.num_entries_ = 0;
        other.capacity_ = 0;
        other.count_cached_.store(0, std::memory_order_relaxed);
        other.count_valid_.store(true, std::memory_order_relaxed);
    }
    return *this;
}

size_t SparseBitset::count() const {
    if (count_valid_.load(std::memory_order_acquire)) {
        return count_cached_.load(std::memory_order_relaxed);
    }
    size_t total = 0;
    for_each_chunk([&](uint32_t, const Chunk& c) { total += c.popcount(); });
    count_cached_.store(total, std::memory_order_relaxed);
    count_valid_.store(true, std::memory_order_release);
    return total;
}

bool SparseBitset::empty() const {
    return count() == 0;
}

const SparseBitset::Chunk* SparseBitset::find_chunk_radix(uint32_t chunk_id) const {
    if ((uint64_t(chunk_id) >> (FANOUT_SHIFT * height_)) != 0) return nullptr;
    uintptr_t p = root_ & ~uintptr_t{1};
    for (uint32_t h = height_; h > 0 && p; --h)
        p = reinterpret_cast<const Node*>(p)->slot[(chunk_id >> (FANOUT_SHIFT * (h - 1))) & (FANOUT - 1)] &
            ~uintptr_t{1};
    return reinterpret_cast<const Chunk*>(p);
}

size_t SparseBitset::num_chunks() const {
    size_t n = 0;
    for_each_chunk([&](uint32_t, const Chunk&) { ++n; });
    return n;
}

}  // namespace engine
}  // namespace HG_NAMESPACE
