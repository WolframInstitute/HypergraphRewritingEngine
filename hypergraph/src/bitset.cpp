#include "hypergraph/bitset.hpp"

// The bodies behind bitset.hpp. `contains` and `find_chunk` are NOT here: they are
// HG_INLINE in the header with the measurement that pins them there, recorded at
// `contains`. Everything the templates need (set, clear, derive, for_each, the arena
// helpers) is a template and stays in the header as well.

namespace HG_NAMESPACE {
namespace engine {

// =============================================================================
// SparseBitset::Chunk
// =============================================================================

SparseBitset::Chunk::Chunk() {
    std::memset(words, 0, sizeof(words));
}

bool SparseBitset::Chunk::get(size_t bit_index) const {
    size_t word_idx = bit_index / 64;
    size_t bit_idx = bit_index % 64;
    return (words[word_idx] >> bit_idx) & 1;
}

void SparseBitset::Chunk::set(size_t bit_index) {
    size_t word_idx = bit_index / 64;
    size_t bit_idx = bit_index % 64;
    words[word_idx] |= (1ULL << bit_idx);
}

void SparseBitset::Chunk::clear(size_t bit_index) {
    size_t word_idx = bit_index / 64;
    size_t bit_idx = bit_index % 64;
    words[word_idx] &= ~(1ULL << bit_idx);
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

// =============================================================================
// SparseBitset
// =============================================================================

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
    // Clear the source to prevent aliasing
    other.entries_ = nullptr;
    other.num_entries_ = 0;
    other.capacity_ = 0;
    other.count_cached_.store(0, std::memory_order_relaxed);
    other.count_valid_.store(true, std::memory_order_relaxed);
}

SparseBitset& SparseBitset::operator=(SparseBitset&& other) noexcept {
    if (this != &other) {
        // Take over other's data
        entries_ = other.entries_;
        num_entries_ = other.num_entries_;
        capacity_ = other.capacity_;
        count_cached_.store(other.count_cached_.load(std::memory_order_relaxed),
                            std::memory_order_relaxed);
        count_valid_.store(other.count_valid_.load(std::memory_order_relaxed),
                           std::memory_order_relaxed);

        // Clear the source to prevent aliasing
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
    for (size_t i = 0; i < num_entries_; ++i) {
        total += entries_[i].chunk->popcount();
    }
    count_cached_.store(total, std::memory_order_relaxed);
    count_valid_.store(true, std::memory_order_release);
    return total;
}

bool SparseBitset::empty() const {
    return count() == 0;
}

size_t SparseBitset::num_chunks() const {
    return num_entries_;
}

bool SparseBitset::find_entry_index(uint32_t chunk_id, size_t& out_idx) const {
    size_t lo = 0, hi = num_entries_;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (entries_[mid].chunk_id < chunk_id) lo = mid + 1; else hi = mid;
    }
    out_idx = lo;
    return lo < num_entries_ && entries_[lo].chunk_id == chunk_id;
}

void SparseBitset::invalidate_count() {
    count_valid_.store(false, std::memory_order_relaxed);
}

}  // namespace engine
}  // namespace HG_NAMESPACE
