// GENMC-ARGS: --disable-estimation
// GENMC-CALIBRATE: -DCALIBRATE_INSTALL_BY_STORE
// GENMC-CALIBRATE: -DCALIBRATE_RELAXED_LINK
//
// GenMC harness: an instance's claim chain (hgcommon::qr_claim_chain) claims each pair exactly
// once while two threads install the blocks past the first one.
//
// THE PROTOCOL. A quotient instance carries a chain of claim blocks, one bit per class match in
// per-class match order. A match past the chain's end installs the next block by one
// compare-and-swap on its predecessor's next link; a claimer that loses the swap gives its block
// back and continues in the winner's. A block never changes after it is installed, so the two
// sides of the instance/match rendezvous reach the same bit, and the bit's fetch_or decides the
// pair. The walk is the shared core both engines run; the context is the host's
// (Hypergraph::QrCtx::claim's Chain in hypergraph/src/hypergraph.cpp): the same block layout,
// next loaded acquire, installed acq_rel/acquire, bits set by acq_rel fetch_or.
//
// THE SHAPE. One instance whose first block holds one word (matches 0..63). Thread A claims
// matches 64 and 200; thread B claims 200 and 64. Match 64 needs a second block and match 200 a
// block past it, so both threads install, in opposite orders, and race on both links.
//
// THE PROPERTY. For each of the two matches exactly one thread wins, and the chain afterwards is
// the first block followed by the installed blocks, each reachable once.
//
// CALIBRATED. -DCALIBRATE_INSTALL_BY_STORE links a new block with a plain store: both threads
// install, one overwrites the other's link, and a pair is won twice. -DCALIBRATE_RELAXED_LINK
// installs and loads the link relaxed, so a thread reaching another's block does not see its
// zeroed words: the checker reports the unordered access.
//
// The size of a new block (at least the words reaching the match and the words of the block
// before) is not part of the property: each claimer continues in the block that won the link,
// so the bit a match maps to depends only on the installed chain, which both sides walk.
#include "genmc_support.hpp"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <new>
#include <pthread.h>

#include "hgcommon/quotient_replay_core.hpp"

namespace {

struct Block {
    std::atomic<Block*> next{nullptr};
    uint32_t words = 0;
    std::atomic<uint64_t>* bits() { return reinterpret_cast<std::atomic<uint64_t>*>(this + 1); }
};

Block* new_block(uint32_t words) {
    void* raw = ::operator new(sizeof(Block) + words * sizeof(uint64_t));
    auto* b = new (raw) Block;
    b->words = words;
    for (uint32_t i = 0; i < words; ++i) new (&b->bits()[i]) std::atomic<uint64_t>(0);
    return b;
}

struct Chain {
    using Block = ::Block*;
    bool is_null(Block b) const { return b == nullptr; }
    uint32_t words(Block b) const { return b->words; }
#if defined(CALIBRATE_RELAXED_LINK)
    static constexpr std::memory_order kLoad = std::memory_order_relaxed;
    static constexpr std::memory_order kSwap = std::memory_order_relaxed;
#else
    static constexpr std::memory_order kLoad = std::memory_order_acquire;
    static constexpr std::memory_order kSwap = std::memory_order_acq_rel;
#endif
    Block next(Block b) const { return b->next.load(kLoad); }
    Block install_next(Block b, uint32_t words) {
        Block made = new_block(words);
#if defined(CALIBRATE_INSTALL_BY_STORE)
        b->next.store(made, std::memory_order_release);
        return made;
#else
        Block expected = nullptr;
        if (b->next.compare_exchange_strong(expected, made, kSwap, kLoad))
            return made;
        ::operator delete(made);
        return expected;
#endif
    }
    bool set_bit(Block b, uint32_t bit) {
        const uint64_t mask = uint64_t{1} << (bit & 63u);
        return (b->bits()[bit >> 6].fetch_or(mask, std::memory_order_acq_rel) & mask) == 0;
    }
};

Block* g_first;
int g_won[2][2];   // [thread][0: match 64, 1: match 200]

bool claim(uint32_t local) {
    Chain c;
    return hgcommon::qr_claim_chain(c, g_first, local) == hgcommon::QR_CLAIM_WON;
}

void* thread_a(void*) {
    g_won[0][0] = claim(64);
    g_won[0][1] = claim(200);
    return nullptr;
}

void* thread_b(void*) {
    g_won[1][1] = claim(200);
    g_won[1][0] = claim(64);
    return nullptr;
}

}  // namespace

int main() {
    g_first = new_block(hgcommon::qr_claim_words(1));
    pthread_t a, b;
    pthread_create(&a, nullptr, thread_a, nullptr);
    pthread_create(&b, nullptr, thread_b, nullptr);
    pthread_join(a, nullptr);
    pthread_join(b, nullptr);
    assert(g_won[0][0] + g_won[1][0] == 1 && "match 64 was not claimed exactly once");
    assert(g_won[0][1] + g_won[1][1] == 1 && "match 200 was not claimed exactly once");
    // The bits of both matches are set in the chain a later walk reaches.
    Chain c;
    assert(hgcommon::qr_claim_chain(c, g_first, 64) == hgcommon::QR_CLAIM_LOST);
    assert(hgcommon::qr_claim_chain(c, g_first, 200) == hgcommon::QR_CLAIM_LOST);
    return 0;
}
