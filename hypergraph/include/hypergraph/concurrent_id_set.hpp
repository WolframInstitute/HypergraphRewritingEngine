#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>

#include "arena.hpp"

namespace hypergraph {

// =============================================================================
// ConcurrentIdSet: lock-free, append-only, open-addressed SET of uint32 keys
// =============================================================================
//
// Key-only: one atomic<uint32_t> per slot (4 bytes), half the footprint of a
// ConcurrentMap<uint32_t, _> whose value word pads each entry to 8 bytes. Used for
// the causal descendant closures, where the pair count dominates memory.
//
// Publication is a single CAS EMPTY_KEY -> key, so a slot is only ever observed as
// EMPTY_KEY or a final key -- there is no intermediate LOCKED state and therefore no
// ambiguous in-flight slot. insert() is wait-free over its probe sequence: it never
// spins, and because a key is published atomically, linear probing keeps every key in
// at most one slot per table (a probing writer that finds a non-empty slot has seen
// that slot's final key, so it cannot deposit a second copy of the same key later in
// the sequence).
//
// Resize keeps the same superseded-table chain discipline as ConcurrentMap: old
// tables are never freed (arena-reclaimed in bulk) and stay reachable via `prev`, so
// contains() and insert() walk the chain and a straggler writing into a retired table
// is still found.
//
// EMPTY_KEY (0 by default) is reserved and must never be inserted: callers offset real
// ids by +1 so id 0 does not collide with the empty sentinel.
template<uint32_t EMPTY_KEY = 0u>
class ConcurrentIdSet {
public:
    static constexpr double LOAD_FACTOR_THRESHOLD = 0.75;

    struct Table {
        std::atomic<uint32_t>* keys;
        size_t capacity;
        size_t mask;  // capacity - 1, for fast modulo
        Table* prev;  // superseded table (resize chain)

        // arena != nullptr: allocated from the arena (no malloc), never individually
        // freed (bulk-reclaimed with the arena). arena == nullptr: ::operator new,
        // freed in the destructor (standalone/test use).
        static Table* create(size_t cap, Table* prev_table,
                             ConcurrentHeterogeneousArena* arena) {
            size_t actual_cap = 1;
            while (actual_cap < cap) actual_cap <<= 1;

            size_t bytes = sizeof(Table) + sizeof(std::atomic<uint32_t>) * actual_cap;
            void* mem = arena ? arena->allocate_raw(bytes, alignof(std::max_align_t))
                              : ::operator new(bytes);
            Table* table = static_cast<Table*>(mem);
            table->keys = reinterpret_cast<std::atomic<uint32_t>*>(
                static_cast<char*>(mem) + sizeof(Table));
            table->capacity = actual_cap;
            table->mask = actual_cap - 1;
            table->prev = prev_table;

            for (size_t i = 0; i < actual_cap; ++i) {
                new (&table->keys[i]) std::atomic<uint32_t>(EMPTY_KEY);
            }
            return table;
        }
    };

    explicit ConcurrentIdSet(size_t initial_capacity = 16,
                             ConcurrentHeterogeneousArena* arena = nullptr)
        : count_(0), arena_(arena) {
        table_.store(Table::create(initial_capacity, nullptr, arena_),
                     std::memory_order_release);
    }

    ~ConcurrentIdSet() {
        if (arena_) return;  // arena-backed tables are bulk-reclaimed
        Table* t = table_.load(std::memory_order_acquire);
        while (t) {
            Table* prev = t->prev;
            ::operator delete(t);
            t = prev;
        }
    }

    ConcurrentIdSet(const ConcurrentIdSet&) = delete;
    ConcurrentIdSet& operator=(const ConcurrentIdSet&) = delete;
    ConcurrentIdSet(ConcurrentIdSet&&) = delete;
    ConcurrentIdSet& operator=(ConcurrentIdSet&&) = delete;

    // Insert key. Returns true iff newly inserted, false if already present.
    bool insert(uint32_t key) {
        Table* table = table_.load(std::memory_order_acquire);
        size_t current_count = count_.load(std::memory_order_relaxed);
        if (current_count > table->capacity * LOAD_FACTOR_THRESHOLD) {
            resize();
            table = table_.load(std::memory_order_acquire);
        }

        // A key that lives in a superseded table must not be duplicated into the
        // current one: check the chain below the current table first.
        if (table->prev && contains_in_chain(table->prev, key)) {
            return false;
        }
        return insert_into_table(table, key, true);
    }

    bool contains(uint32_t key) const {
        return contains_in_chain(table_.load(std::memory_order_acquire), key);
    }

    // Approximate element count (may overcount if the same key is inserted
    // concurrently during a resize window); used only for capacity hints.
    size_t size() const { return count_.load(std::memory_order_relaxed); }

    template<typename F>
    void for_each(F&& f) const {
        Table* table = table_.load(std::memory_order_acquire);
        while (table) {
            for (size_t i = 0; i < table->capacity; ++i) {
                uint32_t k = table->keys[i].load(std::memory_order_acquire);
                if (k != EMPTY_KEY) f(k);
            }
            table = table->prev;
        }
    }

private:
    static size_t hash(uint32_t key) {
        uint64_t h = key;
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33;
        return static_cast<size_t>(h);
    }

    bool insert_into_table(Table* table, uint32_t key, bool increment_count) {
        size_t idx = hash(key) & table->mask;
        for (size_t probe = 0; probe < table->capacity; ++probe) {
            size_t i = (idx + probe) & table->mask;
            std::atomic<uint32_t>& slot = table->keys[i];

            uint32_t cur = slot.load(std::memory_order_acquire);
            if (cur == key) return false;
            if (cur == EMPTY_KEY) {
                if (slot.compare_exchange_strong(cur, key,
                        std::memory_order_acq_rel, std::memory_order_acquire)) {
                    if (increment_count) count_.fetch_add(1, std::memory_order_relaxed);
                    return true;
                }
                // CAS failed: `cur` now holds the winner. If it is our key we are done,
                // otherwise a different key claimed this slot -- keep probing.
                if (cur == key) return false;
            }
            // Slot holds a different key: continue probing.
        }
        // Table full (load factor should prevent this): grow and retry from the top.
        resize();
        return insert(key);
    }

    bool contains_in_chain(Table* table, uint32_t key) const {
        while (table) {
            size_t idx = hash(key) & table->mask;
            for (size_t probe = 0; probe < table->capacity; ++probe) {
                size_t i = (idx + probe) & table->mask;
                uint32_t cur = table->keys[i].load(std::memory_order_acquire);
                if (cur == key) return true;
                if (cur == EMPTY_KEY) break;  // open addressing: absent in this table
            }
            table = table->prev;
        }
        return false;
    }

    void resize() {
        Table* old_table = table_.load(std::memory_order_acquire);
        size_t new_capacity = old_table->capacity * 2;
        Table* new_table = Table::create(new_capacity, old_table, arena_);

        for (size_t i = 0; i < old_table->capacity; ++i) {
            uint32_t k = old_table->keys[i].load(std::memory_order_acquire);
            if (k != EMPTY_KEY) insert_into_table(new_table, k, false);
        }

        if (!table_.compare_exchange_strong(old_table, new_table,
                std::memory_order_release, std::memory_order_acquire)) {
            // Another thread installed first. A heap loser is freed; an arena loser is
            // reclaimed in bulk with the arena.
            if (!arena_) ::operator delete(new_table);
        }
    }

    std::atomic<Table*> table_;
    std::atomic<size_t> count_;
    ConcurrentHeterogeneousArena* arena_ = nullptr;
};

}  // namespace hypergraph
