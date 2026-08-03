#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>

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
template<typename Key = uint32_t, Key EMPTY_KEY = Key{0}>
class ConcurrentIdSet {
    static_assert(std::is_unsigned<Key>::value,
                  "the empty sentinel and the probe arithmetic both assume an unsigned key");
public:
    static constexpr double LOAD_FACTOR_THRESHOLD = 0.75;

    struct Table {
        std::atomic<Key>* keys;
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

            size_t bytes = sizeof(Table) + sizeof(std::atomic<Key>) * actual_cap;
            void* mem = arena ? arena->allocate_raw(bytes, alignof(std::max_align_t))
                              : ::operator new(bytes);
            Table* table = static_cast<Table*>(mem);
            table->keys = reinterpret_cast<std::atomic<Key>*>(
                static_cast<char*>(mem) + sizeof(Table));
            table->capacity = actual_cap;
            table->mask = actual_cap - 1;
            table->prev = prev_table;

            for (size_t i = 0; i < actual_cap; ++i) {
                new (&table->keys[i]) std::atomic<Key>(EMPTY_KEY);
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
    bool insert(Key key) {
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

    bool contains(Key key) const {
        return contains_in_chain(table_.load(std::memory_order_acquire), key);
    }

    // Approximate element count (may overcount if the same key is inserted
    // concurrently during a resize window); used only for capacity hints.
    size_t size() const { return count_.load(std::memory_order_relaxed); }

    // Visits each key ONCE, though a key may sit in several tables at the same time.
    //
    // resize() COPIES into the new table and leaves the old one intact and reachable, because a
    // straggler still writing into a retired table has to remain findable. So a key that survived
    // r resizes occupies r+1 slots across the chain, and a walk that just concatenates the tables
    // emits it r+1 times. Skipping a key that a NEWER table already carries is what makes the walk
    // a set walk. (ConcurrentMap::for_each carries the same guard for the same reason; this is one
    // rule and it was previously written twice, correctly in the used copy and not here.)
    template<typename F>
    void for_each(F&& f) const {
        Table* head = table_.load(std::memory_order_acquire);
        for (Table* t = head; t; t = t->prev) {
            for (size_t i = 0; i < t->capacity; ++i) {
                Key k = t->keys[i].load(std::memory_order_acquire);
                if (k == EMPTY_KEY) continue;
                if (t != head && contains_in_newer(head, t, k)) continue;   // already emitted
                f(k);
            }
        }
    }

private:
    static size_t hash(Key key) {
        uint64_t h = key;
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33;
        return static_cast<size_t>(h);
    }

    bool insert_into_table(Table* table, Key key, bool increment_count) {
        size_t idx = hash(key) & table->mask;
        for (size_t probe = 0; probe < table->capacity; ++probe) {
            size_t i = (idx + probe) & table->mask;
            std::atomic<Key>& slot = table->keys[i];

            Key cur = slot.load(std::memory_order_acquire);
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

    // Is `key` in any table strictly NEWER than `stop`? Walks head..stop exclusive, so it answers
    // "has an earlier iteration of for_each already emitted this".
    bool contains_in_newer(Table* head, Table* stop, Key key) const {
        for (Table* t = head; t && t != stop; t = t->prev) {
            size_t idx = hash(key) & t->mask;
            for (size_t probe = 0; probe < t->capacity; ++probe) {
                size_t i = (idx + probe) & t->mask;
                Key cur = t->keys[i].load(std::memory_order_acquire);
                if (cur == key) return true;
                if (cur == EMPTY_KEY) break;   // open addressing: absent in this table
            }
        }
        return false;
    }

    bool contains_in_chain(Table* table, Key key) const {
        while (table) {
            size_t idx = hash(key) & table->mask;
            for (size_t probe = 0; probe < table->capacity; ++probe) {
                size_t i = (idx + probe) & table->mask;
                Key cur = table->keys[i].load(std::memory_order_acquire);
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
            Key k = old_table->keys[i].load(std::memory_order_acquire);
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
