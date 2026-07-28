#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include "hgcommon/portable_intrinsics.hpp"
#include "arena.hpp"
#include "debug_log.hpp"
#include <cstdio>
#include <cstdlib>
#include <new>
#include <optional>
#include <type_traits>
#include <stdexcept>
#include <string>
#include <utility>

namespace hypergraph {

// =============================================================================
// ConcurrentMap<K, V>: Lock-free hash map with open addressing
// =============================================================================
//
// Append-only concurrent hash map using open addressing with linear probing.
// Optimized for the case where we never delete entries.
//
// Key requirements:
//   - K must be trivially copyable
//   - K must reserve EMPTY_KEY (default 0) to mean "slot is available"
//   - V must reserve ABSENT_VALUE (default V{}, i.e. nullptr or false) to mean "not yet
//     published". A map whose values legitimately include V{} must name a different one.
//   - A key equal to a reserved sentinel is rejected (see reject_sentinel_key)
//   - Good hash distribution expected from caller
//
// Design: two independent publications, each a single compare-exchange.
//   - The KEY goes straight from EMPTY_KEY to its final value; there is no intermediate
//     state, so a slot never has to be re-examined to find out what it holds.
//   - The VALUE is published by exchanging it from ABSENT_VALUE. A thread that finds its
//     key already present but the value not yet published offers its OWN value to that same
//     exchange rather than waiting for whoever claimed the key. Exactly one offer wins, and
//     every caller returns the winner -- so was_inserted means "the value you passed is the
//     one now stored", which is exactly what the get-or-create callers need to decide
//     whether to keep the object they built.
//
// Progress guarantee: NO OPERATION WAITS ON ANOTHER THREAD. Lookup is wait-free. Insert is
// lock-free: every exchange that fails does so because another thread published, and a
// descheduled thread holding no claim cannot stall anyone, because there is no claim to hold.
//
// Resize: When load factor exceeds threshold, allocates new larger table.
//         Old table remains valid, lookups check both via chain.
//

template<typename K, typename V, K EMPTY_KEY = K{0}, K LOCKED_KEY = K{~0ULL},
         V ABSENT_VALUE = V{}>
class ConcurrentMap {
public:
    static constexpr size_t DEFAULT_INITIAL_CAPACITY = 1024;
    static constexpr double LOAD_FACTOR_THRESHOLD = 0.75;

    struct Entry {
        std::atomic<K> key;
        std::atomic<V> value;

        Entry() : key(EMPTY_KEY), value(ABSENT_VALUE) {}
    };

    struct Table {
        Entry* entries;
        size_t capacity;
        size_t mask;  // capacity - 1, for fast modulo
        Table* prev;  // Previous table (for resize chain)

        // When arena != nullptr the table is allocated from the arena (no malloc, no
        // per-map heap contention) and is NEVER individually freed — it is reclaimed in
        // bulk when the arena is. When arena == nullptr it falls back to ::operator new
        // (freed in the destructor), for standalone/test use.
        static Table* create(size_t cap, Table* prev_table,
                             ConcurrentHeterogeneousArena* arena) {
            // Capacity must be power of 2
            size_t actual_cap = 1;
            while (actual_cap < cap) actual_cap <<= 1;

            size_t bytes = sizeof(Table) + sizeof(Entry) * actual_cap;
            void* mem = arena ? arena->allocate_raw(bytes, alignof(std::max_align_t))
                              : ::operator new(bytes);
            Table* table = static_cast<Table*>(mem);
            table->entries = reinterpret_cast<Entry*>(
                static_cast<char*>(mem) + sizeof(Table));
            table->capacity = actual_cap;
            table->mask = actual_cap - 1;
            table->prev = prev_table;

            // Initialize entries
            for (size_t i = 0; i < actual_cap; ++i) {
                new (&table->entries[i]) Entry();
            }

            return table;
        }
    };

    // arena != nullptr routes all table allocation through the arena (no malloc); the
    // tables are then reclaimed in bulk with the arena, not in this destructor.
    explicit ConcurrentMap(size_t initial_capacity = DEFAULT_INITIAL_CAPACITY,
                           ConcurrentHeterogeneousArena* arena = nullptr)
        : count_(0), arena_(arena) {
        table_.store(Table::create(initial_capacity, nullptr, arena_), std::memory_order_release);
    }

    ~ConcurrentMap() {
        // Arena-backed tables are owned by the arena (bulk-freed); nothing to do.
        if (arena_) return;
        // Free all heap tables in the chain.
        Table* t = table_.load(std::memory_order_acquire);
        while (t) {
            Table* prev = t->prev;
            ::operator delete(t);
            t = prev;
        }
    }

    // Re-home a still-empty map onto an arena: swap the fresh heap table for an
    // arena-allocated one, so all future table storage (initial + resizes) comes
    // from the arena and is bulk-reclaimed with it. For a map that is a class member
    // constructed before its arena is known. MUST be called single-threaded during
    // setup, before any insert, on a map constructed without an arena.
    void set_arena(ConcurrentHeterogeneousArena* arena) {
        arena_ = arena;
        Table* old = table_.load(std::memory_order_relaxed);
        size_t cap = old ? old->capacity : DEFAULT_INITIAL_CAPACITY;
        table_.store(Table::create(cap, nullptr, arena_), std::memory_order_release);
        // The pre-rehome table(s) were heap-backed; free them.
        while (old) {
            Table* prev = old->prev;
            ::operator delete(old);
            old = prev;
        }
    }

    // Total heap bytes held by the table chain (current + retained superseded tables),
    // for memory measurement. O(chain length).
    size_t bytes_allocated() const {
        size_t total = 0;
        for (Table* t = table_.load(std::memory_order_acquire); t; t = t->prev) {
            total += sizeof(Table) + sizeof(Entry) * t->capacity;
        }
        return total;
    }

    // Non-copyable, non-movable
    ConcurrentMap(const ConcurrentMap&) = delete;
    ConcurrentMap& operator=(const ConcurrentMap&) = delete;
    ConcurrentMap(ConcurrentMap&&) = delete;
    ConcurrentMap& operator=(ConcurrentMap&&) = delete;

    // A key equal to a sentinel cannot be stored: the slot would read as empty (or as
    // mid-write) and the entry would silently never exist. That is a programmer error in the
    // caller's key domain -- a dense id starting at 0, or a hash that came out 0 -- so it is
    // reported rather than absorbed. FOUR separate correctness bugs in this engine were this
    // exact silent no-op (an id-0 state undercount, a genesis edge id 0 crash, an orbit table
    // for state 0 that could never be found, and a causal self-loop on event 0), each of which
    // took a long investigation because nothing failed at the point of the mistake. Callers
    // offset dense ids by +1 or move the sentinels into a reserved band.
    //
    // The operation name goes in the thrown message, not only in DEBUG_LOG: that macro
    // compiles away unless debug logging is on, and this throw is most likely to be seen in a
    // release build, where knowing which call site tripped it is most of the diagnosis.
    static void reject_sentinel_key(K key, const char* op) {
        if (key == EMPTY_KEY || key == LOCKED_KEY) {
            DEBUG_LOG("ConcurrentMap::%s called with a reserved sentinel key", op);
            throw std::logic_error(
                std::string("ConcurrentMap::") + op +
                ": key collides with a reserved sentinel (EMPTY/LOCKED). "
                "Offset dense ids by +1 or use a reserved sentinel band.");
        }
    }

    // Insert key-value pair if key doesn't exist
    // Returns: (value, was_inserted)
    //   - If key was new: returns (value, true)
    //   - If key existed: returns (existing_value, false)
    std::pair<V, bool> insert_if_absent(K key, V value) {
        reject_sentinel_key(key, "insert_if_absent");
        // Check if we need to resize
        Table* table = table_.load(std::memory_order_acquire);
        size_t current_count = count_.load(std::memory_order_relaxed);
        if (current_count > table->capacity * LOAD_FACTOR_THRESHOLD) {
            resize();
            table = table_.load(std::memory_order_acquire);
        }

        // Check superseded tables before inserting. An entry can live only in one of them --
        // an insert that resolved against a table after its slot had been rehashed completes
        // there -- and inserting again would give one key two entries, which for the
        // get-or-create callers means two container objects and a silently split rendezvous.
        //
        // The scan SETTLES rather than merely looks. A plain lookup reports a key whose value
        // is not yet published as absent, which is the right answer for a reader and the wrong
        // one here: this caller is holding a value to offer, so it completes that entry
        // instead of walking past it and creating a rival. Offering a value is the same
        // exchange every publisher uses, so this closes the window without anyone waiting.
        if (table->prev) {
            auto settled = find_and_settle_in_chain(table->prev, key, value);
            if (settled.has_value()) return *settled;
        }

        return insert_into_table(table, key, value, true);
    }

    // Retained spelling of insert_if_absent. Nothing is ever in a state a caller could
    // usefully wait out: a key is published in one exchange, and an unpublished value is
    // settled by offering one rather than by waiting.
    std::pair<V, bool> insert_if_absent_waiting(K key, V value) {
        return insert_if_absent(key, value);
    }

    // Lookup value by key
    std::optional<V> lookup(K key) const {
        Table* table = table_.load(std::memory_order_acquire);
        return lookup_in_chain(table, key);
    }

    // Retained spelling of lookup, for the same reason as insert_if_absent_waiting.
    std::optional<V> lookup_waiting(K key) const { return lookup(key); }

    // Check if key exists
    bool contains(K key) const {
        return lookup(key).has_value();
    }

    // Get value or default
    V get_or_default(K key, V default_value) const {
        auto result = lookup(key);
        return result.has_value() ? *result : default_value;
    }

    // Current count (approximate, may be slightly off during concurrent inserts)
    // WARNING: Due to race conditions with LOCKED slots, this may over-count
    // when the same key is inserted concurrently by multiple threads.
    // Use count_unique() for accurate counts when no concurrent inserts are happening.
    size_t size() const {
        return count_.load(std::memory_order_relaxed);
    }

    // Count actual unique keys in the map (accurate, O(n) scan)
    // Use this after all inserts are complete for accurate counts.
    // This handles the case where duplicate keys exist due to concurrent inserts.
    // Distinct live keys across the chain. Allocation-free: an entry in a superseded table is
    // counted only if no newer table carries the same key, which is the same one-probe test
    // for_each uses -- no seen-keys set, so no heap and no per-entry hashing.
    size_t count_unique() const {
        size_t unique_count = 0;
        Table* head = table_.load(std::memory_order_acquire);
        for (Table* t = head; t; t = t->prev) {
            for (size_t i = 0; i < t->capacity; ++i) {
                const K key = t->entries[i].key.load(std::memory_order_acquire);
                if (key == EMPTY_KEY) continue;
                if (t->entries[i].value.load(std::memory_order_acquire) == ABSENT_VALUE) continue;
                if (t != head && contains_in_newer(head, t, key)) continue;
                ++unique_count;
            }
        }
        return unique_count;
    }

    bool empty() const {
        return size() == 0;
    }

    // Iterate over all live entries exactly once (not thread-safe during inserts).
    //
    // The superseded chain must be walked: resize() skips LOCKED slots, so an insert that was
    // mid-flight when the table was replaced completes into the OLD table and lives only
    // there. But resize() also rehashes every settled entry into the new table, so a naive
    // chain walk yields those twice -- measured at 22,154 visits for a map holding 10,630
    // entries, and *which* ones doubled depended on when resizes fired, so a caller that
    // fingerprinted the iteration saw thread-dependent results from identical data.
    //
    // So: walk the chain, and emit an entry only if no NEWER table already carries that key.
    // The probe is O(1) and allocation-free -- superseded tables halve in size going back, so
    // the extra probing is bounded by roughly the current table's size.
    template<typename F>
    void for_each(F&& f) const {
        Table* head = table_.load(std::memory_order_acquire);
        for (Table* t = head; t; t = t->prev) {
            for (size_t i = 0; i < t->capacity; ++i) {
                const K key = t->entries[i].key.load(std::memory_order_acquire);
                if (key == EMPTY_KEY) continue;
                const V v = t->entries[i].value.load(std::memory_order_acquire);
                if (v == ABSENT_VALUE) continue;
                if (t != head && contains_in_newer(head, t, key)) continue;   // already emitted
                f(key, v);
            }
        }
    }

private:
    // Hash function - simple but effective for well-distributed keys
    // Caller is expected to provide good hash (e.g., for canonical_hash)
    static size_t hash(K key) {
        // FNV-1a style mixing for integer keys
        uint64_t h = static_cast<uint64_t>(key);
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33;
        return static_cast<size_t>(h);
    }

    // Insert into one table's probe run.
    //
    // Two exchanges, neither of which can leave another thread waiting. The key exchange
    // takes the slot straight from EMPTY_KEY to its final value, so a slot is never in a
    // state that has to be re-read to interpret. The value exchange decides, among every
    // thread offering a value for this key, which one is stored -- the thread that claimed
    // the key has no special standing, so a thread that finds the value still unpublished
    // simply offers its own instead of waiting to be told.
    std::pair<V, bool> insert_into_table(Table* table, K key, V value, bool increment_count) {
        const size_t start = hash(key) & table->mask;

        for (size_t probe = 0; probe < table->capacity; ++probe) {
            Entry& entry = table->entries[(start + probe) & table->mask];
            K current = entry.key.load(std::memory_order_acquire);

            if (current == EMPTY_KEY) {
                if (entry.key.compare_exchange_strong(current, key,
                                                      std::memory_order_acq_rel,
                                                      std::memory_order_acquire)) {
                    if (increment_count) count_.fetch_add(1, std::memory_order_relaxed);
                    return publish_value(entry, value);
                }
                // Lost the slot; `current` now holds whichever key won it.
            }

            if (current == key) return publish_value(entry, value);
            // Different key: keep probing.
        }

        // Table full (should not happen under the load factor). Grow and retry at this level,
        // preserving increment_count: re-entering through insert_if_absent would force
        // counting on, double-counting when the caller is resize()'s rehash.
        resize();
        return insert_into_table(table_.load(std::memory_order_acquire), key, value,
                                 increment_count);
    }

    // Settle this entry's value. Returns the stored value and whether it is the caller's.
    std::pair<V, bool> publish_value(Entry& entry, V value) {
        // A stored value equal to ABSENT_VALUE would read as "not published yet", so the
        // entry would be invisible to every lookup -- the same silent-disappearance the key
        // sentinels caused four times over, moved to the other field. Report it instead.
        if (value == ABSENT_VALUE) {
            throw std::logic_error(
                "ConcurrentMap: stored value collides with ABSENT_VALUE, so the entry would "
                "read as unpublished. Name a different ABSENT_VALUE for this map.");
        }

        V current = entry.value.load(std::memory_order_acquire);
        if (current != ABSENT_VALUE) return {current, false};

        if (entry.value.compare_exchange_strong(current, value,
                                                std::memory_order_acq_rel,
                                                std::memory_order_acquire)) {
            return {value, true};
        }
        return {current, false};   // another thread's value won; `current` holds it
    }

    // Find `key` anywhere in the chain and ensure its value is settled, offering `value` if it
    // is not. Returns the settled value and whether it is the caller's, or nullopt if the key
    // is absent from every table.
    std::optional<std::pair<V, bool>> find_and_settle_in_chain(Table* table, K key, V value) {
        while (table) {
            const size_t start = hash(key) & table->mask;
            for (size_t probe = 0; probe < table->capacity; ++probe) {
                Entry& entry = table->entries[(start + probe) & table->mask];
                const K current = entry.key.load(std::memory_order_acquire);
                if (current == EMPTY_KEY) break;          // not in THIS table; try the next
                if (current == key) return publish_value(entry, value);
            }
            table = table->prev;
        }
        return std::nullopt;
    }

    std::optional<V> lookup_in_chain(Table* table, K key) const {
        while (table) {
            auto result = lookup_in_table(table, key);
            if (result.has_value()) return result;
            table = table->prev;
        }
        return std::nullopt;
    }

    // Wait-free. A key is either published or not; an entry whose value has not been settled
    // yet reads as absent, which is the same answer a caller would get a moment earlier.
    std::optional<V> lookup_in_table(Table* table, K key) const {
        const size_t start = hash(key) & table->mask;
        for (size_t probe = 0; probe < table->capacity; ++probe) {
            const Entry& entry = table->entries[(start + probe) & table->mask];
            const K current = entry.key.load(std::memory_order_acquire);
            if (current == EMPTY_KEY) return std::nullopt;   // empty slot ends the probe run
            if (current == key) {
                const V v = entry.value.load(std::memory_order_acquire);
                if (v == ABSENT_VALUE) return std::nullopt;
                return v;
            }
        }
        return std::nullopt;
    }

    // Is `key` present in any table strictly newer than `older`? Used to emit each key once
    // while still walking the chain (a mid-flight insert can settle only in an old table, so
    // the chain cannot be skipped, but resize() rehashes settled entries so it double-visits).
    bool contains_in_newer(Table* head, Table* older, K key) const {
        for (Table* t = head; t && t != older; t = t->prev) {
            size_t idx = hash(key) & t->mask;
            for (size_t probe = 0; probe < t->capacity; ++probe) {
                const K k = t->entries[idx].key.load(std::memory_order_acquire);
                if (k == key) return true;
                if (k == EMPTY_KEY) break;           // key would have been placed by here
                idx = (idx + 1) & t->mask;
            }
        }
        return false;
    }

    template<typename F>
    void for_each_in_chain(Table* table, F&& f) const {
        while (table) {
            for (size_t i = 0; i < table->capacity; ++i) {
                const K key = table->entries[i].key.load(std::memory_order_acquire);
                if (key == EMPTY_KEY) continue;
                const V value = table->entries[i].value.load(std::memory_order_acquire);
                if (value != ABSENT_VALUE) f(key, value);
            }
            table = table->prev;
        }
    }

    void resize() {
        Table* old_table = table_.load(std::memory_order_acquire);
        size_t new_capacity = old_table->capacity * 2;

        Table* new_table = Table::create(new_capacity, old_table, arena_);

        // Rehash every settled entry. An entry whose value is not yet published stays behind
        // in the old table -- the thread settling it is still working against that table, and
        // the chain walk keeps it reachable.
        for (size_t i = 0; i < old_table->capacity; ++i) {
            const K key = old_table->entries[i].key.load(std::memory_order_acquire);
            if (key == EMPTY_KEY) continue;
            const V value = old_table->entries[i].value.load(std::memory_order_acquire);
            if (value != ABSENT_VALUE) insert_into_table(new_table, key, value, false);
        }

        // Try to install new table
        if (!table_.compare_exchange_strong(
                old_table, new_table,
                std::memory_order_release,
                std::memory_order_acquire)) {
            // Another thread resized first. Discard our table: only free if heap-backed;
            // an arena-backed loser is reclaimed in bulk with the arena (rare, small).
            if (!arena_) ::operator delete(new_table);
        }
    }

    std::atomic<Table*> table_;
    std::atomic<size_t> count_;
    // When non-null, all tables are arena-allocated (no malloc) and bulk-reclaimed.
    ConcurrentHeterogeneousArena* arena_ = nullptr;
};

}  // namespace hypergraph
