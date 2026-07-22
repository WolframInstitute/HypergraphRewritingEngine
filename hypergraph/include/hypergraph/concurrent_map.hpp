#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include "hgcommon/portable_intrinsics.hpp"
#include "arena.hpp"
#include <cstdio>
#include <cstdlib>
#include <new>
#include <optional>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace hypergraph {

// =============================================================================
// ConcurrentMap<K, V>: Lock-free hash map with open addressing
// =============================================================================
//
// Append-only concurrent hash map using open addressing with linear probing.
// Optimized for the case where we never delete entries.
//
// Thread safety: Lock-free insert_if_absent and lookup (NO spin-waits).
//
// Key requirements:
//   - K must be trivially copyable
//   - K must have TWO reserved values:
//     * EMPTY_KEY (default 0) - slot is available
//     * LOCKED_KEY - slot is being written (readers skip, no spin)
//   - Good hash distribution expected from caller
//
// Design: Uses sentinel-based insertion inspired by Folly AtomicUnorderedMap.
//   - Insert: CAS empty→LOCKED, write value, write key
//   - Lookup: Skip LOCKED slots (no spin-wait), treat as "not found yet"
//   - This is lock-free: no thread ever blocks waiting for another
//
// Resize: When load factor exceeds threshold, allocates new larger table.
//         Old table remains valid, lookups check both via chain.
//

template<typename K, typename V, K EMPTY_KEY = K{0}, K LOCKED_KEY = K{~0ULL}>
class ConcurrentMap {
public:
    static constexpr size_t DEFAULT_INITIAL_CAPACITY = 1024;
    static constexpr double LOAD_FACTOR_THRESHOLD = 0.75;

    struct Entry {
        std::atomic<K> key;
        std::atomic<V> value;

        Entry() : key(EMPTY_KEY), value{} {}
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

    // Insert key-value pair if key doesn't exist
    // Returns: (value, was_inserted)
    //   - If key was new: returns (value, true)
    //   - If key existed: returns (existing_value, false)
    std::pair<V, bool> insert_if_absent(K key, V value) {
        // Check if we need to resize
        Table* table = table_.load(std::memory_order_acquire);
        size_t current_count = count_.load(std::memory_order_relaxed);
        if (current_count > table->capacity * LOAD_FACTOR_THRESHOLD) {
            resize();
            table = table_.load(std::memory_order_acquire);
        }

        // Prev-chain dedup (a key may live only in an old table while a resize is in
        // flight) is deferred into insert_into_table: it consults the chain exactly once,
        // at the first EMPTY slot, having already proven the key absent from the current
        // table. A key already present in the current table returns from the probe's own
        // key-match and never walks the chain.
        return insert_into_table(table, key, value, true, table->prev, false);
    }

    // Like insert_if_absent but waits for LOCKED slots in prev tables.
    // Use this when deterministic deduplication is critical (e.g., canonical state maps).
    // The waiting ensures we don't miss concurrent inserts that are mid-flight.
    std::pair<V, bool> insert_if_absent_waiting(K key, V value) {
        // Check if we need to resize
        Table* table = table_.load(std::memory_order_acquire);
        size_t current_count = count_.load(std::memory_order_relaxed);
        if (current_count > table->capacity * LOAD_FACTOR_THRESHOLD) {
            resize();
            table = table_.load(std::memory_order_acquire);
        }

        // As insert_if_absent, but the deferred prev-chain probe waits for LOCKED slots so
        // a mid-flight concurrent insert of the same key is never missed.
        return insert_into_table(table, key, value, true, table->prev, true);
    }

    // Lookup value by key
    std::optional<V> lookup(K key) const {
        Table* table = table_.load(std::memory_order_acquire);
        return lookup_in_chain(table, key);
    }

    // Lookup value by key, waiting for LOCKED slots to resolve
    // Use this when you need to see entries that are being inserted concurrently
    std::optional<V> lookup_waiting(K key) const {
        Table* table = table_.load(std::memory_order_acquire);
        return lookup_in_chain_waiting(table, key);
    }

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
    size_t count_unique() const {
        size_t unique_count = 0;
        Table* table = table_.load(std::memory_order_acquire);

        // We need to track which keys we've seen because duplicates can exist
        // across different positions in the table due to the LOCKED race condition
        std::unordered_set<K> seen_keys;

        while (table) {
            for (size_t i = 0; i < table->capacity; ++i) {
                K key = table->entries[i].key.load(std::memory_order_acquire);
                if (key != EMPTY_KEY && key != LOCKED_KEY) {
                    if (seen_keys.insert(key).second) {
                        ++unique_count;
                    }
                }
            }
            table = table->prev;
        }
        return unique_count;
    }

    bool empty() const {
        return size() == 0;
    }

    // Iterate over all entries (not thread-safe during inserts)
    template<typename F>
    void for_each(F&& f) const {
        Table* table = table_.load(std::memory_order_acquire);
        for_each_in_chain(table, std::forward<F>(f));
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

    // prev_for_dedup: when non-null, the tail of the resize chain (table->prev). A key can
    //   transiently live only in an old table while a resize is in flight, so before this
    //   insert publishes a NEW key it must confirm the key is absent from that chain. The
    //   probe below reaches an EMPTY slot only after proving the key absent from `table`
    //   (linear-probe invariant), so the chain is walked at most once, and only for keys
    //   that are genuinely new to `table`. Passing nullptr (e.g. the resize rehash, which
    //   moves known-unique keys forward) skips the chain probe entirely.
    // wait_for_prev: use the LOCKED-waiting chain probe, for maps whose dedup must not miss
    //   a mid-flight concurrent insert of the same key.
    std::pair<V, bool> insert_into_table(Table* table, K key, V value, bool increment_count,
                                         Table* prev_for_dedup = nullptr,
                                         bool wait_for_prev = false) {
        size_t h = hash(key);
        size_t idx = h & table->mask;

        // Track LOCKED slots we encountered - need to check them for duplicate keys
        size_t locked_slots[64];
        size_t num_locked = 0;
        bool prev_checked = false;  // chain probed at most once, at the first EMPTY slot

        for (size_t probe = 0; probe < table->capacity; ++probe) {
            size_t i = (idx + probe) & table->mask;
            Entry& entry = table->entries[i];

            K current_key = entry.key.load(std::memory_order_acquire);

            if (current_key == EMPTY_KEY) {
                // First EMPTY slot: the key is absent from `table` up to here, and linear
                // probing guarantees it is absent from the whole table. Before claiming,
                // confirm it is not living only in a superseded table mid-resize.
                if (prev_for_dedup && !prev_checked) {
                    auto existing = wait_for_prev
                                        ? lookup_in_chain_waiting(prev_for_dedup, key)
                                        : lookup_in_chain(prev_for_dedup, key);
                    if (existing.has_value()) {
                        return {*existing, false};
                    }
                    prev_checked = true;
                }

                // Slot is empty, try to claim it with LOCKED sentinel
                if (entry.key.compare_exchange_strong(
                        current_key, LOCKED_KEY,
                        std::memory_order_acq_rel,
                        std::memory_order_acquire)) {

                    // Successfully claimed slot - but we may have skipped LOCKED slots
                    // that are inserting the same key. Must wait for them to resolve.
                    // Note: LOCKED state is very short-lived (just value+key writes), so
                    // unbounded spin is safe here - if it takes "too long" something is broken.
                    if (num_locked > 0) {
                        for (size_t li = 0; li < num_locked; ++li) {
                            size_t ci = locked_slots[li];
                            K check_key;
                            // Spin until LOCKED resolves (should be very fast - nanoseconds)
                            do {
                                check_key = table->entries[ci].key.load(std::memory_order_acquire);
                                hgcommon::cpu_relax();
                            } while (check_key == LOCKED_KEY);

                            if (check_key == key) {
                                // Duplicate found! Release our claimed slot
                                entry.key.store(EMPTY_KEY, std::memory_order_release);
                                V existing = table->entries[ci].value.load(std::memory_order_acquire);
                                return {existing, false};
                            }
                        }
                    }

                    // No duplicate - proceed with insert
                    entry.value.store(value, std::memory_order_release);
                    entry.key.store(key, std::memory_order_release);
                    if (increment_count) {
                        count_.fetch_add(1, std::memory_order_relaxed);
                    }
                    return {value, true};
                }
                // CAS failed, re-read key
                current_key = entry.key.load(std::memory_order_acquire);
            }

            if (current_key == LOCKED_KEY) {
                // Slot is being written - remember it for later checking
                if (num_locked < 64) {
                    locked_slots[num_locked++] = i;
                }
                continue;
            }

            if (current_key == key) {
                // Key already exists - value is guaranteed to be set
                V existing = entry.value.load(std::memory_order_acquire);
                return {existing, false};
            }

            // Collision with different key, continue probing
        }

        // Table is full (shouldn't happen with proper load factor management)
        resize();
        return wait_for_prev ? insert_if_absent_waiting(key, value)
                             : insert_if_absent(key, value);
    }

    std::optional<V> lookup_in_chain(Table* table, K key) const {
        while (table) {
            auto result = lookup_in_table(table, key);
            if (result.has_value()) {
                return result;
            }
            table = table->prev;
        }
        return std::nullopt;
    }

    // Like lookup_in_chain but waits for LOCKED slots to resolve
    // Used in insert_if_absent to ensure we don't miss concurrent inserts
    std::optional<V> lookup_in_chain_waiting(Table* table, K key) const {
        while (table) {
            auto result = lookup_in_table_waiting(table, key);
            if (result.has_value()) {
                return result;
            }
            table = table->prev;
        }
        return std::nullopt;
    }

    std::optional<V> lookup_in_table_waiting(Table* table, K key) const {
        size_t h = hash(key);
        size_t idx = h & table->mask;

        for (size_t probe = 0; probe < table->capacity; ++probe) {
            size_t i = (idx + probe) & table->mask;
            const Entry& entry = table->entries[i];

            K current_key = entry.key.load(std::memory_order_acquire);

            // Wait for LOCKED slots to resolve - they might be inserting our key
            // MUST spin indefinitely - if we give up on a LOCKED slot that holds our key,
            // we'd incorrectly return nullopt and cause duplicate insertions.
            // LOCKED state is very brief (just value + key writes), so this is safe.
            while (current_key == LOCKED_KEY) {
                hgcommon::cpu_relax();
                current_key = entry.key.load(std::memory_order_acquire);
            }

            if (current_key == key) {
                return entry.value.load(std::memory_order_acquire);
            }

            if (current_key == EMPTY_KEY) {
                return std::nullopt;
            }
        }

        return std::nullopt;
    }

    std::optional<V> lookup_in_table(Table* table, K key) const {
        size_t h = hash(key);
        size_t idx = h & table->mask;

        for (size_t probe = 0; probe < table->capacity; ++probe) {
            size_t i = (idx + probe) & table->mask;
            const Entry& entry = table->entries[i];

            K current_key = entry.key.load(std::memory_order_acquire);

            if (current_key == key) {
                // Key found - value is guaranteed to be set because
                // key is written AFTER value in insert_into_table
                return entry.value.load(std::memory_order_acquire);
            }

            if (current_key == LOCKED_KEY) {
                // Slot is being written - no spin-wait, just continue probing
                // If the writer was inserting our key, we'll miss it here
                // but that's fine: the insert will complete and a retry
                // of lookup will find it (or the writer returns success)
                continue;
            }

            if (current_key == EMPTY_KEY) {
                // Empty slot reached, key not in this table
                return std::nullopt;
            }

            // Collision with different key, continue probing
        }

        return std::nullopt;
    }

    template<typename F>
    void for_each_in_chain(Table* table, F&& f) const {
        while (table) {
            for (size_t i = 0; i < table->capacity; ++i) {
                K key = table->entries[i].key.load(std::memory_order_acquire);
                // Skip EMPTY and LOCKED entries
                if (key != EMPTY_KEY && key != LOCKED_KEY) {
                    // Key is fully written, value is guaranteed to be set
                    V value = table->entries[i].value.load(std::memory_order_acquire);
                    f(key, value);
                }
            }
            table = table->prev;
        }
    }

    void resize() {
        Table* old_table = table_.load(std::memory_order_acquire);
        size_t new_capacity = old_table->capacity * 2;

        Table* new_table = Table::create(new_capacity, old_table, arena_);

        // Rehash all entries from old table
        // Skip EMPTY and LOCKED entries (LOCKED entries will be in new table via insert path)
        for (size_t i = 0; i < old_table->capacity; ++i) {
            K key = old_table->entries[i].key.load(std::memory_order_acquire);
            if (key != EMPTY_KEY && key != LOCKED_KEY) {
                V value = old_table->entries[i].value.load(std::memory_order_acquire);
                insert_into_table(new_table, key, value, false);
            }
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
