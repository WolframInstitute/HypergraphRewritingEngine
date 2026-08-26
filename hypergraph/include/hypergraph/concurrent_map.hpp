#pragma once
#include "hgcommon/namespace.hpp"

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include "hgcommon/core.hpp"
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

namespace HG_NAMESPACE {
namespace engine {

// Map keys built from engine ids are packed by ONE rule, shared with the device
// (hgcommon/core.hpp), and named here so host call sites reach them unqualified.
using hgcommon::id_key;
using hgcommon::id_from_key;
using hgcommon::IdPair;
using hgcommon::id_pair_from_key;

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
//     every caller returns the winner -- so was_inserted means "THIS CALL's offer won the
//     publishing exchange", which is exactly what the get-or-create callers need to decide
//     whether to keep the object they built.
//
//     EXACTLY ONE CALL EVER SEES was_inserted FOR A KEY, across every table generation and
//     however many times a value is offered. It is the exchange that says so, not a
//     comparison of the stored value against the caller's: offering the same value twice
//     would pass such a comparison the second time and hand out a second winner.
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

    // What a map allocates before anyone puts anything in it.
    //
    // An engine holds seventeen of these, and a run uses the ones its configuration calls for --
    // a run without quotient exploration never touches the quotient maps at all. Allocating
    // DEFAULT_INITIAL_CAPACITY entries in the constructor zeroes every one of them regardless,
    // and set_arena then allocates a SECOND full-size table to re-home the map onto the arena
    // and throws the first away, so each re-homed map paid for 2048 zeroed entries before its
    // first insert. Measured with callgrind on bench_cpu_evolve: __memset_avx2_unaligned_erms
    // was 51.3% of all instructions executed, against 0.46% in ir_canonical_hash.
    //
    // A map therefore starts at one slot and grows on its first insert. The growth path is the
    // ordinary one, so this adds no second mechanism -- and it jumps straight to
    // DEFAULT_INITIAL_CAPACITY rather than doubling from one, so a map that is used reaches the
    // same table in one step and a map that is not used never allocates it.
    static constexpr size_t LAZY_INITIAL_CAPACITY = 1;
    static constexpr double LOAD_FACTOR_THRESHOLD = 0.75;

    // The value slot's THIRD state: a resize seals every unsettled slot of the table it
    // supersedes with this, so no value can ever settle in a superseded table after the seal
    // (see resize). Pointer maps use address 1, which no real object occupies; integral maps
    // the all-ones value, or all-ones-minus-one when ABSENT_VALUE is itself all-ones.
    // publish_value rejects a stored value equal to it, exactly as it rejects ABSENT_VALUE,
    // so a caller whose value domain collides finds out loudly. V must be able to hold three
    // distinct states -- a V of bool cannot; such maps store uint8_t (their `true` converts
    // to 1, distinct from ABSENT 0 and FORWARDED 0xFF).
    static V forwarded_value() {
        if constexpr (std::is_pointer_v<V>) {
            return reinterpret_cast<V>(static_cast<uintptr_t>(1));
        } else {
            const V allones = static_cast<V>(~static_cast<uint64_t>(0));
            return ABSENT_VALUE == allones ? static_cast<V>(static_cast<uint64_t>(allones) - 1)
                                           : allones;
        }
    }

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
        // TRUE ONCE EVERY VALUE THAT SETTLED HERE HAS A SETTLED COPY IN A STRICTLY NEWER
        // TABLE, so a chain walk may skip this table without missing anything. Set by the
        // thread that ran this table's migration, after its carry loop returns: the carry
        // offers at the live head (drive_at_head skips the chain scan on the carry path), and
        // each insert_into_table call returns only when the copy's value exchange is done --
        // so the flag's release publishes settled copies, never in-flight ones. The seal that
        // precedes the migration is what freezes the settled set this promise is about.
        std::atomic<bool> drained{false};
        // TRUE when EVERY older table is drained: a walk from this table need probe nothing
        // below it. Seeded true on the first table (nothing is older) and propagated by each
        // resize whose retiring table both drained and carried the flag.
        std::atomic<bool> chain_clear{false};

        // When arena != nullptr the table is allocated from the arena (no malloc, no
        // per-map heap contention) and is NEVER individually freed — it is reclaimed in
        // bulk when the arena is. When arena == nullptr it falls back to ::operator new
        // (freed in the destructor), for standalone/test use.
        static Table* create(size_t cap, Table* prev_table,
                             ConcurrentHeterogeneousArena* arena) {
            // Capacity must be power of 2
            size_t actual_cap = 1;
            while (actual_cap < cap) actual_cap <<= 1;

            // THE ENTRY ARRAY STARTS ON A CACHE LINE, which the header size otherwise decides
            // for it. Table is 40 bytes and Entry is 16, so entries placed immediately after the
            // header begin at a 40-byte phase: the third Entry of every four spans two lines, and
            // a probe run pays a second line fetch one time in four, on every map, for the life
            // of the run.
            //
            // The alignment is applied to the POINTER inside an over-allocated block rather than
            // to the allocation, so it holds whatever the base alignment is -- ::operator new
            // gives only max_align_t -- and the three ::operator delete sites keep freeing the
            // same address they were given.
            constexpr size_t kLine = 64;
            size_t bytes = sizeof(Table) + kLine + sizeof(Entry) * actual_cap;
            void* mem = arena ? arena->allocate_raw(bytes, kLine)
                              : ::operator new(bytes);
            Table* table = static_cast<Table*>(mem);
            auto raw = reinterpret_cast<uintptr_t>(static_cast<char*>(mem) + sizeof(Table));
            table->entries = reinterpret_cast<Entry*>((raw + (kLine - 1)) & ~(uintptr_t)(kLine - 1));
            table->capacity = actual_cap;
            table->mask = actual_cap - 1;
            table->prev = prev_table;
            new (&table->drained) std::atomic<bool>(false);
            // The first table has nothing older, so a walk from it never needs the chain.
            new (&table->chain_clear) std::atomic<bool>(prev_table == nullptr);

            // Initialize entries
            for (size_t i = 0; i < actual_cap; ++i) {
                new (&table->entries[i]) Entry();
            }

            return table;
        }
    };

    // arena != nullptr routes all table allocation through the arena (no malloc); the
    // tables are then reclaimed in bulk with the arena, not in this destructor.
    // `working_capacity` is the size the FIRST growth jumps to. It is a parameter so a model
    // checker can bound the protocol: the seal pass exchanges two atomics per slot of the
    // retiring table, so a growth out of a 1024-slot table is 2048 atomic operations for a
    // checker to interleave, and the double-growth harnesses cannot be exhausted because of it
    // -- measured, both --bound=1 and --bound=2 time out at 580s, and the two-worker shape
    // produced no verdict in 55 minutes.
    //
    // The property those harnesses state -- install, seal, migrate, chain-scan, re-drive, and
    // exactly one caller told it inserted -- does not depend on the table being 1024 wide. The
    // capacity decides probe-run LENGTH, and with the one or two keys a harness uses the runs
    // are trivial at either size. So the constant is a parameter and the default is unchanged.
    explicit ConcurrentMap(size_t initial_capacity = LAZY_INITIAL_CAPACITY,
                           ConcurrentHeterogeneousArena* arena = nullptr,
                           size_t working_capacity = DEFAULT_INITIAL_CAPACITY)
        : count_(0), arena_(arena), working_capacity_(working_capacity) {
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
        size_t cap = old ? old->capacity : LAZY_INITIAL_CAPACITY;
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
#ifdef HG_VERIFICATION
            // A model checker resolves every symbol the module names while it initialises memory,
            // before any thread runs, and a throw of a standard exception names that exception's
            // typeinfo -- an external constant with no definition here. Reaching the throw is not
            // required for it to fault. An assertion states the same precondition, and a violated
            // one is a safety property the checker reports rather than a crash it dies on.
            (void)op;
            assert(false && "ConcurrentMap: key collides with a reserved sentinel (EMPTY/LOCKED)");
#else
            throw std::logic_error(
                std::string("ConcurrentMap::") + op + ": key " +
                std::to_string(static_cast<unsigned long long>(key)) + " collides with " +
                (key == EMPTY_KEY ? "EMPTY" : "LOCKED") +
                ". Offset dense ids by +1 or use a reserved sentinel band.");
#endif
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

        (void)table;

        // THE VERDICT IS THE EXCHANGE. drive_at_head reports whether THIS caller's offer won
        // the ABSENT-to-value exchange that settled the key, and that exchange is unique per
        // key: a claimant's scan of the older tables either finds the settled value, or finds
        // the one unsettled fresh claim and offers into it, or forecloses the slot a late claim
        // would need (see find_and_settle_in_chain). A copy of a settled value in flight to a
        // newer table is never offered into, because its original is found first. No comparison
        // of the stored value against the caller's is made afterwards: two callers may offer the
        // same value, and a comparison would call both of them the inserter.
        // concurrent_map_double_growth_3t drives two claimants offering the SAME value through
        // a growth under a third thread and asserts exactly one is told it inserted.
        return drive_at_head(key, value, true);
    }

    // Drive `key` at the head, entering through the CHAIN.
    //
    // An entry can live in only one table -- an insert that resolved against a table after its
    // slot had been rehashed completes there -- and inserting again would give one key two
    // entries, which for the get-or-create callers means two container objects and a silently
    // split rendezvous.
    //
    // The scan SETTLES rather than merely looks. A plain lookup reports a key whose value is
    // not yet published as absent, which is the right answer for a reader and the wrong one
    // here: this caller is holding a value to offer, so it completes that entry instead of
    // walking past it and creating a rival. Offering a value is the same exchange every
    // publisher uses, so this closes the window without anyone waiting.
    //
    // EVERY RE-DRIVE COMES BACK THROUGH THE SCAN. A path that re-drives -- a seal, an
    // exhausted probe run, a growth -- arrives with an absence verdict for the ONE table it
    // came from, while tables installed meanwhile may hold a rival's settled claim the carry
    // has not yet placed in the head. Claiming fresh on that verdict would tell two callers
    // was_inserted for one key. Rescanning from the current head re-establishes it: each older
    // table either yields its claim to the scan or is foreclosed by the scan's seal.
    //
    // The scan is skipped only for a CARRY (increment_count false): that caller is placing a
    // value already settled in a superseded table, not deciding absence.
    std::pair<V, bool> drive_at_head(K key, V value, bool increment_count) {
        Table* head = table_.load(std::memory_order_acquire);
        // chain_clear is read BEFORE any probe of the head, which is what makes skipping the
        // scan sound: true means every older table drained before the flag's release, so the
        // settled copies those drains promised are visible to every probe this call makes.
        if (increment_count && head->prev &&
            !head->chain_clear.load(std::memory_order_acquire)) {
            auto settled = find_and_settle_in_chain(head->prev, key, value);
            if (settled.has_value()) return *settled;
        }
        // One key can be CLAIMED in two tables -- a claimant working from a superseded table
        // racing a rival at the head -- but it can only ever SETTLE through one exchange, and
        // every caller answers from that exchange. Claims serialize per slot (a claim is a
        // compare-exchange, which reads the slot's latest value), and the chain scan above
        // SEALS the slot any late claim would need: a scan whose probe run ends at an EMPTY
        // slot exchanges it to LOCKED before moving on (see find_and_settle_in_chain). So in
        // each superseded table, either the scan discovers the claim -- and the exchange it
        // offers into is the ONE exchange for the key -- or its seal lands first and no claim
        // for the key can ever settle there: a later claim's exchange reads the LOCKED slot
        // and re-drives at the head. Reading LOCKED also hands the claimant the newer head:
        // the sealer loaded table_ after the install, and the exchange carries that
        // visibility, so the re-drive cannot land back in the table it just left. resize()
        // seals whole tables through the same exchanges (EMPTY keys to LOCKED, ABSENT values
        // to FORWARDED) before migrating.
        //
        // A settled value that beat the retiring table's seal stays in that superseded table
        // and is carried to the head by the growth. The carry is a key exchange followed by a
        // value exchange, and between them the head holds a copy that looks like a fresh
        // unsettled claim; the scan distinguishes the two by what lies older (see
        // find_and_settle_in_chain), so the copy is never offered into and the exchange that
        // settled the original stays the only winner. Three harnesses bound this, and they
        // bound different things: concurrent_map_double_growth_2t EXHAUSTS two workers across
        // two growths under RC11, concurrent_map_double_growth_3t exhausts three workers with
        // both claimants offering the same value under a four-context bound, and
        // concurrent_map_double_growth samples three workers -- estimation, not proof.
        return insert_into_table(head, key, value, increment_count);
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
        // Read before the probe, which is what makes the skip sound; see drive_at_head.
        if (table->chain_clear.load(std::memory_order_acquire))
            return lookup_in_table(table, key);
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
                if (key == EMPTY_KEY || key == LOCKED_KEY) continue;
                const V v = t->entries[i].value.load(std::memory_order_acquire);
                if (v == ABSENT_VALUE || v == forwarded_value()) continue;
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
                if (key == EMPTY_KEY || key == LOCKED_KEY) continue;
                const V v = t->entries[i].value.load(std::memory_order_acquire);
                if (v == ABSENT_VALUE || v == forwarded_value()) continue;
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
                    return settle(table, entry, key, value, increment_count);
                }
                // Lost the slot; `current` now holds whichever key won it.
            }

            if (current == LOCKED_KEY) {
                // The seal: this table is superseded and takes no new claims. LOCKED replaced
                // a slot that was EMPTY at seal time, and no probe run ever passed an EMPTY
                // slot, so a key present in this table sits strictly before its run's first
                // LOCKED -- reaching one means the key is not here. Re-drive at the head.
                return drive_at_head(key, value, increment_count);
            }

            if (current == key) return settle(table, entry, key, value, increment_count);
            // Different key: keep probing.
        }

        // Table full (should not happen under the load factor). Retry at this level, preserving
        // increment_count: re-entering through insert_if_absent would force counting on,
        // double-counting when the caller is resize()'s rehash.
        Table* head = table_.load(std::memory_order_acquire);

        // A probe run exhausted in a SUPERSEDED table calls for the head, not for growth. The
        // head is already a larger table with room, and growing it would install another one --
        // and every extra installation is another chance for one key to be claimed in two tables
        // at once.
        if (head != table) return drive_at_head(key, value, increment_count);

        // The head itself is full, so grow it regardless of the load factor: capacity elsewhere
        // in the chain cannot relieve an exhausted probe run here.
        resize(/*only_if_loaded=*/false);
        return drive_at_head(key, value, increment_count);
    }

    // Settle this entry's value. Returns the stored value and whether it is the caller's, or
    // nullopt when the slot is SEALED (value FORWARDED by a resize) -- the caller must then
    // re-drive at the head, where the entry this one forwards to lives.
    std::optional<std::pair<V, bool>> publish_value(Entry& entry, V value) {
        // A stored value equal to a reserved value state would read as "not published yet" or
        // as a seal, so the entry would be invisible to every lookup -- the same
        // silent-disappearance the key sentinels caused four times over, moved to the other
        // field. Report it instead.
        if (value == ABSENT_VALUE || value == forwarded_value()) {
#ifdef HG_VERIFICATION
            // See reject_sentinel_key for why a model-checked build asserts instead of throwing.
            assert(false && "ConcurrentMap: stored value collides with ABSENT/FORWARDED");
#else
            throw std::logic_error(
                "ConcurrentMap: stored value collides with ABSENT_VALUE or forwarded_value(), "
                "so the entry would read as unpublished or sealed. Adjust this map's value "
                "domain.");
#endif
        }

        V current = entry.value.load(std::memory_order_acquire);
        if (current == forwarded_value()) return std::nullopt;
        if (current != ABSENT_VALUE) return std::make_pair(current, false);

        if (entry.value.compare_exchange_strong(current, value,
                                                std::memory_order_acq_rel,
                                                std::memory_order_acquire)) {
            return std::make_pair(value, true);
        }
        if (current == forwarded_value()) return std::nullopt;   // the seal won the exchange
        return std::make_pair(current, false);   // another thread's value won
    }

    // Settle through `entry` in `table`. publish_value performs the one compare-exchange that
    // decides who published this key's value, and its answer is the verdict: nothing is
    // compared or recomputed afterwards. A value that settled here is carried into the head by
    // whichever growth retires this table -- the seal that would have forwarded this slot lost
    // to the publish, so the carry reads the settled value -- and until then it is answerable
    // from the chain walk.
    //
    // NOTHING IS COPIED TO THE HEAD FROM HERE. A copy is a key claimed in a newer table with
    // its value still unpublished, which is exactly what a fresh claim looks like to a scan;
    // the carry is the one place copies are made, and it copies only settled values, so every
    // copy has a settled original in an older table for the scan to find first.
    //
    // A sealed slot (publish_value returns nullopt) means this table is superseded and nothing
    // will ever settle here. Re-drive at the head, where the claim belongs.
    std::pair<V, bool> settle(Table* table, Entry& entry, K key, V value,
                              bool increment_count) {
        (void)table;
        if (auto r = publish_value(entry, value)) return *r;
        return drive_at_head(key, value, increment_count);
    }

    // Find `key` anywhere in the chain and ensure its value is settled, offering `value` if it
    // is not. Returns the settled value and whether it is the caller's, or nullopt if the key
    // is absent from every table. A sealed (FORWARDED) claim is a claim that never settled and
    // never will; the chain runs newest-first, so nothing newer holds it either -- skip it and
    // keep scanning for an older settled original the carry may not have placed yet.
    //
    // SEAL ON SCAN. A probe run that reaches an EMPTY slot does not just report "not here":
    // it exchanges the slot to LOCKED first. The slot it seals is exactly the slot any
    // in-flight claim for `key` in this table must take (a claim takes the first EMPTY slot
    // of this same run, and claims are exchanges, so they serialize per slot). Either this
    // seal wins -- then no claim for `key` can ever settle in this table, because a later
    // claim's exchange reads the LOCKED slot and re-drives at the head -- or a claim got
    // there first, in which case the failed exchange hands back that claim and the scan
    // interprets it. This is what makes the scan complete: it cannot walk past a concurrent
    // claim, it either discovers it or forecloses it.
    //
    // AN UNSETTLED CLAIM IS OFFERED INTO ONLY AFTER EVERY OLDER TABLE HAS BEEN SCANNED. A key
    // with its value still ABSENT is one of two things, and they look identical: a fresh claim
    // whose publisher has not yet exchanged, or a COPY -- the carry placing a value that
    // settled in an older table, between its key exchange and its value exchange. Offering
    // into a copy hands the key a second winner: the original's publisher was already told it
    // inserted, and the offer into the copy wins an exchange nothing else is contending.
    // GenMC produced exactly that execution: one claimant settled in the initial table, the
    // other's scan found the in-flight copy of that value in the next table and won its
    // exchange, and both reported was_inserted. The two cases are separated by what lies
    // OLDER: a copy always has a settled original in an older table, a fresh claim never does.
    // So the scan remembers the newest unsettled entry and keeps walking; it returns an older
    // settled value if there is one, and offers into the remembered entry only when there is
    // not. The walk past the entry seals as it goes, so no claim can land in the older tables
    // between the walk and the offer.
    std::optional<std::pair<V, bool>> find_and_settle_in_chain(Table* table, K key, V value) {
        Entry* unsettled = nullptr;
        // EVERY DRAINED FLAG IS READ BEFORE ANY TABLE IS PROBED. Reading a table's flag only
        // when the walk reaches it reopens the hand-off window ConcurrentKeySet::contains
        // had: a newer table probed before a carry lands its copy, the older table drained by
        // the time the walk arrives and skipped -- the key invisible at both, and the offer
        // that follows is a second exchange for a settled key. Read up front, a flag that is
        // TRUE orders this walk after the drain's release, so the copies it promised are
        // visible to every probe below; a flag that is FALSE merely probes a table whose
        // entries are all still in place, which is always sound. Tables past the snapshot are
        // probed unconditionally, the sound direction.
        constexpr size_t kSnapshot = 64;
        bool skip[kSnapshot];
        size_t chain_len = 0;
        for (Table* t = table; t; t = t->prev, ++chain_len)
            if (chain_len < kSnapshot)
                skip[chain_len] = t->drained.load(std::memory_order_acquire);
        size_t depth = 0;
        while (table) {
            if (depth < kSnapshot && skip[depth]) {
                table = table->prev;
                ++depth;
                continue;
            }
            ++depth;
            const size_t start = hash(key) & table->mask;
            for (size_t probe = 0; probe < table->capacity; ++probe) {
                Entry& entry = table->entries[(start + probe) & table->mask];
                K current = entry.key.load(std::memory_order_acquire);
                if (current == EMPTY_KEY) {
                    if (entry.key.compare_exchange_strong(current, LOCKED_KEY,
                                                          std::memory_order_acq_rel,
                                                          std::memory_order_acquire))
                        break;   // sealed: `key` can never settle in this table; try the next
                    // Lost the exchange; `current` holds what beat us -- interpret it below.
                }
                if (current == LOCKED_KEY) break;         // not in THIS table; try the next
                if (current == key) {
                    const V v = entry.value.load(std::memory_order_acquire);
                    if (v == forwarded_value()) break;    // sealed dead claim; try the next
                    if (v != ABSENT_VALUE) return std::make_pair(v, false);
                    if (!unsettled) unsettled = &entry;   // fresh claim or copy: decide below
                    break;
                }
            }
            table = table->prev;
        }
        if (!unsettled) return std::nullopt;
        // Nothing settled older, so this is the one fresh claim for `key`. Offer into it: the
        // exchange decides between this caller and the claimant. If the slot was sealed in the
        // meantime the claim is dead and its claimant re-drives; so does this caller.
        auto r = publish_value(*unsettled, value);
        if (!r) return std::nullopt;
        return r;
    }

    std::optional<V> lookup_in_chain(Table* table, K key) const {
        // Drained flags read up front, before any probe -- same discipline and same reason as
        // find_and_settle_in_chain, minus the offer: here the hazard of a late-read flag is a
        // settled key reported absent.
        constexpr size_t kSnapshot = 64;
        bool skip[kSnapshot];
        size_t chain_len = 0;
        for (Table* t = table; t; t = t->prev, ++chain_len)
            if (chain_len < kSnapshot)
                skip[chain_len] = t->drained.load(std::memory_order_acquire);
        size_t depth = 0;
        while (table) {
            const bool probe = depth >= kSnapshot || !skip[depth];
            ++depth;
            if (probe) {
                auto result = lookup_in_table(table, key);
                if (result.has_value()) return result;
            }
            table = table->prev;
        }
        return std::nullopt;
    }

    // Wait-free. A key is either published or not; an entry whose value has not been settled
    // yet -- or whose slot a resize sealed -- reads as absent, which is the same answer a
    // caller would get a moment earlier. LOCKED ends a probe run exactly as EMPTY does: it
    // replaced a slot that was EMPTY at seal time, and no probe run ever passed an EMPTY slot.
    std::optional<V> lookup_in_table(Table* table, K key) const {
        const size_t start = hash(key) & table->mask;
        for (size_t probe = 0; probe < table->capacity; ++probe) {
            const Entry& entry = table->entries[(start + probe) & table->mask];
            const K current = entry.key.load(std::memory_order_acquire);
            if (current == EMPTY_KEY || current == LOCKED_KEY) return std::nullopt;
            if (current == key) {
                const V v = entry.value.load(std::memory_order_acquire);
                if (v == ABSENT_VALUE || v == forwarded_value()) return std::nullopt;
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
                if (k == key)
                    return t->entries[idx].value.load(std::memory_order_acquire)
                           != forwarded_value();     // a sealed dead claim shadows nothing
                if (k == EMPTY_KEY || k == LOCKED_KEY) break;  // placed by here if anywhere
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
                if (key == EMPTY_KEY || key == LOCKED_KEY) continue;
                const V value = table->entries[i].value.load(std::memory_order_acquire);
                if (value != ABSENT_VALUE && value != forwarded_value()) f(key, value);
            }
            table = table->prev;
        }
    }

    void resize(bool only_if_loaded = true) {
        Table* old_table = table_.load(std::memory_order_acquire);

        // Grow only if the CURRENT head still needs it. Callers decide to resize by comparing the
        // count against the table they were holding, and by the time they get here another thread
        // may have installed a bigger one that already has room. Growing anyway installs a SECOND
        // table, and a second installation is what lets one key be claimed twice: a thread that
        // resolved against the middle table scans its ancestors, finds nothing, and claims there,
        // while a thread working from the new head has already walked past that middle table in
        // its own scan and claims the same key at the head. Both are then told they inserted.
        //
        // insert_into_table passes false: it calls this because a probe run was exhausted, which
        // capacity elsewhere in the chain does not relieve.
        if (only_if_loaded &&
            count_.load(std::memory_order_relaxed) <=
                old_table->capacity * LOAD_FACTOR_THRESHOLD) {
            return;
        }

        // From the one-slot table a map starts with, go straight to the working size rather
        // than doubling ten times to reach it.
        size_t new_capacity = old_table->capacity < working_capacity_
                                  ? working_capacity_
                                  : old_table->capacity * 2;

        Table* new_table = Table::create(new_capacity, old_table, arena_);

        // INSTALL FIRST. The seal below redirects claimants and publishers to the head, so
        // the head they are redirected to has to exist before any slot is sealed -- and the
        // redirect must SEE it: a claimant that reads LOCKED, or a publisher whose exchange
        // loses to FORWARDED, acquire-reads a value this thread wrote after the install, so
        // its re-drive load of table_ observes the new head (or a newer one).
        if (!table_.compare_exchange_strong(
                old_table, new_table,
                std::memory_order_release,
                std::memory_order_acquire)) {
            // Another thread resized first. Discard our table: only free if heap-backed;
            // an arena-backed loser is reclaimed in bulk with the arena (rare, small).
            if (!arena_) ::operator delete(new_table);
            return;
        }

        // SEAL. Every EMPTY key becomes LOCKED (no new claims here; a probe run ends at the
        // first LOCKED exactly as it ended at the first EMPTY) and every ABSENT value becomes
        // FORWARDED (no late settle; the publisher that loses this exchange re-drives at the
        // head). Chain scans seal individual slots through this same key exchange
        // (find_and_settle_in_chain); this pass seals the whole table, and only the thread
        // whose install succeeded runs it.
        for (size_t i = 0; i < old_table->capacity; ++i) {
            K ek = EMPTY_KEY;
            old_table->entries[i].key.compare_exchange_strong(
                ek, LOCKED_KEY, std::memory_order_acq_rel, std::memory_order_acquire);
            V av = ABSENT_VALUE;
            old_table->entries[i].value.compare_exchange_strong(
                av, forwarded_value(), std::memory_order_acq_rel, std::memory_order_acquire);
        }

        // MIGRATE. After the seal the superseded table's settled set is FROZEN -- a value
        // either settled before its slot's seal and is visible to this acquire scan, or it
        // lost to the seal and settles at the head under its own power -- so this copy is
        // complete, and every key that ever settled here is answerable from the head chain.
        // The old entries stay in place; newer tables are scanned first, so they are shadowed.
        //
        // Each carry call returns with its copy SETTLED at whichever table was head when it
        // landed: the carry path never chain-scans (drive_at_head skips the scan when the
        // count is not incremented), so a bounce off a sealed slot re-offers at the newer
        // head rather than resolving to the original below. That is what makes `drained`
        // safe to set after this loop -- every settled key of this table has a settled copy
        // strictly above it, published before the flag's release.
        for (size_t i = 0; i < old_table->capacity; ++i) {
            const K key = old_table->entries[i].key.load(std::memory_order_acquire);
            if (key == EMPTY_KEY || key == LOCKED_KEY) continue;
            const V value = old_table->entries[i].value.load(std::memory_order_acquire);
            if (value == ABSENT_VALUE || value == forwarded_value()) continue;
            insert_into_table(table_.load(std::memory_order_acquire), key, value, false);
        }
        old_table->drained.store(true, std::memory_order_release);
        // With everything below old_table already drained and old_table now drained too, a
        // walk from its successor needs no chain at all. Set on the successor even if it has
        // itself been superseded meanwhile: the flag is about what lies BELOW that table, and
        // the resize retiring it reads it to keep the propagation going.
        if (old_table->chain_clear.load(std::memory_order_acquire))
            new_table->chain_clear.store(true, std::memory_order_release);
    }

    std::atomic<Table*> table_;
    std::atomic<size_t> count_;
    // When non-null, all tables are arena-allocated (no malloc) and bulk-reclaimed.
    ConcurrentHeterogeneousArena* arena_ = nullptr;
    // The size the first growth jumps to; see the constructor.
    size_t working_capacity_ = DEFAULT_INITIAL_CAPACITY;
};

}  // namespace engine
}  // namespace HG_NAMESPACE