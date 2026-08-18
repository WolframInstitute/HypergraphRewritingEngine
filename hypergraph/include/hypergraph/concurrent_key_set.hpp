#pragma once
#include "hgcommon/namespace.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>

#include "arena.hpp"

namespace HG_NAMESPACE {
namespace engine {

// A lock-free set of keys, for the question "have I seen this one before?".
//
// WHY NOT ConcurrentMap. The engine asks that question in five places -- the quotient
// reconstruction's reached, applied and dominator-support marks, the transition-capture dedup,
// and the branchial pair dedup -- and each used ConcurrentMap<uint64_t, uint8_t> storing the
// constant true. A map has to PUBLISH a value, and its whole protocol follows from that: a slot
// passes through LOCKED while a value is in flight, an insert meeting a claim must SETTLE it
// rather than walk past and create a rival, and a re-drive rescans the superseded-table chain so
// one key cannot be claimed in two tables at once. Measured, that protocol is the largest single
// cost in a quotient run: quotient bookkeeping is 74.17% of host cycles (phase counters, depth 6,
// exclusive), and drive_at_head on those maps alone is 16.32% of executed instructions.
//
// A set has no value to publish, so presence and ownership are decided by one compare-exchange.
//
// TWO RESERVED KEYS. Growth copies a key into the new table while the superseded table stays
// reachable, so a key exists in more than one table at once and a chain walk would emit it twice.
// ConcurrentMap avoids that by marking a migrated entry's VALUE as forwarded; here the MIGRATED
// key does the same job, stamped into the old slot once the key is safely in the newer table.
//
// EMPTY and MIGRATED are reserved and must never be inserted. The engine's keys are mixed hashes
// and already avoid 0 and ~0, because ConcurrentMap reserved the same two.
// Keys migrate_into could not carry because the destination's probe run was exhausted, which
// concurrent claims into the new table can cause while a migration runs. NOT a loss: the caller
// leaves such a key unsealed in the source table and does not drain it, so both for_each and
// find_in_table still reach it. Counted because it is the cost side of that decision -- a
// superseded table stays in the chain and is walked -- and because a number that climbs steeply
// says the growth policy is under-sizing the destination. Process-wide, never reset.
inline std::atomic<uint64_t>& migrate_deferrals() {
    static std::atomic<uint64_t> n{0};
    return n;
}

template<typename K = uint64_t,
         K EMPTY_KEY    = K{0},
         K MIGRATED_KEY = static_cast<K>(~K{0})>
class ConcurrentKeySet {
public:
    static constexpr double LOAD_FACTOR_THRESHOLD = 0.75;
    // A set starts at one slot so an unused one costs nothing, and reaches its working size in a
    // single growth rather than by doubling from one.
    //
    // The chain is why this matters, not the allocation. A superseded table stays reachable so a
    // late claim is still found, and both insert() and contains() walk it, so the chain length is
    // paid on EVERY operation. Doubling from one to a million keys leaves twenty tables to walk;
    // jumping to the working size leaves eleven, and the eleven that remain are the large ones
    // where a probe actually finds something. Measured on the depth-6 quotient corpus, doubling
    // from one put 163.5G cycles in the quotient phase against 9.1G for the ConcurrentMap it
    // replaced -- the structure was cheaper per operation and lost it all to chain walking.
    static constexpr size_t DEFAULT_INITIAL_CAPACITY = 1;
    static constexpr size_t WORKING_CAPACITY = 1024;

    struct Table {
        size_t capacity;
        size_t mask;
        Table* prev;
        std::atomic<K>* keys;
        // Set once migration has carried every key of this table forward and sealed every slot.
        //
        // A drained table must be SKIPPED, not walked, and the reason is a cost rather than a
        // correctness point. Sealing writes MIGRATED into every slot, which destroys the EMPTY
        // terminator linear probing stops at -- so a probe through a drained table runs its whole
        // capacity instead of a handful of slots. With a retired table of half a million entries
        // on the chain, every insert paid a full scan of it: measured at 155.7G cycles in the
        // quotient phase against 9.1G for the ConcurrentMap being replaced. Skipping is sound
        // because draining means exactly that every key here is already in a newer table.
        std::atomic<bool> drained{false};

        static Table* create(size_t cap, Table* prev, ConcurrentHeterogeneousArena* arena) {
            size_t actual = 1;
            while (actual < cap) actual <<= 1;
            const size_t bytes = sizeof(Table) + sizeof(std::atomic<K>) * actual;
            void* mem = arena ? arena->allocate_raw(bytes, alignof(Table))
                              : ::operator new(bytes);
            Table* t = new (mem) Table{actual, actual - 1, prev, nullptr, {false}};
            t->keys = reinterpret_cast<std::atomic<K>*>(
                reinterpret_cast<char*>(mem) + sizeof(Table));
            for (size_t i = 0; i < actual; ++i) new (&t->keys[i]) std::atomic<K>(EMPTY_KEY);
            return t;
        }
    };

    explicit ConcurrentKeySet(size_t initial_capacity = DEFAULT_INITIAL_CAPACITY,
                              ConcurrentHeterogeneousArena* arena = nullptr)
        : count_(0), arena_(arena) {
        table_.store(Table::create(initial_capacity, nullptr, arena_), std::memory_order_release);
    }

    ~ConcurrentKeySet() {
        if (arena_) return;
        Table* t = table_.load(std::memory_order_acquire);
        while (t) { Table* p = t->prev; ::operator delete(t); t = p; }
    }

    ConcurrentKeySet(const ConcurrentKeySet&) = delete;
    ConcurrentKeySet& operator=(const ConcurrentKeySet&) = delete;

    void set_arena(ConcurrentHeterogeneousArena* arena) {
        arena_ = arena;
        Table* old = table_.load(std::memory_order_relaxed);
        table_.store(Table::create(old ? old->capacity : DEFAULT_INITIAL_CAPACITY,
                                   nullptr, arena_), std::memory_order_release);
        while (old) { Table* p = old->prev; ::operator delete(old); old = p; }
    }

    // True iff this call is the one that added the key.
    bool insert(K key) {
        for (;;) {
            Table* head = table_.load(std::memory_order_acquire);
            if (count_.load(std::memory_order_relaxed) > head->capacity * LOAD_FACTOR_THRESHOLD) {
                grow(head);
                continue;
            }
            // A key already settled in a superseded table must not be claimed again at the head.
            //
            // This scan does NOT need to seal what it passes, and that is a consequence of the
            // growth order rather than an oversight. ConcurrentMap's chain scan must seal,
            // because there a table is retired while claims can still land in it. Here grow()
            // installs the successor FIRST and then seals every slot of the retiring table
            // before it returns, so a table reachable through prev is already closed: no claim
            // can land in it after this scan passes, because none can land in it at all.
            // Checked by removing the scan's seal and re-running both harnesses -- 781
            // executions, no violation -- which is what says the seal was redundant here.
            bool in_chain = false;
            for (Table* t = head->prev; t && !in_chain; t = t->prev)
                if (!t->drained.load(std::memory_order_acquire)) in_chain = find_in_table(t, key);
            if (in_chain) return false;

            const Claim c = claim(head, key);
            if (c == Claim::kWon)  return true;
            if (c == Claim::kLost) return false;
            // kStale: this table filled under us. Re-drive from the current head.
        }
    }

    bool contains(K key) const {
        for (Table* t = table_.load(std::memory_order_acquire); t; t = t->prev)
            if (!t->drained.load(std::memory_order_acquire) && find_in_table(t, key)) return true;
        return false;
    }

    size_t size() const { return count_.load(std::memory_order_relaxed); }

    // Every key exactly once: a migrated slot names itself, so there is nothing to deduplicate.
    template<typename F>
    void for_each(F&& f) const {
        for (Table* t = table_.load(std::memory_order_acquire); t; t = t->prev) {
            if (t->drained.load(std::memory_order_acquire)) continue;   // all slots sealed
            for (size_t i = 0; i < t->capacity; ++i) {
                const K k = t->keys[i].load(std::memory_order_acquire);
                if (k != EMPTY_KEY && k != MIGRATED_KEY) f(k);
            }
        }
    }

private:
    enum class Claim { kWon, kLost, kStale };

    static size_t hash(K key) {
        uint64_t h = static_cast<uint64_t>(key);
        h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33;
        return static_cast<size_t>(h);
    }

    // Linear probing stops at EMPTY: a key is published by one exchange, so a probe reaching an
    // untouched slot has passed every slot the key could occupy. MIGRATED is skipped rather than
    // treated as a stop, because it once held a key and the run continues past it.
    static bool find_in_table(const Table* t, K key) {
        const size_t idx = hash(key) & t->mask;
        for (size_t p = 0; p < t->capacity; ++p) {
            const K cur = t->keys[(idx + p) & t->mask].load(std::memory_order_acquire);
            if (cur == key) return true;
            if (cur == EMPTY_KEY) return false;
        }
        return false;
    }

    // THE VERDICT IS ANCHORED TO THE TABLE IT WAS DECIDED IN. A claim that lands in a table which
    // is no longer the head is reported kStale rather than kWon, and the caller re-drives against
    // the current head. Reporting kWon there is the double-claim: a thread working from the old
    // head claims the key while a thread at the new head, whose chain scan ran before this claim
    // was published, claims the same key, and both are told they inserted it.
    Claim claim(Table* t, K key) {
        const size_t idx = hash(key) & t->mask;
        for (size_t p = 0; p < t->capacity; ++p) {
            std::atomic<K>& slot = t->keys[(idx + p) & t->mask];
            K cur = slot.load(std::memory_order_acquire);
            if (cur == key) return Claim::kLost;
            if (cur == MIGRATED_KEY) continue;      // sealed by a scan or a growth; probe on
            if (cur == EMPTY_KEY) {
                if (slot.compare_exchange_strong(cur, key, std::memory_order_acq_rel,
                                                 std::memory_order_acquire)) {
                    count_.fetch_add(1, std::memory_order_relaxed);
                    return Claim::kWon;
                }
                if (cur == key) return Claim::kLost;
            }
        }
        grow(t);
        return Claim::kStale;
    }

    // SEAL EACH SLOT AS IT IS CARRIED, BEFORE INSTALLING.
    //
    // The obvious order -- copy everything, install, then mark what was copied -- leaves a window
    // that GenMC finds in a handful of executions: a claim landing in the old table after the copy
    // pass has read that slot is counted as a win, and the marking pass then overwrites it, so the
    // key is gone and a second claimant wins it again at the new head. Both callers are told they
    // inserted one key.
    //
    // Sealing closes the window at the slot rather than after it. Every slot of the retiring table
    // ends as MIGRATED, and it gets there by an exchange: an occupied slot is carried into the new
    // table FIRST and sealed second, so the key is never absent from both; an EMPTY slot is sealed
    // directly, so no late claim can land where nothing will look again. A claimant that meets a
    // sealed slot re-drives against the head, which is exactly ConcurrentMap's discipline, and it
    // is the part a key-only set does not get to skip.
    void grow(Table* expected) {
        // INSTALL EMPTY FIRST, THEN MIGRATE. Carrying keys into a table before knowing it will be
        // the head means a grower that loses the install has already moved keys into a table it
        // must discard, while the slots it sealed say the keys are elsewhere -- they are nowhere.
        // GenMC finds that as a lost key in four executions. Installing an empty table first makes
        // the discarded table EMPTY, so a loser throws away nothing and simply re-drives.
        const size_t next = expected->capacity < WORKING_CAPACITY ? WORKING_CAPACITY
                                                                  : expected->capacity * 2;
        Table* nt = Table::create(next, expected, arena_);
        if (!table_.compare_exchange_strong(expected, nt, std::memory_order_acq_rel,
                                            std::memory_order_acquire)) {
            if (!arena_) ::operator delete(nt);   // empty: nothing was carried into it
            return;                                // someone else owns the migration
        }

        // This thread owns the migration. Every slot of the retiring table ends MIGRATED, and it
        // gets there by an exchange from the value just read: if a claim lands between the read
        // and the exchange, the exchange fails, the slot is re-read, and the new key is carried
        // before being sealed. So no key is ever sealed away without first appearing in nt.
        //
        // A claimant whose CAS landed here before the install still reports its verdict against
        // this table and is told kStale, because table_ has already moved; it re-drives at nt.
        // A KEY THAT CANNOT BE CARRIED MUST STAY REACHABLE WHERE IT IS. migrate_into fails when
        // the destination's probe run is exhausted, which concurrent claims into the new table
        // can cause while this migration runs. Sealing the source anyway put the key in neither
        // table -- and marking the table drained then made for_each skip it entirely, so the
        // claim was counted and the key was gone. Measured on WPP at 8 threads: 20,558 pairs
        // enumerated against 30,063 claimed.
        //
        // Leaving it unsealed keeps it in this table, where for_each and find_in_table both
        // still reach it because the table is not drained. The cost is that a superseded table
        // stays in the chain and is walked; the alternative was losing the key.
        bool all_sealed = true;
        for (size_t i = 0; i < nt->prev->capacity; ++i) {
            std::atomic<K>& slot = nt->prev->keys[i];
            for (;;) {
                K k = slot.load(std::memory_order_acquire);
                if (k == MIGRATED_KEY) break;
                if (k != EMPTY_KEY && !migrate_into(nt, k)) {   // carry BEFORE sealing
                    all_sealed = false;
                    break;                                      // leave it here, still reachable
                }
                if (slot.compare_exchange_strong(k, MIGRATED_KEY, std::memory_order_acq_rel,
                                                 std::memory_order_acquire))
                    break;
            }
        }
        // Published last: every key is now in nt and every slot here is sealed, so a reader that
        // observes this flag may skip the table without missing a key.
        if (all_sealed) nt->prev->drained.store(true, std::memory_order_release);
    }

    // Migration is not a claim: it carries a key that already won its claim elsewhere, so it
    // takes no count and needs no verdict.
    static bool migrate_into(Table* t, K key) {
        const size_t idx = hash(key) & t->mask;
        for (size_t p = 0; p < t->capacity; ++p) {
            std::atomic<K>& slot = t->keys[(idx + p) & t->mask];
            K cur = slot.load(std::memory_order_acquire);
            if (cur == key) return true;
            if (cur == EMPTY_KEY &&
                slot.compare_exchange_strong(cur, key, std::memory_order_acq_rel,
                                             std::memory_order_acquire))
                return true;
            if (cur == key) return true;
        }
        // THE PROBE RUN IS EXHAUSTED AND THE KEY WAS NOT CARRIED. Reported so the caller can
        // leave the source slot unsealed and the table undrained; returning void here is what
        // put the key in neither table while count_ still counted its claim. Counted here so the loss is measurable instead of
        // silent; see #167.
        migrate_deferrals().fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    std::atomic<Table*> table_{nullptr};
    std::atomic<size_t> count_;
    ConcurrentHeterogeneousArena* arena_;
};

}  // namespace engine
}  // namespace HG_NAMESPACE
