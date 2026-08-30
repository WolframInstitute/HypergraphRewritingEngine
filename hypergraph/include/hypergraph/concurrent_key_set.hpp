#pragma once
#include "hgcommon/core.hpp"
#include "hgcommon/namespace.hpp"

#include <atomic>
#include <cstddef>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <new>
#if HG_ENGINE_STATS
#include <unordered_map>
#include <unordered_set>
#endif

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
// DEFINED HERE, and it has to be: verification/genmc/key_set_exactly_once.cpp,
// key_set_distinct_keys_across_growth.cpp and key_set_enumeration.cpp each compile this
// header on its own and link no library, and the migrate path they check calls this. An
// out-of-line definition would leave it undefined at link for all three.
inline std::atomic<uint64_t>& migrate_deferrals() {
    static std::atomic<uint64_t> n{0};
    return n;
}
// Stats builds only: every insert call and every win, process-wide over every set. The ratio
// is what decides whether a thread-local filter in front of a set is worth its cost -- a set
// whose inserts are mostly repeats spends its cross-CCX traffic on keys it already holds.
inline std::atomic<uint64_t>& key_set_insert_calls() {
    static std::atomic<uint64_t> n{0};
    return n;
}
inline std::atomic<uint64_t>& key_set_insert_wins() {
    static std::atomic<uint64_t> n{0};
    return n;
}
// Stats builds only: how many repeats were of a key THIS thread had inserted into THIS set
// before. A thread-local filter in front of a set can remove only those; a repeat first made by
// another worker is found by the shared probe and nothing local knows it.
inline std::atomic<uint64_t>& key_set_insert_same_worker_repeats() {
    static std::atomic<uint64_t> n{0};
    return n;
}
#if HG_ENGINE_STATS
inline bool key_set_note_local(const void* set, uint64_t key) {
    static thread_local std::unordered_map<const void*, std::unordered_set<uint64_t>> seen;
    return !seen[set].insert(key).second;   // true when this thread inserted it before
}
#endif

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

        // Set once EVERY table reachable through prev is drained, so an insert can skip the
        // chain walk entirely rather than testing each table's drained flag in turn. The walk
        // is the top cost inside insert on a set that has grown: measured at 214 million
        // instructions on disc-l3a2g2r2 depth 3, where the chain is about twenty tables and
        // every one of them is drained.
        //
        // Only ever set, never cleared, and only after the table below has been drained, so a
        // reader that observes it may skip the walk without missing a key. A reader that
        // observes it stale-false walks, which is what the flag replaces rather than changes.
        std::atomic<bool> chain_clear{false};

        static Table* create(size_t cap, Table* prev, ConcurrentHeterogeneousArena* arena) {
            size_t actual = 1;
            while (actual < cap) actual <<= 1;
            const size_t bytes = sizeof(Table) + sizeof(std::atomic<K>) * actual;
            bool  zero = false;
            void* mem = arena ? arena->allocate_raw(bytes, alignof(Table), &zero)
                              : ::operator new(bytes);
            // The first table has nothing below it, so its chain is clear by construction.
            Table* t = new (mem) Table{actual, actual - 1, prev, nullptr, {false},
                                       {prev == nullptr}};
            t->keys = reinterpret_cast<std::atomic<K>*>(
                reinterpret_cast<char*>(mem) + sizeof(Table));
            // Bytes the arena knows to be zero already hold the empty sentinel in every slot
            // when the sentinel is zero bytes; the fill was half of all zeroing on wpp depth 7
            // (callgrind: 55.0 M instructions in 1,273 table creations).
            if (!(zero && EMPTY_KEY == K{})) {
                for (size_t i = 0; i < actual; ++i) new (&t->keys[i]) std::atomic<K>(EMPTY_KEY);
            }
            return t;
        }
    };

    // `working_capacity` is the size the FIRST growth jumps to. It is a parameter for the same
    // reason ConcurrentMap's is: the migration exchanges an atomic per slot of the retiring
    // table and the successor is initialised slot by slot, so growing into a 1024-wide table
    // gives a model checker thousands of operations to interleave and the double-claim shapes
    // cannot be exhausted. The property -- install, carry, seal, chain-scan, re-drive, and
    // exactly one caller told it inserted -- does not depend on the width; the capacity decides
    // probe-run length, and a harness holds few enough keys that the runs are trivial either
    // way. So the constant is a parameter and the default is unchanged.
    explicit ConcurrentKeySet(size_t initial_capacity = DEFAULT_INITIAL_CAPACITY,
                              ConcurrentHeterogeneousArena* arena = nullptr,
                              size_t working_capacity = WORKING_CAPACITY)
        : arena_(arena), working_capacity_(working_capacity), count_(0) {
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
    // A key equal to a reserved sentinel cannot be stored: EMPTY_KEY would leave the slot
    // reading as free and MIGRATED_KEY as carried, so the claim would silently never exist.
    // The same mistake cost four investigations on ConcurrentMap before its guard existed;
    // this is that guard, for this container. The operation name goes in the thrown message
    // because the throw is most likely seen in a release build, where knowing the call site
    // is most of the diagnosis.
    static void reject_sentinel_key(K key, const char* op) {
        if (key == EMPTY_KEY || key == MIGRATED_KEY) {
#ifdef HG_VERIFICATION
            // A model checker faults on the throw's typeinfo while initialising; an assert is
            // the same precondition stated as a safety property it can report.
            (void)op;
            assert(false && "ConcurrentKeySet: key collides with a reserved sentinel");
#else
            throw std::logic_error(
                std::string("ConcurrentKeySet::") + op +
                ": key collides with the EMPTY/MIGRATED sentinel. Offset dense ids or use a "
                "reserved band.");
#endif
        }
    }

    bool insert(K key) {
        reject_sentinel_key(key, "insert");
        HG_STAT(key_set_insert_calls().fetch_add(1, std::memory_order_relaxed));
        HG_STAT(if (key_set_note_local(this, static_cast<uint64_t>(key)))
                    key_set_insert_same_worker_repeats().fetch_add(1, std::memory_order_relaxed));
        for (;;) {
            Table* head = table_.load(std::memory_order_acquire);
            // WHEN TO GROW. A probe run that walked past kProbeLimit slots raised want_grow_
            // (claim()); that is the only trigger the release build has, and the only shared
            // word it reads per insert is that flag, written once per growth rather than once
            // per win. Builds that keep the exact count (stats and verification) also grow at
            // the load factor, so the harnesses drive growth the way they always did.
            //
            // ONE GROWER PER TRIGGER, AND NOBODY WAITS -- the same election ConcurrentMap makes.
            // grow() builds its replacement BEFORE the exchange that installs it, so every thread
            // that took the trigger would otherwise build one and all but one abandon it in an
            // arena that cannot free it. A thread that does not take the ticket re-drives at the
            // head rather than waiting; a probe exhausted on a table being sealed re-drives too
            // (claim()), which is what makes progress independent of who holds the ticket.
            if (want_grow_.load(std::memory_order_relaxed) || count_past_threshold(head)) {
                grow_guarded(head, /*forced=*/false);
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
            if (!head->chain_clear.load(std::memory_order_acquire)) {
                for (Table* t = head->prev; t && !in_chain; t = t->prev)
                    if (!t->drained.load(std::memory_order_acquire)) in_chain = find_in_table(t, key);
            }
            if (in_chain) return false;

            const Claim c = claim(head, key);
            if (c == Claim::kWon)  return true;
            if (c == Claim::kLost) return false;
            // kStale: this table filled under us. Re-drive from the current head.
        }
    }

    bool contains(K key) const {
        reject_sentinel_key(key, "contains");
        // OLDEST TABLE FIRST. During a growth a key moves old -> new: the carry lands in the
        // successor BEFORE the source slot seals, and the seal is the release this walk's
        // acquire pairs with -- so a probe that meets the key's slot already MIGRATED is
        // ordered after the carry, and the destination, walked LATER in this order, shows the
        // key. Newest-first visits each table on the wrong side of the hand-off (the head
        // before the carry lands, the source after its seal) and reports a settled key absent.
        // insert()'s chain scan survives the same window only because its head claim re-probes
        // the run the carry lands in and meets the key there; contains has no claim, so the
        // walk order is the whole answer. Calibrated: key_set_contains_during_growth reports
        // the violation against the newest-first walk in its first explored execution.
        //
        // The retry: carries only land in the table that is head when their growth installs
        // it, so with the head unchanged across a walk the destination of every in-flight
        // carry is the table this order probes last. A head that MOVED mid-walk can have
        // received and re-emitted the key (a double hop into a successor the snapshot never
        // reaches), so the verdict "absent" stands only when the head it was decided against
        // is still the head. Growths are finite -- capacity is monotone -- so this loop
        // terminates.
        for (;;) {
            Table* head = table_.load(std::memory_order_acquire);
            if (find_oldest_first(head, key)) return true;
            if (table_.load(std::memory_order_acquire) == head) return false;
        }
    }

    // The claim counter. Drives the growth decision and nothing else, because it is a VERDICT
    // tally: it counts inserts that reported a win, not keys the container can hand back. Those
    // two quantities diverged once already -- migrate_into dropped a key while count_ still
    // counted its claim (f694c062) -- so anything reporting a SET SIZE to a caller uses
    // count_enumerated() below, which walks what for_each would emit.
    // The exact win count where a build keeps one (stats, verification); the release build
    // keeps none and walks the tables, which no engine path calls.
    size_t size() const {
#if HG_ENGINE_STATS || defined(HG_VERIFICATION)
        return count_.load(std::memory_order_relaxed);
#else
        return count_enumerated();
#endif
    }

    // The number of keys `for_each` will emit. O(capacity) over the chain, so it is for
    // per-run observables rather than hot paths -- and it cannot disagree with enumeration,
    // because it IS enumeration.
    size_t count_enumerated() const {
        size_t n = 0;
        for_each([&](K) { ++n; });
        return n;
    }

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

    // The chain, probed oldest table first; see contains() for why that order is the
    // correctness condition. Recursion depth is the number of live tables: growths double the
    // capacity, so it is logarithmic in the largest table plus any unsealed stragglers.
    static bool find_oldest_first(const Table* t, K key) {
        if (!t) return false;
        if (find_oldest_first(t->prev, key)) return true;
        return !t->drained.load(std::memory_order_acquire) && find_in_table(t, key);
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
        // The run length that signals load counts occupied slots only. A sealed (MIGRATED)
        // slot is not load: a growth in progress seals the old table slot by slot, and a claim
        // that skipped a run of sealed slots to reach a still-empty one would otherwise report
        // a long run on a table about to be retired, and the freshly installed successor would
        // grow at once -- measured as a growth per seal, doubling to a 512M-entry table.
        size_t occupied = 0;
        for (size_t p = 0; p < t->capacity; ++p) {
            std::atomic<K>& slot = t->keys[(idx + p) & t->mask];
            K cur = slot.load(std::memory_order_acquire);
            if (cur == key) return Claim::kLost;
            if (cur == MIGRATED_KEY) continue;      // sealed by a scan or a growth; probe on
            if (cur == EMPTY_KEY) {
                if (slot.compare_exchange_strong(cur, key, std::memory_order_acq_rel,
                                                 std::memory_order_acquire)) {
                    note_win(occupied);
                    HG_STAT(key_set_insert_wins().fetch_add(1, std::memory_order_relaxed));
                    return Claim::kWon;
                }
                if (cur == key) return Claim::kLost;
            }
            ++occupied;
        }
        grow_guarded(t, /*forced=*/true);
        return Claim::kStale;
    }

    // A win after a probe run of `probe` slots. The run length is the load signal: linear
    // probing's expected unsuccessful run is (1 + 1/(1-a)^2)/2 at load a, so kProbeLimit slots
    // is reached around a = 0.8 and the next insert grows the table. The flag is written only
    // when it is not already set, so a growth costs one shared write, not one per win. Builds
    // that keep the exact count add to it here; the release build does not, and that removed
    // count is the one line every worker on every L3 domain wrote per win (measured on the box,
    // wpp depth 7, 16 threads: this insert was 21-23% of the fills served from another domain).
    static constexpr size_t kProbeLimit = 32;
    void note_win(size_t probe) {
        if (probe >= kProbeLimit && !want_grow_.load(std::memory_order_relaxed))
            want_grow_.store(true, std::memory_order_relaxed);
#if HG_ENGINE_STATS || defined(HG_VERIFICATION)
        count_.fetch_add(1, std::memory_order_relaxed);
#endif
    }
    bool count_past_threshold(const Table* head) const {
#if HG_ENGINE_STATS || defined(HG_VERIFICATION)
        return count_.load(std::memory_order_relaxed) > head->capacity * LOAD_FACTOR_THRESHOLD;
#else
        (void)head;
        return false;
#endif
    }

    // Every table allocation goes through one ticket. grow() seals the old table's empty
    // slots before it installs the new one, so a claim that reaches the old table during the
    // seal walks it to exhaustion; letting that claim allocate its own successor left a
    // full-size table in the arena for every such thread whose install lost (measured with a
    // lagging count on CausalDeterminism.NonQuotientFullyDeterministic: 418 lost installs,
    // 31 of them 16M-entry tables, 10 GB). Without the ticket the caller re-drives at the
    // head, which is the same loop the trigger branch spins in.
    // `forced` is a probe exhausted on t itself (nothing left to claim); otherwise the
    // trigger is re-read under the ticket, against t, because a thread that read the flag
    // before the growth it asked for completed would otherwise take the freed ticket with the
    // successor as its head and grow that too (measured: one growth per growth, doubling to a
    // 268M-entry table at 13,716 keys).
    void grow_guarded(Table* t, bool forced) {
        if (growing_.exchange(true, std::memory_order_acquire)) return;
        struct ReleaseTicket {
            std::atomic<bool>& flag;
            ~ReleaseTicket() { flag.store(false, std::memory_order_release); }
        } release{growing_};
        if (table_.load(std::memory_order_acquire) != t) return;
        if (!forced && !want_grow_.load(std::memory_order_relaxed) && !count_past_threshold(t)) return;
        grow(t);
        want_grow_.store(false, std::memory_order_relaxed);
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
        // CLOSE THE RETIRING TABLE TO NEW CLAIMS BEFORE ITS SUCCESSOR EXISTS.
        //
        // Carrying a slot and sealing it in one step keeps the key from being LOST, and that is
        // all it does. It leaves every slot the carry has not reached yet still EMPTY, so a
        // claimant holding this table as its head can still win one AFTER the successor is
        // installed -- and a rival that scanned this table a moment earlier, found nothing, and
        // claimed at the new head has already been told it inserted. Both are told they inserted
        // one key. Losing a key and double-reporting a claim are different failures and the
        // carry-then-seal exchange only addresses the first.
        //
        // Sealing every EMPTY slot BEFORE the install removes the second. After this pass the
        // table has no EMPTY slot, so a probe run for an absent key exhausts and re-drives at the
        // head instead of claiming here; and every key that did win here was published before any
        // successor existed, which is before any rival could load the new head, so a rival's chain
        // scan sees it and reports it lost rather than claiming it again. The pass moves nothing,
        // so it is safe ahead of the install: a grower that loses the install has only closed
        // slots of a table that is retiring regardless.
        //
        // Measured: qc_applied_ (the per-(instance, match) claim gating qr_apply) reported a win
        // twice for one key on about one run in a hundred at 32 threads, which the determinism
        // gate saw as one extra raw event, one extra causal edge and its branchial pairs --
        // claims 18313 against 18312 with the claim tally one ahead of the keys the set could
        // enumerate.
        for (size_t i = 0; i < expected->capacity; ++i) {
            std::atomic<K>& slot = expected->keys[i];
            K k = slot.load(std::memory_order_acquire);
            if (k == EMPTY_KEY)
                slot.compare_exchange_strong(k, MIGRATED_KEY, std::memory_order_acq_rel,
                                             std::memory_order_acquire);
        }

        // INSTALL EMPTY FIRST, THEN MIGRATE. Carrying keys into a table before knowing it will be
        // the head means a grower that loses the install has already moved keys into a table it
        // must discard, while the slots it sealed say the keys are elsewhere -- they are nowhere.
        // GenMC finds that as a lost key in four executions. Installing an empty table first makes
        // the discarded table EMPTY, so a loser throws away nothing and simply re-drives.
        const size_t next = expected->capacity < working_capacity_ ? working_capacity_
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
        // Published after the flag above, so a reader seeing a clear chain sees the drain that
        // makes it clear. A table left unsealed keeps every table above it walking, which is the
        // conservative direction.
        if (all_sealed && nt->prev->chain_clear.load(std::memory_order_acquire))
            nt->chain_clear.store(true, std::memory_order_release);
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
        HG_STAT(migrate_deferrals().fetch_add(1, std::memory_order_relaxed));
        return false;
    }

    // THE HOTTEST READ AND THE HOTTEST WRITE DO NOT SHARE A LINE, for the reason ConcurrentMap
    // gives at the same place. table_ is loaded by every operation and written only when a growth
    // installs one; count_ is a read-modify-write on every accepted key, which takes its line
    // EXCLUSIVE and invalidates it in every other core's cache. Adjacent, each accepted key
    // evicts the pointer every other worker is about to load. This set backs the causal
    // seen-sets, which the instruction profile puts at 7.5% of the engine, so the interference is
    // on a path that runs constantly.
    std::atomic<Table*> table_{nullptr};
    ConcurrentHeterogeneousArena* arena_;
    size_t working_capacity_ = WORKING_CAPACITY;

    alignas(64) std::atomic<size_t> count_;
    // The growth ticket; see insert. Not a lock -- a thread that fails to take it proceeds to its
    // probe without waiting, and the re-drive on an exhausted run is not ticketed at all.
    std::atomic<bool> growing_{false};
    // Raised by a win whose probe run reached kProbeLimit; cleared by the growth it triggers.
    std::atomic<bool> want_grow_{false};
};

// INDEPENDENT TABLES CHOSEN BY THE KEY, for a set whose insert rate is the workload.
//
// ConcurrentKeySet is one table, so every worker inserting probes slots every other worker is
// writing. Those probes are plain LOADS of contended lines, which is why neither the growth
// election nor the count_/table_ line separation removed the cost: there is no single hot field
// left, the traffic is the table itself. MEASURED on cycle4, four workers spanning two L3
// instances of an EPYC 9174F: insert is 12.9% of the run at one thread and 41.2% at four, and
// the run takes 44.0 ms against 19.8 ms serial.
//
// A KEY ALWAYS SELECTS THE SAME SHARD, so dedup stays EXACT -- two workers racing the same key
// meet in the same table and exactly one wins, which is the property the claim depends on.
// Sharding by WORKER would break that, because the two paths into qc_apply can offer one
// (instance, match) pair from different workers and both would win.
//
// This is the vendor-agnostic half of the fix. Confining workers to one cache domain also
// removes the penalty (measured: four workers inside one L3 instance show none at all) but
// needs the topology, differs per part, and caps the run at one domain's cores.
//
// The shard index takes the HIGH bits: keys here are already avalanched (qr_apply_key is an FNV
// mix, qc_pair_key packs two dense ids), and the low bits are what the tables' own probe uses,
// so taking the same bits for both would correlate the shard with the slot.
// The shard count of the sharded key sets, and so the size of every object that holds one
// (Hypergraph, CausalGraph). The engine harnesses define it small (verification/genmc/engine_*.cpp)
// so the checker zero-fills a few shards per replay instead of 64; the shipped value is below.
#ifndef HG_KEY_SET_SHARDS
#define HG_KEY_SET_SHARDS 64
#endif
template <typename K = uint64_t,
          K EMPTY_SENTINEL = K{0},
          K LOCKED_SENTINEL = ~K{0},
          size_t SHARDS = HG_KEY_SET_SHARDS>
class ShardedKeySet {
    static_assert((SHARDS & (SHARDS - 1)) == 0, "SHARDS must be a power of two");

public:
    using Shard = ConcurrentKeySet<K, EMPTY_SENTINEL, LOCKED_SENTINEL>;

    // BY VALUE, not pointers: the engine is de-heaped and a shard array of `new` would put 64
    // allocations back per set. DEFAULT_INITIAL_CAPACITY is 1, so an unused shard costs a table
    // of one slot and the array is cheap until keys arrive.
    ShardedKeySet() = default;

    // The same spelling every other arena-backed member uses, so an owner seats this exactly as
    // it seats a plain ConcurrentKeySet and nothing has to know it is sharded.
    void set_arena(ConcurrentHeterogeneousArena* arena) {
        for (Shard& s : shard_) s.set_arena(arena);
    }

    ShardedKeySet(const ShardedKeySet&) = delete;
    ShardedKeySet& operator=(const ShardedKeySet&) = delete;

    bool insert(K key) { return shard_[index(key)].insert(key); }

    // A key lives in exactly one shard, so this asks exactly one of them.
    bool contains(K key) const { return shard_[index(key)].contains(key); }

    template <typename F>
    void for_each(F&& f) const { for (const Shard& s : shard_) s.for_each(f); }

    size_t size() const {
        size_t n = 0;
        for (const Shard& s : shard_) n += s.size();
        return n;
    }

    size_t count_enumerated() const {
        size_t n = 0;
        for (const Shard& s : shard_) n += s.count_enumerated();
        return n;
    }

private:
    static size_t index(K key) {
        return static_cast<size_t>((static_cast<uint64_t>(key) >> 40)) & (SHARDS - 1);
    }
    Shard shard_[SHARDS];
};

}  // namespace engine
}  // namespace HG_NAMESPACE
