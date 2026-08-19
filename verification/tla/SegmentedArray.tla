------------------------------ MODULE SegmentedArray ------------------------------
(* SegmentedArray's ordering invariant, and the completeness that rests on it.

   WHY THIS IS TLA+ AND NOT GENMC. Every other concurrent structure here is model-checked
   against its own header by a GenMC harness. GenMC v0.17.0 cannot execute this one: merely
   CONSTRUCTING a SegmentedArray<uint64_t,4> segfaults it inside SAddrAllocator::allocate, in
   stack and in static storage, with and without the class's throw path. Isolated: a
   hand-written std::atomic<uint64_t*>[4] plus the same hgcommon::ctz64 call verifies in one
   execution and 0.00s, so it is this class the tool cannot take, not the includes and not the
   intrinsic. tools/safe_verify.sh already says what to do about an argument GenMC cannot
   finish: move it to TLA+, which is state-bounded rather than execution-bounded.

   WHAT THE STRUCTURE IS. An append-only array over a directory of segments. emplace claims an
   index with a fetch_add, then writes into the segment that index falls in, creating the
   segment if absent, and finally advances a high-water mark `count`. A reader walks
   [0, count) and resolves each index through the directory.

   THE INVARIANT. The header states it: "Segments are therefore created in order, which gives:
   if a segment exists, so do all below it." It is what makes a walk safe, and the reason is
   the high-water mark: a thread claiming an index in segment 2 advances `count` past segment
   1's ENTIRE range, so if segment 1 were absent at that moment the walk would resolve an index
   inside it to a null segment. The mark says how far to walk; it does not say the directory is
   dense, and DenseBelowCount is what does.

   CreateInOrder is the shipped protocol: get_or_create_segment creates every segment below the
   one asked for before the one asked for. FALSE models creating only the segment asked for,
   which is the defect this invariant excludes and which TLC reports. *)
EXTENDS Integers, FiniteSets, TLC

CONSTANTS Threads,        (* the emplacing threads *)
          NumSegments,    (* directory size *)
          SegSize,        (* elements per segment *)
          CreateInOrder   (* TRUE = shipped, FALSE = create only the segment asked for *)

Capacity == NumSegments * SegSize
SegOf(i) == i \div SegSize

VARIABLES claim,     (* next index to hand out; the fetch_add counter *)
          count,     (* the published high-water mark *)
          exists,    (* exists[s] = segment s has been created *)
          written,   (* the set of indices whose element has been written *)
          held       (* held[t] = the index thread t claimed, or -1 between claims *)

vars == <<claim, count, exists, written, held>>

Init ==
    /\ claim   = 0
    /\ count   = 0
    /\ exists  = [s \in 0..(NumSegments - 1) |-> FALSE]
    /\ written = {}
    /\ held    = [t \in Threads |-> -1]

(* fetch_add: the index is handed out and nothing else happens yet. *)
Claim(t) ==
    /\ held[t] = -1
    /\ claim < Capacity
    /\ held'  = [held EXCEPT ![t] = claim]
    /\ claim' = claim + 1
    /\ UNCHANGED <<count, exists, written>>

(* Create the segment the claimed index falls in. Shipped: every segment below it first. *)
Create(t) ==
    /\ held[t] # -1
    /\ LET s == SegOf(held[t]) IN
       /\ ~exists[s]
       /\ exists' = IF CreateInOrder
                    THEN [k \in DOMAIN exists |-> IF k <= s THEN TRUE ELSE exists[k]]
                    ELSE [exists EXCEPT ![s] = TRUE]
    /\ UNCHANGED <<claim, count, written, held>>

(* Write the element, then publish by advancing the high-water mark. The mark is advanced
   independently by each emplace -- it is not a per-index flag -- so it can pass indices whose
   own write is still outstanding, which is exactly what the header's contract warns of and
   why the completeness property below is stated at quiescence. *)
Write(t) ==
    /\ held[t] # -1
    /\ exists[SegOf(held[t])]
    /\ held[t] \notin written
    /\ written' = written \cup {held[t]}
    /\ UNCHANGED <<claim, count, exists, held>>

Publish(t) ==
    /\ held[t] # -1
    /\ held[t] \in written
    /\ count'  = IF held[t] + 1 > count THEN held[t] + 1 ELSE count
    /\ held'   = [held EXCEPT ![t] = -1]
    /\ UNCHANGED <<claim, exists, written>>

Next == \E t \in Threads : Claim(t) \/ Create(t) \/ Write(t) \/ Publish(t)

Spec == Init /\ [][Next]_vars

(* THE INVARIANT. Every index the mark admits resolves to a segment that exists. This is the
   property a walk over [0, count) depends on, and it holds at every reachable state, not only
   at quiescence: a reader may run at any moment. *)
DenseBelowCount ==
    \A i \in 0..(Capacity - 1) : (i < count) => exists[SegOf(i)]

(* COMPLETENESS, at quiescence. Once no thread holds a claim, everything handed out has been
   written and the mark admits all of it -- so a walk reaches every element exactly once. *)
Quiescent == \A t \in Threads : held[t] = -1

CompleteWhenQuiescent ==
    Quiescent => /\ count = claim
                 /\ \A i \in 0..(Capacity - 1) : (i < claim) => i \in written

=============================================================================
