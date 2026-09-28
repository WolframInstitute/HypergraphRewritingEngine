------------------------- MODULE QuotientContinuation -------------------------
(* The quotient replay continued past its depth bound reaches what one run to the final bound
   reaches.

   THE PROTOCOL, from hypergraph/src/hypergraph.cpp and parallel_evolution.cpp. The replay pairs
   every INSTANCE of a canonical class (one raw occurrence at one depth) with every MATCH captured
   on the class's expanded representative. Each pairing is an application: claimed once per
   (instance, match), it creates an instance of the match's target class one depth deeper.
     - qc_add_instance publishes the instance, and if its depth is below the bound, scans the
       class's captured matches and applies each. A point (class, depth) created at or past the
       bound is pushed on qc_blocked_.
     - qc_capture_expansion publishes the match, then scans the class's instances at every depth
       below the bound and applies each.
     - Between runs, with no worker running, raise_quotient_max_steps raises the bound;
       evolve_more enumerates qc_blocked_ for points with old <= depth < new and submits one
       quotient_redrive_point job per point. A redrive applies every instance at the point to
       every captured match of its class. Redrives run in the same wait as the resumed
       exploration, so they race new instances and new captures.
   Captures happen as the exploration reaches a class, so a match may be captured in any run.

   PROPERTIES, checked when the last run is quiescent:
     Complete:    every instance below the bound has been applied to every match of its class.
     Exact:       the instances are the paths one run to the final bound creates.
     NoneBeyond:  no application is at or past the bound in force.
   Publish and scan are separate steps; each pending step fires atomically. The publish-then-scan
   ordering under RC11 is verification/genmc/quotient_instance_match_rendezvous.cpp, and the
   claim is the same harness's claim-once property.

   CALIBRATION. PushBlocked = FALSE never pushes a point on qc_blocked_: the instances standing
   at the old bound are never applied to the matches captured before the raise. *)
EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS Runs,          (* run i evolves to depth bound i: Runs = 3 is two continuations *)
          PushBlocked    (* TRUE: shipped; FALSE: the calibration *)

Bounds == [i \in 1..Runs |-> i]

(* Two classes. A has matches mAB (to B) and mAA (to A); B has mBA (to A) and mBB (to B). *)
Classes == {"A", "B"}
Root    == "A"
Matches == {"mAB", "mAA", "mBA", "mBB"}
From == [m \in Matches |-> IF m \in {"mAB", "mAA"} THEN "A" ELSE "B"]
To   == [m \in Matches |-> IF m \in {"mAB", "mBB"} THEN "B" ELSE "A"]

(* An instance is named by its path of matches from the root, so two instances at one point are
   distinct values and an application is keyed on (path, match). *)
ClassOfPath(p) == IF p = <<>> THEN Root ELSE To[p[Len(p)]]

Final == Bounds[Len(Bounds)]

VARIABLES run,        (* index into Bounds of the run in progress; Len(Bounds) + 1 when done *)
          insts,      (* published instances, as paths *)
          points,     (* <<class, depth>> points created *)
          blocked,    (* qc_blocked_ *)
          captured,   (* published matches *)
          claimed,    (* <<path, match>> applications that won the claim *)
          pending     (* steps enqueued and not yet fired *)

vars == <<run, insts, points, blocked, captured, claimed, pending>>

Bound == Bounds[run]

(* Every pending step has the same fields, so TLC can compare any two. *)
StepRec(kind, p, m, c, d) == [k |-> kind, p |-> p, m |-> m, c |-> c, d |-> d]
ScanMatchRec(m) == StepRec("match", <<>>, m, "-", 0)
ScanInstRec(p)  == StepRec("inst", p, "-", "-", 0)
Apply(p, m)     == StepRec("apply", p, m, "-", 0)
RedriveRec(pt)  == StepRec("redrive", <<>>, "-", pt[1], pt[2])

Init == /\ run = 1
        /\ insts = {<<>>}
        /\ points = {<<Root, 0>>}
        /\ blocked = {}
        /\ captured = {}
        /\ claimed = {}
        /\ pending = {ScanInstRec(<<>>)}

Running == run <= Len(Bounds)

(* qc_capture_expansion, publish half: the match is pushed on its class's list. *)
Capture(m) ==
    /\ Running
    /\ m \notin captured
    /\ captured' = captured \cup {m}
    /\ pending' = pending \cup {ScanMatchRec(m)}
    /\ UNCHANGED <<run, insts, points, blocked, claimed>>

(* qc_capture_expansion, scan half: every instance of the class below the bound. *)
ScanMatch(m) ==
    /\ pending' = (pending \ {ScanMatchRec(m)})
                  \cup {Apply(p, m) : p \in {q \in insts : ClassOfPath(q) = From[m] /\ Len(q) < Bound}}
    /\ UNCHANGED <<run, insts, points, blocked, captured, claimed>>

(* qc_add_instance, scan half: every captured match of the class. Enqueued only below the bound. *)
ScanInst(p) ==
    /\ pending' = (pending \ {ScanInstRec(p)})
                  \cup {Apply(p, m) : m \in {n \in captured : From[n] = ClassOfPath(p)}}
    /\ UNCHANGED <<run, insts, points, blocked, captured, claimed>>

(* qr_apply: the claim, then qc_add_instance's publish half for the child. *)
DoApply(p, m) ==
    LET c  == Append(p, m)
        pt == <<To[m], Len(c)>>
        newPoint == pt \notin points
    IN /\ IF <<p, m>> \in claimed
          THEN UNCHANGED <<insts, points, blocked, claimed>> /\ pending' = pending \ {Apply(p, m)}
          ELSE /\ claimed' = claimed \cup {<<p, m>>}
               /\ insts' = insts \cup {c}
               /\ points' = points \cup {pt}
               /\ blocked' = IF newPoint /\ Len(c) >= Bound /\ PushBlocked
                             THEN blocked \cup {pt} ELSE blocked
               /\ pending' = (pending \ {Apply(p, m)})
                             \cup (IF Len(c) < Bound THEN {ScanInstRec(c)} ELSE {})
       /\ UNCHANGED <<run, captured>>

(* quotient_redrive_point: every instance at the point against every captured match. *)
Redrive(pt) ==
    /\ pending' = (pending \ {RedriveRec(pt)})
                  \cup {Apply(p, m) : p \in {q \in insts : ClassOfPath(q) = pt[1] /\ Len(q) = pt[2]},
                                      m \in {n \in captured : From[n] = pt[1]}}
    /\ UNCHANGED <<run, insts, points, blocked, captured, claimed>>

Fire(s) ==
    CASE s.k = "match"   -> ScanMatch(s.m)
      [] s.k = "inst"    -> ScanInst(s.p)
      [] s.k = "apply"   -> DoApply(s.p, s.m)
      [] s.k = "redrive" -> Redrive(<<s.c, s.d>>)

Step == /\ Running
        /\ \E s \in pending : Fire(s)

(* The run is quiescent. With another bound to go, raise it and submit the redrives of the
   points the old bound left standing, enumerated before any of them runs. After the last run
   every match of a reachable class has been captured, which the exploration guarantees. *)
Continue ==
    /\ Running
    /\ pending = {}
    /\ IF run < Len(Bounds)
       THEN /\ run' = run + 1
            /\ pending' = {RedriveRec(pt) :
                             pt \in {q \in blocked : Bounds[run] <= q[2] /\ q[2] < Bounds[run + 1]}}
       ELSE /\ captured = {m \in Matches : \E p \in insts : ClassOfPath(p) = From[m] /\ Len(p) < Final}
            /\ run' = run + 1
            /\ UNCHANGED pending
    /\ UNCHANGED <<insts, points, blocked, captured, claimed>>

Next == \/ \E m \in Matches : Capture(m)
        \/ Step
        \/ Continue

Spec == Init /\ [][Next]_vars

Done == run = Len(Bounds) + 1

(* Every path of length at most n, the instances one run to bound n creates. *)
RECURSIVE PathsUpTo(_)
PathsUpTo(n) == IF n = 0 THEN {<<>>}
                ELSE PathsUpTo(n - 1) \cup
                     {Append(p, m) : p \in {q \in PathsUpTo(n - 1) : Len(q) = n - 1},
                                     m \in {k \in Matches : TRUE}}
ValidPath(p) == \A i \in 1..Len(p) : From[p[i]] = ClassOfPath(SubSeq(p, 1, i - 1))

Complete == Done => \A p \in insts : \A m \in Matches :
                (Len(p) < Final /\ From[m] = ClassOfPath(p)) => <<p, m>> \in claimed

Exact == Done => insts = {p \in PathsUpTo(Final) : ValidPath(p)}

NoneBeyond == \A a \in claimed : Len(a[1]) < (IF Done THEN Final ELSE Bound)

================================================================================
