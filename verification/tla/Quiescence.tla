------------------------------ MODULE Quiescence ------------------------------
(* The job system's two global properties, checked together:

   1. Termination detection: is_quiescent() must not say "done" while work remains.
   2. No lost wakeup under the searching-worker throttle: no job is left queued while every
      worker is parked with nothing left to wake it.

   WHY TLA+ AND NOT GENMC. Both properties are predicates over every worker read while those
   workers run, which is a protocol rather than one structure's memory operations. The two GenMC
   harnesses (verification/genmc/job_system_no_lost_wakeup*.cpp) check the memory orders of the
   same handshake on the real header; this checks the protocol at every interleaving of whole
   steps, which is sequentially consistent by construction.

   THE PROTOCOL, from job_system.hpp and hgcommon/park_gate.hpp. A submit bumps `submitted`, puts
   the job in a queue, and then decides whether to wake a parked worker: it wakes one when some
   worker is idle and either no worker is searching or the submitter's backlog is a burst. A
   worker whose own deque is empty becomes a searcher (or declines to, when too many search),
   steals or gives up; a searcher that found a job leaves the count and, if it was the last
   searcher, may wake one more; a searcher that found nothing leaves the count, announces itself
   idle, samples the park word, takes one exhaustive look, and parks unless the word moved. The
   caller (a thread that is not a worker) seeds the run through the injector and then waits.
   A wake moves the park word and unblocks one sleeping worker, as FUTEX_WAKE with a count of 1.

   THE TERMINATION ORDER. is_quiescent() reads the two counters, the injector, every worker's
   deque and executing flag, and re-reads the counters. CountsFirst models that order; FALSE
   checks the queues first, which admits a scan that sees empty queues before a worker finishes
   its job and the counters agree afterwards.

   THE WAKE ORDER. A searcher leaves the count BEFORE its last look (LeaveBeforeLook). A submit
   that skipped its wake because it read the searcher in the count is then ordered before that
   searcher's last look, which finds the job. FALSE leaves the count after the look: the
   submitter reads the searcher, skips the wake, the searcher's look has already missed the job,
   and it parks with the job queued. *)
EXTENDS Integers, FiniteSets, TLC

CONSTANTS Workers, MaxJobs, CountsFirst, CheckExecuting, SubmitBeforeComplete,
          Throttle,          (* TRUE: skip the wake while a worker is searching *)
          LeaveBeforeLook    (* TRUE: a searcher leaves the count before its last look *)

VARIABLES submitted,   (* jobs ever submitted *)
          completed,   (* jobs ever completed *)
          injector,    (* jobs waiting, unclaimed *)
          deque,       (* deque[w] = jobs queued on worker w *)
          executing,   (* executing[w] = 1 while w runs a job *)
          spawns,      (* spawns[w] = children w will still submit *)
          deciding,    (* deciding[w]: w pushed a job and has not yet made its wake decision *)
          ext,         (* the caller: "seed", "decide" after a push, "wait" *)
          pc,          (* worker w's place in the idle protocol *)
          searching,   (* the searching count *)
          counted,     (* counted[w]: w is included in `searching` *)
          idle,        (* the idle count *)
          seq,         (* the park word *)
          sampled,     (* sampled[w] = the park word w read before its last look *)
          sleeping,    (* workers blocked in the futex wait *)
          phase,       (* the checker's progress: "idle", "half", "done" *)
          latched      (* what its FIRST read observed *)

vars == <<submitted, completed, injector, deque, executing, spawns, deciding, ext, pc, searching,
          counted, idle, seq, sampled, sleeping, phase, latched>>

(* pc values:
     "own"      run a job, or try the own deque
     "search"   counted in `searching`, stealing
     "leave"    found nothing; about to leave the count (LeaveBeforeLook) or announce
     "look"     announced idle, park word sampled; the exhaustive look
     "left"     LeaveBeforeLook = FALSE only: looked and found nothing, still counted
     "parked"   in the park; blocked while in `sleeping` *)
PCs == {"own", "search", "leave", "look", "left", "parked"}

MaxSeq == 2 * MaxJobs + 2

TypeOK == /\ submitted \in 0..MaxJobs
          /\ completed \in 0..MaxJobs
          /\ injector \in 0..MaxJobs
          /\ deque \in [Workers -> 0..MaxJobs]
          /\ executing \in [Workers -> 0..1]
          /\ spawns \in [Workers -> 0..1]
          /\ deciding \in [Workers -> BOOLEAN]
          /\ ext \in {"seed", "decide", "wait"}
          /\ pc \in [Workers -> PCs]
          /\ searching \in 0..Cardinality(Workers)
          /\ counted \in [Workers -> BOOLEAN]
          /\ idle \in 0..Cardinality(Workers)
          /\ seq \in 0..MaxSeq
          /\ sampled \in [Workers -> 0..MaxSeq]
          /\ sleeping \subseteq Workers
          /\ phase \in {"idle", "half", "done"}
          /\ latched \in BOOLEAN

Init == /\ submitted = 1
        /\ completed = 0
        /\ injector = 1
        /\ deque = [w \in Workers |-> 0]
        /\ executing = [w \in Workers |-> 0]
        /\ spawns = [w \in Workers |-> 0]
        /\ deciding = [w \in Workers |-> FALSE]
        /\ ext = "seed"
        /\ pc = [w \in Workers |-> "own"]
        /\ searching = 0
        /\ counted = [w \in Workers |-> FALSE]
        /\ idle = 0
        /\ seq = 0
        /\ sampled = [w \in Workers |-> 0]
        /\ sleeping = {}
        /\ phase = "idle"
        /\ latched = FALSE

QueuedJobs == injector + (LET S[ws \in SUBSET Workers] ==
                              IF ws = {} THEN 0
                              ELSE LET w == CHOOSE x \in ws : TRUE IN deque[w] + S[ws \ {w}]
                          IN S[Workers])

(* CheckExecuting is the jobs_executing half of the scan. *)
QueuesEmpty == /\ injector = 0
               /\ \A w \in Workers : deque[w] = 0 /\ (CheckExecuting => executing[w] = 0)

CountsAgree == submitted = completed

(* Wake one parked worker: move the park word, then wake one sleeper (unpark_one), if there is
   one. A worker that sampled the old word and has not yet blocked returns from its park at once.
   Bumps number at most 2 x MaxJobs (one per submit, one per job found), so the word never
   reaches MaxSeq. *)
Bump == /\ seq' = IF seq < MaxSeq THEN seq + 1 ELSE seq
        /\ IF sleeping = {} THEN UNCHANGED sleeping
                            ELSE \E v \in sleeping : sleeping' = sleeping \ {v}
NoBump == UNCHANGED <<seq, sleeping>>

NewSpawn == IF submitted < MaxJobs THEN {0, 1} ELSE {0}   (* a job may owe one child *)

(* ---- Worker: job execution ---- *)

TakeOwn(w) ==
    /\ pc[w] = "own"
    /\ executing[w] = 0
    /\ deque[w] > 0
    /\ deque' = [deque EXCEPT ![w] = deque[w] - 1]
    /\ executing' = [executing EXCEPT ![w] = 1]
    /\ \E sp \in NewSpawn : spawns' = [spawns EXCEPT ![w] = sp]
    /\ UNCHANGED <<submitted, completed, injector, deciding, ext, pc, searching, counted, idle, seq, sleeping, sampled,
                   phase, latched>>

(* THE CHILD IS SUBMITTED WHILE THE PARENT IS STILL EXECUTING (SubmitBeforeComplete). The push
   and the wake decision are two steps: other workers run between them. *)
SubmitChild(w) ==
    /\ (SubmitBeforeComplete => executing[w] = 1)
    /\ spawns[w] = 1
    /\ ~deciding[w]
    /\ submitted < MaxJobs
    /\ submitted' = submitted + 1
    /\ deque' = [deque EXCEPT ![w] = deque[w] + 1]
    /\ spawns' = [spawns EXCEPT ![w] = 0]
    /\ deciding' = [deciding EXCEPT ![w] = TRUE]
    /\ UNCHANGED <<ext, completed, injector, executing, pc, searching, counted, idle, seq, sleeping, sampled, phase,
                   latched>>

(* The wake decision, after the push and the barrier. Burst: the submitter's own deque holds
   more than twice the searchers. *)
Decide(w) ==
    /\ deciding[w]
    /\ deciding' = [deciding EXCEPT ![w] = FALSE]
    /\ IF idle > 0 /\ (~Throttle \/ searching = 0 \/ deque[w] > 2 * searching)
          THEN Bump ELSE NoBump
    /\ UNCHANGED <<ext, submitted, completed, injector, deque, executing, spawns, pc, searching, counted, idle,
                   sampled, phase, latched>>

(* The caller is a thread that is not a worker. It seeds the run through the injector, then waits
   for quiescence; the termination check runs on it, so only after seeding. It has no deque, so its
   backlog is zero: under the throttle it wakes a worker only when none searches. *)
ExtSubmit ==
    /\ ext = "seed"
    /\ submitted < MaxJobs
    /\ submitted' = submitted + 1
    /\ injector' = injector + 1
    /\ ext' = "decide"
    /\ UNCHANGED <<completed, deque, executing, spawns, deciding, pc, searching, counted, idle,
                   seq, sleeping, sampled, phase, latched>>

ExtDecide ==
    /\ ext = "decide"
    /\ ext' = "seed"
    /\ IF idle > 0 /\ (~Throttle \/ searching = 0) THEN Bump ELSE NoBump
    /\ UNCHANGED <<submitted, completed, injector, deque, executing, spawns, deciding, pc,
                   searching, counted, idle, sampled, phase, latched>>

ExtWait ==
    /\ ext = "seed"
    /\ ext' = "wait"
    /\ UNCHANGED <<submitted, completed, injector, deque, executing, spawns, deciding, pc,
                   searching, counted, idle, seq, sleeping, sampled, phase, latched>>

CompleteJob(w) ==
    /\ executing[w] = 1
    /\ (SubmitBeforeComplete => spawns[w] = 0 /\ ~deciding[w])
    /\ completed' = completed + 1
    /\ executing' = [executing EXCEPT ![w] = 0]
    /\ UNCHANGED <<submitted, injector, deque, spawns, deciding, ext, pc, searching, counted, idle, seq, sleeping,
                   sampled, phase, latched>>

(* ---- Worker: the idle protocol ---- *)

(* The own deque is empty: become a searcher, or decline to (the limit on concurrent searchers,
   modelled as a free choice so every limit policy is covered) and go straight to the last look. *)
StartSearch(w) ==
    /\ pc[w] = "own"
    /\ executing[w] = 0
    /\ ~deciding[w]
    /\ deque[w] = 0
    /\ \/ /\ pc' = [pc EXCEPT ![w] = "search"]
          /\ searching' = searching + 1
          /\ counted' = [counted EXCEPT ![w] = TRUE]
          /\ UNCHANGED <<idle, sampled>>
       \/ /\ pc' = [pc EXCEPT ![w] = "look"]
          /\ idle' = idle + 1
          /\ sampled' = [sampled EXCEPT ![w] = seq]
          /\ UNCHANGED <<searching, counted>>
    /\ UNCHANGED <<submitted, completed, injector, deque, executing, spawns, deciding, ext, seq, sleeping,
                   phase, latched>>

(* A searcher takes a job from the injector or from another worker's deque. It leaves the count;
   the last searcher out may wake one more worker if any is idle. The engine wakes when the queue
   it took from still holds two jobs; the model leaves the choice free, so it covers that rule
   and every other. *)
FoundWork(w) ==
    /\ pc[w] = "search"
    /\ \/ /\ injector > 0
          /\ injector' = injector - 1
          /\ UNCHANGED deque
       \/ \E v \in Workers \ {w} :
              /\ deque[v] > 0
              /\ deque' = [deque EXCEPT ![v] = deque[v] - 1]
              /\ UNCHANGED injector
    /\ searching' = searching - 1
    /\ counted' = [counted EXCEPT ![w] = FALSE]
    /\ IF searching = 1 /\ idle > 0 THEN (Bump \/ NoBump) ELSE NoBump
    /\ pc' = [pc EXCEPT ![w] = "own"]
    /\ executing' = [executing EXCEPT ![w] = 1]
    /\ \E sp \in NewSpawn : spawns' = [spawns EXCEPT ![w] = sp]
    /\ UNCHANGED <<submitted, completed, deciding, ext, idle, sampled, phase, latched>>

(* The bounded random search may miss work that is there: giving up is always enabled. *)
GiveUp(w) ==
    /\ pc[w] = "search"
    /\ pc' = [pc EXCEPT ![w] = "leave"]
    /\ UNCHANGED <<submitted, completed, injector, deque, executing, spawns, deciding, ext, searching,
                   counted, idle, seq, sleeping, sampled, phase, latched>>

(* Leave the count (LeaveBeforeLook), announce idle, sample the park word, in that order. *)
Leave(w) ==
    /\ pc[w] = "leave"
    /\ idle' = idle + 1
    /\ sampled' = [sampled EXCEPT ![w] = seq]
    /\ pc' = [pc EXCEPT ![w] = "look"]
    /\ searching' = IF LeaveBeforeLook THEN searching - 1 ELSE searching
    /\ counted' = IF LeaveBeforeLook THEN [counted EXCEPT ![w] = FALSE] ELSE counted
    /\ UNCHANGED <<submitted, completed, injector, deque, executing, spawns, deciding, ext, seq, sleeping,
                   phase, latched>>

TakeAnywhere(w) ==
    /\ \/ /\ injector > 0
          /\ injector' = injector - 1
          /\ UNCHANGED deque
       \/ \E v \in Workers :
              /\ deque[v] > 0
              /\ deque' = [deque EXCEPT ![v] = deque[v] - 1]
              /\ UNCHANGED injector
    /\ executing' = [executing EXCEPT ![w] = 1]
    /\ \E sp \in NewSpawn : spawns' = [spawns EXCEPT ![w] = sp]
    /\ idle' = idle - 1
    /\ pc' = [pc EXCEPT ![w] = "own"]
    /\ searching' = IF counted[w] THEN searching - 1 ELSE searching
    /\ counted' = [counted EXCEPT ![w] = FALSE]
    /\ UNCHANGED sleeping

(* The exhaustive look: every deque and the injector. A worker still counted (LeaveBeforeLook =
   FALSE) leaves the count after it. *)
Look(w) ==
    /\ pc[w] = "look"
    /\ \/ TakeAnywhere(w)
       \/ /\ QueuedJobs = 0
          /\ pc' = [pc EXCEPT ![w] = IF counted[w] THEN "left" ELSE "parked"]
          /\ sleeping' = IF ~counted[w] /\ seq = sampled[w] THEN sleeping \cup {w} ELSE sleeping
          /\ UNCHANGED <<injector, deque, executing, spawns, idle, searching, counted>>
    /\ UNCHANGED <<submitted, completed, deciding, ext, seq, sampled, phase, latched>>

LeaveLate(w) ==
    /\ pc[w] = "left"
    /\ searching' = searching - 1
    /\ counted' = [counted EXCEPT ![w] = FALSE]
    /\ pc' = [pc EXCEPT ![w] = "parked"]
    /\ sleeping' = IF seq = sampled[w] THEN sleeping \cup {w} ELSE sleeping
    /\ UNCHANGED <<submitted, completed, injector, deque, executing, spawns, deciding, ext, idle, seq,
                   sampled, phase, latched>>

(* A parked worker that is not blocked returns from its park. *)
Wake(w) ==
    /\ pc[w] = "parked"
    /\ w \notin sleeping
    /\ idle' = idle - 1
    /\ pc' = [pc EXCEPT ![w] = "own"]
    /\ UNCHANGED <<submitted, completed, injector, deque, executing, spawns, deciding, ext, searching,
                   counted, seq, sleeping, sampled, phase, latched>>

(* ---- The termination checker ---- *)

StartCheck ==
    /\ ext = "wait"
    /\ phase = "idle"
    /\ latched' = IF CountsFirst THEN CountsAgree ELSE QueuesEmpty
    /\ phase' = "half"
    /\ UNCHANGED <<submitted, completed, injector, deque, executing, spawns, deciding, ext, pc,
                   searching, counted, idle, seq, sleeping, sampled>>

FinishCheck ==
    /\ phase = "half"
    /\ latched
    /\ (IF CountsFirst THEN (QueuesEmpty /\ CountsAgree) ELSE CountsAgree)
    /\ phase' = "done"
    /\ UNCHANGED <<submitted, completed, injector, deque, executing, spawns, deciding, ext, pc,
                   searching, counted, idle, seq, sleeping, sampled, latched>>

AbandonCheck ==
    /\ phase = "half"
    /\ ~(latched /\ (IF CountsFirst THEN (QueuesEmpty /\ CountsAgree) ELSE CountsAgree))
    /\ phase' = "idle"
    /\ latched' = FALSE
    /\ UNCHANGED <<submitted, completed, injector, deque, executing, spawns, deciding, ext, pc,
                   searching, counted, idle, seq, sleeping, sampled>>

Next == \/ \E w \in Workers :
              \/ TakeOwn(w) \/ SubmitChild(w) \/ Decide(w) \/ CompleteJob(w)
              \/ StartSearch(w) \/ FoundWork(w) \/ GiveUp(w) \/ Leave(w) \/ Look(w)
              \/ LeaveLate(w) \/ Wake(w)
        \/ ExtSubmit \/ ExtDecide \/ ExtWait
        \/ StartCheck \/ FinishCheck \/ AbandonCheck

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

WorkRemains == \/ injector > 0
               \/ \E w \in Workers : deque[w] > 0 \/ executing[w] = 1 \/ spawns[w] = 1
               \/ submitted # completed

(* A checker that has finished reading must not report done while work remains. *)
NoPrematureQuiescence == (phase = "done") => ~WorkRemains

(* No job is queued while every worker is blocked in its park. *)
NoLostWakeup == (ext # "decide" /\ \A w \in Workers : w \in sleeping) => QueuedJobs = 0

(* The counts equal the workers in the corresponding states. *)
CountsExact == /\ searching = Cardinality({w \in Workers : counted[w]})
               /\ idle = Cardinality({w \in Workers : pc[w] \in {"look", "left", "parked"}})
               /\ \A w \in sleeping : pc[w] = "parked"
=============================================================================
