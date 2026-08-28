------------------------------ MODULE Quiescence ------------------------------
(* Termination detection: is_quiescent() must not say "done" while work remains.

   WHY THIS EXISTS. The job system's WAKE protocol is covered by two GenMC harnesses -- a parked
   worker is always woken, including across cache domains. Nothing covered the other half: the
   predicate that decides a run is FINISHED. A false positive there returns a smaller multiway
   system with no indication it is smaller, which is the failure the container ceiling used to
   produce and is harder to see, because no warning is attached to it.

   WHY TLA+ AND NOT GENMC. The property is a global predicate over every worker read while those
   workers run, which is a protocol rather than one structure's memory operations, and it is the
   same reason SegmentedArray is here.

   THE PROTOCOL, from job_system.hpp. A submit bumps `submitted` and puts the job somewhere. A
   worker takes a job, marks itself executing, may submit children WHILE EXECUTING, then bumps
   `completed` and stops executing. is_quiescent() reads, in this order: the two counters, the
   injector, then every worker's deque and executing flag, and finally re-reads the counters
   behind a fence.

   THE ORDER IS THE ARGUMENT. A worker that will submit a child has not completed yet, so the
   counters differ and the scan is never reached; a worker that has submitted its child and then
   completed leaves the child visible in a deque or the injector before the counters can agree.
   CountsFirst models that order. FALSE checks the queues BEFORE the counters, which admits the
   interleaving where the scan sees empty queues, the worker then finishes its job and submits
   nothing more, and the counters agree afterwards -- the predicate reporting done having looked
   at each half at the moment the other half held the work. *)
EXTENDS Integers, FiniteSets, TLC

CONSTANTS Workers, MaxJobs, CountsFirst, CheckExecuting, SubmitBeforeComplete

VARIABLES submitted,   (* jobs ever submitted *)
          completed,   (* jobs ever completed *)
          injector,    (* jobs waiting, unclaimed *)
          deque,       (* deque[w] = jobs queued on worker w *)
          executing,   (* executing[w] = 1 while w runs a job *)
          spawns,      (* spawns[w] = children w will still submit before completing *)
          phase,       (* the checker's progress: "idle", "half", "done" *)
          latched      (* what its FIRST read observed *)

vars == <<submitted, completed, injector, deque, executing, spawns, phase, latched>>

TypeOK == /\ submitted \in 0..MaxJobs
          /\ completed \in 0..MaxJobs
          /\ injector \in 0..MaxJobs
          /\ deque \in [Workers -> 0..MaxJobs]
          /\ executing \in [Workers -> 0..1]
          /\ spawns \in [Workers -> 0..1]
          /\ phase \in {"idle", "half", "done"}
          /\ latched \in BOOLEAN

Init == /\ submitted = 1
        /\ completed = 0
        /\ injector = 1
        /\ deque = [w \in Workers |-> 0]
        /\ executing = [w \in Workers |-> 0]
        /\ spawns = [w \in Workers |-> 0]
        /\ phase = "idle"
        /\ latched = FALSE

(* CheckExecuting is the jobs_executing half of the scan. Dropping it is a real defect and not a
   reordering: a worker INSIDE a job has not completed, so the counters do not agree yet -- but a
   scan that reaches the queues while that worker's child is not yet pushed sees nothing, and the
   counters can agree by the time the second read happens. *)
QueuesEmpty == /\ injector = 0
               /\ \A w \in Workers : deque[w] = 0 /\ (CheckExecuting => executing[w] = 0)

CountsAgree == submitted = completed

(* A worker claims a job from the injector and begins executing it. A job may owe one child. *)
TakeFromInjector(w) ==
    /\ executing[w] = 0
    /\ injector > 0
    /\ injector' = injector - 1
    /\ executing' = [executing EXCEPT ![w] = 1]
    /\ spawns' = [spawns EXCEPT ![w] = IF submitted < MaxJobs THEN 1 ELSE 0]
    /\ UNCHANGED <<submitted, completed, deque, phase, latched>>

TakeFromOwnDeque(w) ==
    /\ executing[w] = 0
    /\ deque[w] > 0
    /\ deque' = [deque EXCEPT ![w] = deque[w] - 1]
    /\ executing' = [executing EXCEPT ![w] = 1]
    /\ spawns' = [spawns EXCEPT ![w] = 0]
    /\ UNCHANGED <<submitted, completed, injector, phase, latched>>

(* THE CHILD IS SUBMITTED WHILE THE PARENT IS STILL EXECUTING. That is what the engine does and
   it is what makes the counter check sound: the parent has not completed, so the counters cannot
   agree while a child is still owed. *)
SubmitChild(w) ==
    /\ (SubmitBeforeComplete => executing[w] = 1)
    /\ spawns[w] = 1
    /\ submitted < MaxJobs
    /\ submitted' = submitted + 1
    /\ deque' = [deque EXCEPT ![w] = deque[w] + 1]
    /\ spawns' = [spawns EXCEPT ![w] = 0]
    /\ UNCHANGED <<completed, injector, executing, phase, latched>>

(* SubmitBeforeComplete is the engine's order: a job submits its children and only then returns,
   so the parent is still uncompleted while the child is owed and the counters cannot agree. FALSE
   models completing FIRST and submitting after, which opens a window where the counters agree and
   every queue is empty while a child is still owed -- and that is a premature quiescence no
   ordering of the reads can defend against, because at that instant there is nothing to see. *)
CompleteJob(w) ==
    /\ executing[w] = 1
    /\ (SubmitBeforeComplete => spawns[w] = 0)
    /\ completed' = completed + 1
    /\ executing' = [executing EXCEPT ![w] = 0]
    /\ UNCHANGED <<submitted, injector, deque, spawns, phase, latched>>

(* THE CHECKER READS THE TWO HALVES IN SEPARATE STEPS, which is the whole point: workers run
   between them. CountsFirst reads the counters first and re-reads them at the end, so a
   disagreement that appeared in between is caught. The broken order reads the QUEUES first and
   trusts that read, so work that was in a queue at the first read and has moved by the second is
   never seen by either. *)
StartCheck ==
    /\ phase = "idle"
    /\ latched' = IF CountsFirst THEN CountsAgree ELSE QueuesEmpty
    /\ phase' = "half"
    /\ UNCHANGED <<submitted, completed, injector, deque, executing, spawns>>

FinishCheck ==
    /\ phase = "half"
    /\ latched
    /\ (IF CountsFirst THEN (QueuesEmpty /\ CountsAgree) ELSE CountsAgree)
    /\ phase' = "done"
    /\ UNCHANGED <<submitted, completed, injector, deque, executing, spawns, latched>>

AbandonCheck ==
    /\ phase = "half"
    /\ ~(latched /\ (IF CountsFirst THEN (QueuesEmpty /\ CountsAgree) ELSE CountsAgree))
    /\ phase' = "idle"
    /\ latched' = FALSE
    /\ UNCHANGED <<submitted, completed, injector, deque, executing, spawns>>

Next == \/ \E w \in Workers :
              \/ TakeFromInjector(w) \/ TakeFromOwnDeque(w)
              \/ SubmitChild(w) \/ CompleteJob(w)
        \/ StartCheck \/ FinishCheck \/ AbandonCheck

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

WorkRemains == \/ injector > 0
               \/ \E w \in Workers : deque[w] > 0 \/ executing[w] = 1 \/ spawns[w] = 1
               \/ submitted # completed

(* THE CONTRACT: a checker that has finished reading must not report done while work remains. *)
NoPrematureQuiescence == (phase = "done") => ~WorkRemains
=============================================================================
