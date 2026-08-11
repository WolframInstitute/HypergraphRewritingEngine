#!/usr/bin/env wolframscript
(* The session verbs, through the shipped engine binary, at the Wolfram layer.

   THE CLAIM UNDER TEST is not "a session answers": it is that a session CONTINUES. A session
   stepped k times must see exactly what one HGEvolve of depth k sees, which a restarting
   implementation fails on the second step -- it would report depth-1 numbers forever -- and an
   accumulate-but-never-consume implementation fails by over-counting. Comparing every prefix
   against an independent one-shot run is what separates the three.

   WHY THIS IS A SEPARATE SCRIPT rather than a case in verify_paclet.wls: the golden corpus
   checks the ENGINE's answers, and this checks the SESSION's lifetime -- open, refusal of a
   second open, close, reopen. A lifetime cannot be asserted from a single call, so it needs a
   script that makes several in order and reads them together.

   The session verbs require the persistent socket worker. HGEvolve's one-shot fallback would
   mint a handle inside a process that exits with the reply, so a session verb takes the worker
   or refuses -- and this script reports that refusal as a FAIL, because a machine that cannot
   run a worker cannot verify sessions. *)

repo = ParentDirectory[DirectoryName[$InputFileName]];
PacletDirectoryLoad[FileNameJoin[{repo, "paclet"}]];
Needs["HypergraphRewriting`"];

failures = 0;
check[label_, cond_] := (
  Print[If[TrueQ[cond], "PASS", "FAIL"], "  ", label];
  If[!TrueQ[cond], failures++]);

rules = {{{1, 2}} -> {{1, 3}, {3, 2}}};
init  = {{1, 2}};
prop  = {"NumStates", "NumEvents"};
maxDepth = 4;

(* The independent answer. Each depth is its own process-isolated run, so nothing it produces
   can be an artefact of the session machinery it is used to check. *)
oneShot = Table[HGEvolve[rules, init, k, prop], {k, 0, maxDepth}];
Print["one-shot depths 0..", maxDepth, ": ", oneShot];
check["the one-shot ladder is strictly growing (the workload actually branches)",
      OrderedQ[#["NumStates"] & /@ oneShot] &&
      Length[DeleteDuplicates[#["NumStates"] & /@ oneShot]] == maxDepth + 1];

s = HGSessionOpen[rules, init, prop];
check["HGSessionOpen returns a session", Head[s] === HGSessionObject];
If[Head[s] =!= HGSessionObject,
  Print["\nno session: the remaining checks cannot run"];
  Print["\nfailures: ", failures + 1];
  Exit[1]];

(* Depth 0 first: an Open must NOT have explored. A session that evolved at open would pass
   every later step comparison shifted by one, and only this check sees it. *)
check["a fresh session is at depth 0", HGSessionQuery[s] === oneShot[[1]]];

Do[
  Module[{r = HGSessionStep[s, 1]},
    check["step " <> ToString[k] <> " == one evolve of " <> ToString[k] <> "  " <>
          ToString[r], r === oneShot[[k + 1]]]],
  {k, 1, maxDepth}];

check["Query does not advance the session", HGSessionQuery[s] === oneShot[[-1]]];
check["Step 0 does not advance the session", HGSessionStep[s, 0] === oneShot[[-1]]];

(* D7: one session at a time, and a second Open is REFUSED rather than evicting the first --
   evicting would discard a caller's exploration without being asked. The first session must
   still answer afterwards, which is what makes this a refusal rather than a broken worker. *)
check["a second Open is refused", Quiet[HGSessionOpen[rules, init, prop]] === $Failed];
check["the first session still answers after the refusal", HGSessionQuery[s] === oneShot[[-1]]];

check["Close succeeds", HGSessionClose[s] === Null];
check["a verb on a closed session fails rather than answering",
      Quiet[HGSessionStep[s, 1]] === $Failed];

(* And the slot is genuinely free: a Close that only detached the handle would fail here. *)
s2 = HGSessionOpen[rules, init, prop];
check["a session can be opened after the first closed", Head[s2] === HGSessionObject];
If[Head[s2] === HGSessionObject,
  check["one Step of " <> ToString[maxDepth] <> " == " <> ToString[maxDepth] <> " Steps of 1",
        HGSessionStep[s2, maxDepth] === oneShot[[-1]]];
  HGSessionClose[s2]];

(* SUBSET CONTINUATION. Steering must narrow what runs NEXT without narrowing what remains
   REACHABLE, and the only check that separates those two is: steer, then resume, and land on
   the exhaustive answer. A run that dropped the unselected frontier entries would pass a
   "fewer states" check and fail this one. *)
s4 = HGSessionOpen[rules, init, prop];
If[Head[s4] === HGSessionObject,
  Module[{f, steered, resumed},
    HGSessionStep[s4, 2];
    f = HGSessionFrontier[s4];
    check["the frontier is reported and has more than one branch to choose from",
          ListQ[f] && Length[f] > 1];
    steered = HGSessionStep[s4, 1, Automatic, "From" -> {First[f]}];
    check["a steered step explores LESS than the exhaustive one  " <> ToString[steered],
          steered["NumStates"] < oneShot[[4]]["NumStates"]];
    check["the unselected branch is RETAINED, not dropped",
          MemberQ[HGSessionFrontier[s4], Last[f]]];
    resumed = HGSessionStep[s4, 1];
    check["steer then resume lands on the exhaustive answer  " <> ToString[resumed],
          resumed === oneShot[[-1]]];
    check["naming a state that is not on the frontier is an error, not a silent no-op",
          Quiet[HGSessionStep[s4, 1, Automatic, "From" -> {999999}]] === $Failed]];
  HGSessionClose[s4]];

(* DELTA DELIVERY. The wire is incremental; the ANSWER is not. So the check is not "is the reply
   smaller" -- it is that stepping with Delta and stepping with Full produce the SAME graph at
   every depth. A merge that dropped a delta silently converges to whatever the last reply
   carried, which is a smaller graph that is internally consistent and wrong; that is the failure
   this actually caught during development, when the delta marker was read as a symbol while the
   wire carries it as an integer. *)
Module[{gFull, gDelta, sA, sB},
  sA = HGSessionOpen[rules, init, "StatesGraphStructure"];
  gFull = Table[Module[{g = HGSessionStep[sA, 1]},
                  {VertexCount[g], EdgeCount[g]}], {maxDepth}];
  HGSessionClose[sA];
  sB = HGSessionOpen[rules, init, "StatesGraphStructure"];
  gDelta = Table[Module[{g = HGSessionStep[sB, 1, Automatic, "Delivery" -> "Delta"]},
                   {VertexCount[g], EdgeCount[g]}], {maxDepth}];
  check["Delta delivery gives the SAME graph as Full at every depth  " <> ToString[gDelta],
        gDelta === gFull];
  (* And a full delivery mid-session resynchronises rather than merging into a stale record. *)
  check["a Full delivery after deltas still gives the whole graph",
        Module[{g = HGSessionQuery[sB]}, {VertexCount[g], EdgeCount[g]}] === Last[gFull]];
  HGSessionClose[sB]];

(* A property the session was not opened to record is reported, not served as emptiness. *)
s3 = HGSessionOpen[rules, init, "NumStates"];
If[Head[s3] === HGSessionObject,
  check["an unrecorded artifact warns rather than returning silently",
        Quiet[Check[HGSessionStep[s3, 2, "CausalEdges"]; False, True],
              {HGEvolve::warn, HGEvolve::missingdata}]];
  HGSessionClose[s3]];

Print["\nfailures: ", failures];
Print[If[failures == 0, "SESSIONS_VERIFIED", "SESSIONS_FAILED"]];
Exit[If[failures == 0, 0, 1]]
