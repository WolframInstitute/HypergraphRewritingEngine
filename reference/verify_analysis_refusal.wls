(* Gate: blackhole analyses REFUSE non-binary input with a served warning, and the
   branchial family carries the branch_id-degeneracy advisory on binary input.
   Run with the Windows wolframscript against a fresh copy of the repo paclet. *)

copyDir = FileNameJoin[{$TemporaryDirectory,
  "hgpaclet_" <> IntegerString[Hash[CreateUUID[]], 16, 12]}];
CopyDirectory[FileNameJoin[{DirectoryName[$InputFileName], "..", "paclet"}], copyDir];
PacletDirectoryLoad[copyDir];
Needs["HypergraphRewriting`"];

warned[] := MemberQ[$MessageList, HoldForm[MessageName[HGEvolve, "warn"]]];
mark = 0;
freshWarn[] := Module[{c = Count[$MessageList, HoldForm[MessageName[HGEvolve, "warn"]]], r},
  r = c > mark; mark = c; r];

(* Case 1: arity-3 edges + DimensionData -> refusal: warn fires, property unserved. *)
r1 = Quiet[HGEvolve[{{x, y}, {y, z}} -> {{x, y}, {y, w}, {w, z}},
    {{1, 2, 3}, {1, 2}, {2, 3}}, 2, "DimensionData", "DimensionAnalysis" -> True],
  {LibraryFunction::load, URLSubmit::offline, HGEvolve::missingdata}];
w1 = freshWarn[];
Print["case1 refusal-warning=", w1, " unserved=", r1 === $Failed || Length[r1] == 0];

(* Case 2: binary input + BranchialData -> advisory fires, data served. *)
r2 = Quiet[HGEvolve[{{x, y}} -> {{x, z}, {z, y}}, {{1, 2}}, 3, "BranchialData", "BranchialAnalysis" -> True],
  {LibraryFunction::load, URLSubmit::offline}];
w2 = freshWarn[];
Print["case2 advisory=", w2, " served=", r2 =!= $Failed];

(* Case 3: binary input, plain States -> NO warning. *)
r3 = Quiet[HGEvolve[{{x, y}} -> {{x, z}, {z, y}}, {{1, 2}}, 2, "States"],
  {LibraryFunction::load, URLSubmit::offline}];
w3 = freshWarn[];
Print["case3 quiet=", ! w3, " served=", r3 =!= $Failed];

ok = w1 && (r1 === $Failed || Length[r1] == 0) && w2 && r2 =!= $Failed && ! w3 && r3 =!= $Failed;
Print[If[ok, "REFUSAL GATE PASS", "REFUSAL GATE FAIL"]];
