(* Gate: IncludeCanonicalHashes serves the IR hash in EVERY state-canonicalization
   mode. The reachable iso-class set of an evolution does not depend on the mode, so
   the SET of served hashes under None must equal the set under Full. WL values in
   the None column (the A10 defect) are numerically disjoint from IR values, so this
   discriminates exactly. *)

copyDir = FileNameJoin[{$TemporaryDirectory,
  "hgpaclet_" <> IntegerString[Hash[CreateUUID[]], 16, 12]}];
CopyDirectory[FileNameJoin[{DirectoryName[$InputFileName], "..", "paclet"}], copyDir];
PacletDirectoryLoad[copyDir];
Needs["HypergraphRewriting`"];

rule = {{x, y}, {y, z}} -> {{x, y}, {y, w}, {w, z}, {z, x}};
init = {{1, 2}, {2, 3}};

hashSet[mode_] := Module[{states},
  states = Quiet[HGEvolve[rule, init, 3, "States",
    "CanonicalizeStates" -> mode, "IncludeCanonicalHashes" -> True],
    {LibraryFunction::load, URLSubmit::offline}];
  If[!AssociationQ[states], Return[$Failed]];
  DeleteDuplicates[Sort[Lookup[Values[states], "CanonicalHash", Nothing]]]];

full = hashSet[Full];
none = hashSet[None];
auto = hashSet[Automatic];

Print["fullN=", Length[full], " noneN=", Length[none], " autoN=", Length[auto]];
Print["none==full=", none === full, " auto==full=", auto === full];

ok = ListQ[full] && Length[full] > 1 && none === full && auto === full;
Print[If[ok, "CANONICAL HASH MODES GATE PASS", "CANONICAL HASH MODES GATE FAIL"]];
