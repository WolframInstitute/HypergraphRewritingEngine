(* Gate: GeodesicSources are the user's initial-state labels, translated through
   the engine's renumbering. Initial state uses labels {10, 20, 30} -- engine ids
   are 0..2, so a pre-translation FFI pointed source 10 at a nonexistent vertex.
   An unknown label warns and is skipped. *)

copyDir = FileNameJoin[{$TemporaryDirectory,
  "hgpaclet_" <> IntegerString[Hash[CreateUUID[]], 16, 12]}];
CopyDirectory[FileNameJoin[{DirectoryName[$InputFileName], "..", "paclet"}], copyDir];
PacletDirectoryLoad[copyDir];
Needs["HypergraphRewriting`"];

rule = {{x, y}} -> {{x, z}, {z, y}};
init = {{10, 20}, {20, 30}};

mark = 0;
freshWarn[] := Module[{c = Count[$MessageList, HoldForm[MessageName[HGEvolve, "warn"]]], r},
  r = c > mark; mark = c; r];

(* Case 1: a KNOWN user label as source -> geodesic data served, no warning. *)
g1 = Quiet[HGEvolve[rule, init, 4, "GeodesicData",
    "GeodesicAnalysis" -> True, "GeodesicSources" -> {10}],
  {LibraryFunction::load, URLSubmit::offline}];
w1 = freshWarn[];
served1 = AssociationQ[g1] && Length[g1] > 0;
Print["case1 served=", served1, " warned=", w1];

(* Case 2: an UNKNOWN label -> warns, still serves via auto-select. *)
g2 = Quiet[HGEvolve[rule, init, 4, "GeodesicData",
    "GeodesicAnalysis" -> True, "GeodesicSources" -> {999}],
  {LibraryFunction::load, URLSubmit::offline}];
w2 = freshWarn[];
served2 = AssociationQ[g2] && Length[g2] > 0;
Print["case2 served=", served2, " warned=", w2];

ok = served1 && !w1 && served2 && w2;
Print[If[ok, "GEODESIC SOURCES GATE PASS", "GEODESIC SOURCES GATE FAIL"]];
