(* Gate: a malformed option value must not desynchronize the WXF option parser.
   CanonicalizeEvents -> {"InputState", 42} fails mid-list (the header parses, the
   integer element does not); recovery re-aligns to the value start and skips it,
   so options AFTER it still apply. Discriminator: CanonicalizeStates -> Full is
   parsed AFTER CanonicalizeEvents and changes NumStates on wolfram-2to4
   (canonical merge vs raw tree); a desynced parser loses it. *)

copyDir = FileNameJoin[{$TemporaryDirectory,
  "hgpaclet_" <> IntegerString[Hash[CreateUUID[]], 16, 12]}];
CopyDirectory[FileNameJoin[{DirectoryName[$InputFileName], "..", "paclet"}], copyDir];
PacletDirectoryLoad[copyDir];
Needs["HypergraphRewriting`"];

rule = {{x, y}, {y, z}} -> {{x, y}, {y, w}, {w, z}, {z, x}};
init = {{1, 2}, {2, 3}};

nFullClean = Quiet[HGEvolve[rule, init, 3, "NumStates", "CanonicalizeStates" -> Full], {LibraryFunction::load, URLSubmit::offline}];
nNone = Quiet[HGEvolve[rule, init, 3, "NumStates"], {LibraryFunction::load, URLSubmit::offline}];
nFullAfterBad = Quiet[HGEvolve[rule, init, 3, "NumStates",
  "CanonicalizeEvents" -> {"InputState", 42}, "CanonicalizeStates" -> Full],
  {LibraryFunction::load, URLSubmit::offline}];
warned = MemberQ[$MessageList, HoldForm[MessageName[HGEvolve, "warn"]]];

Print["full=", nFullClean, " none=", nNone, " fullAfterBad=", nFullAfterBad,
  " skip-warned=", warned];

ok = IntegerQ[nFullClean] && IntegerQ[nFullAfterBad] &&
  nFullClean =!= nNone &&            (* the discriminator discriminates *)
  nFullAfterBad === nFullClean &&    (* the option AFTER the bad one applied *)
  warned;                            (* and the skip is on the trail *)
Print[If[ok, "DESYNC GATE PASS", "DESYNC GATE FAIL"]];
