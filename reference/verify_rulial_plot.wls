(* Gate: the rulial-space plot (#17). ColorByRule -> True colors each StatesGraph
   transition by the rule that fired it and adds a swatch legend over the rules
   that actually fired; Structure variants (topology-only payloads) are
   unaffected. Two-rule system where both rules stay applicable. *)

copyDir = FileNameJoin[{$TemporaryDirectory,
  "hgpaclet_" <> IntegerString[Hash[CreateUUID[]], 16, 12]}];
CopyDirectory[FileNameJoin[{DirectoryName[$InputFileName], "..", "paclet"}], copyDir];
PacletDirectoryLoad[copyDir];
Needs["HypergraphRewriting`"];

rules = {
  {{x, y}} -> {{x, z}, {z, y}},
  {{x, y}} -> {{y, x}}
};
init = {{1, 2}};

g = Quiet[HGEvolve[rules, init, 3, "StatesGraph", "ColorByRule" -> True],
  {LibraryFunction::load, URLSubmit::offline}];
isLegended = Head[g] === Legended;
inner = If[isLegended, First[g], g];
isGraph = Head[inner] === Graph;

(* Both rules fired: the edge payloads carry 2 distinct RuleIndex values. *)
ruleIdxs = Sort @ DeleteDuplicates @ Cases[EdgeList[inner],
  DirectedEdge[_, _, tag_Association /; KeyExistsQ[tag, "RuleIndex"]] :>
    tag["RuleIndex"]];
legendHasRules = isLegended &&
  !FreeQ[g, SwatchLegend[_, labels_ /; Length[labels] == Length[ruleIdxs]]];

(* Edge styles: at least one per fired rule beyond the base type styles. *)
styleCount = Length[Cases[Options[inner, EdgeStyle], _Directive, Infinity]];

Print["legended=", isLegended, " graph=", isGraph, " ruleIdxs=", ruleIdxs,
  " legendMatches=", legendHasRules, " directiveStyles=", styleCount];

(* Structure variant: lean payloads, no RuleIndex, no legend. *)
gs = Quiet[HGEvolve[rules, init, 3, "StatesGraphStructure", "ColorByRule" -> True],
  {LibraryFunction::load, URLSubmit::offline}];
structPlain = Head[gs] === Graph;
Print["structurePlain=", structPlain];

ok = isLegended && isGraph && ruleIdxs === {0, 1} && legendHasRules &&
  styleCount > 0 && structPlain;
Print[If[ok, "RULIAL PLOT GATE PASS", "RULIAL PLOT GATE FAIL"]];
