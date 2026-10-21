#!/usr/bin/env wolframscript
(* Random differential fuzz: HGEvolve (CPU and GPU) against reference/MultiwayReference.wl.

   wolframscript -file oracle_fuzz.wls pacletDir outDir seedFrom seedTo [gpu] [auth] [minutes]

     gpu      1 (default) runs every configuration on TargetDevice "GPU" as well
     auth     1 also runs Wolfram/Multicomputation MultiwaySystem for Automatic events (default 0)
     minutes  stop starting cases after this many minutes (default 0: no limit)

   The case class is chosen by the seed: Mod[seed, Length[$classes]] indexes $classes, so any 14
   consecutive seeds cover every class. Per case the reference runs once (with genesis events,
   full capture) and every engine configuration is compared with it:

     states   NumStates and the multiset of canonical forms of "States" under CanonicalizeStates
              None (every raw state), Automatic (equal sorted edge lists) and Full (isomorphism)
     events   NumEvents under CanonicalizeEvents None, Full and key lists (exact); Automatic and
              "Positional" lie between the marked-form count and the raw count, and their equality
              with the reference's rank counts is reported as amber (SPEC 4.2, 5.6)
     causal   NumCausalEdges with and without CausalTransitiveReduction, with and without
              ShowGenesisEvents; EdgeCount of "CausalGraph" under EdgeDeduplication True and False
     branch   NumBranchialEdges
     caps     a cap or sampling option at a value that keeps everything gives the uncapped
              numbers; a small cap gives the same numbers on both devices, under quotient
              exploration, and no more states or events than
              the uncapped run
     session  a session stepped one step at a time, and a step split over two "From" subsets of
              the frontier, give the reference's numbers at every depth
     named    a named initial condition gives the numbers of its edge list
     present  relabelled vertices (including symbols and offsets near 2^32), permuted edges and
              renamed rule variables change no number

   Writes outDir/results.m (one InputForm association per case) and outDir/disc.txt (one per case
   with a hard difference). Amber and known classes are in results.m. Exit status 1 when any case
   has a hard difference. *)

args = $ScriptCommandLine;
pacletDir = args[[2]]; outDir = args[[3]];
seedFrom = ToExpression[args[[4]]]; seedTo = ToExpression[args[[5]]];
useGPU = If[Length[args] >= 6, args[[6]] =!= "0", True];
useAuth = Length[args] >= 7 && args[[7]] === "1";
minutes = If[Length[args] >= 8, ToExpression[args[[8]]], 0];
verbose = Length[args] >= 9 && args[[9]] === "verbose";
refDir = DirectoryName[$InputFileName];
tStart = AbsoluteTime[];

PacletDirectoryLoad[pacletDir];
Needs["HypergraphRewriting`"];
Get[FileNameJoin[{refDir, "MultiwayReference.wl"}]];
If[useAuth, Quiet[Needs["Wolfram`Multicomputation`"]]];

Quiet[CreateDirectory[outDir]];
resFile = FileNameJoin[{outDir, "results.m"}];
discFile = FileNameJoin[{outDir, "disc.txt"}];
logLine[file_, s_String] := Module[{st = OpenAppend[file, CharacterEncoding -> "UTF8"]},
  WriteString[st, s, "\n"]; Close[st]];

(* ---------------- case generator ---------------- *)
$classes = {"random", "arity", "unary", "selfloop", "disconnected", "deleteOnly", "identical",
  "cyclic", "nonconfluent", "symmetric", "named", "multiroot", "large", "symbolic"};

renameLhs[lhs_] := lhs /. Thread[DeleteDuplicates[Flatten[lhs]] -> Range[Length[DeleteDuplicates[Flatten[lhs]]]]];

(* A rule over the arities given: nL left edges on a pool of p variables, a right side of nRhs
   edges over the left variables and up to two fresh ones. *)
genRule[arities_, nL_, nRhs_] := Module[{p, lhs, lvars, fresh, rhs},
  p = RandomInteger[{1, Max[1, nL*Max[arities] - 1]}];
  lhs = renameLhs[Table[RandomInteger[{1, p}, RandomChoice[arities]], {nL}]];
  lvars = Union[Flatten[lhs]];
  fresh = Length[lvars] + Range[RandomInteger[{0, 2}]];
  rhs = Table[Table[If[fresh =!= {} && RandomReal[] < 0.3, RandomChoice[fresh], RandomChoice[lvars]],
      {RandomChoice[arities]}], {nRhs}];
  {lhs, rhs}];

(* One rule's left side instantiated (not necessarily injectively), plus extra random edges. *)
instantiate[rules_, nv_, extra_, arities_] := Module[{lhs = RandomChoice[rules][[1]], vs, sub},
  vs = Union[Flatten[lhs]];
  sub = Thread[vs -> RandomInteger[{1, Max[nv, 1]}, Length[vs]]];
  RandomSample[Join[lhs /. sub, Table[RandomInteger[{1, Max[nv, 1]}, RandomChoice[arities]], {extra}]]]];

symmetricState[] := Module[{n = RandomInteger[{3, 5}]},
  Switch[RandomInteger[{1, 7}],
    1, Flatten[Table[If[i != j, {i, j}, Nothing], {i, Min[n, 4]}, {j, Min[n, 4]}], 1],   (* complete digraph *)
    2, Table[{i, Mod[i, n + 1] + 1}, {i, n + 1}],                                        (* directed cycle *)
    3, Table[{1, i}, {i, 2, n + 1}],                                                     (* star *)
    4, Join[Table[{i, Mod[i, n] + 1}, {i, n}], Table[{Mod[i, n] + 1, i}, {i, n}]],       (* undirected cycle *)
    5, Flatten[Table[{If[x < 3, {3 (y - 1) + x, 3 (y - 1) + x + 1}, Nothing],
        If[y < 2, {3 (y - 1) + x, 3 y + x}, Nothing]}, {y, 2}, {x, 3}], 2],               (* 3x2 grid *)
    6, Flatten[Table[{{3 (y - 1) + x, 3 (y - 1) + Mod[x, 3] + 1}, {3 (y - 1) + x, 3 Mod[y, 3] + x}},
        {y, 3}, {x, 3}], 2],                                                              (* 3x3 torus *)
    7, Table[{i, i}, {i, n}]]];                                                           (* disjoint self-loops *)

namedInits = {
  {"Grid", {"GridWidth" -> 3, "GridHeight" -> 2}},
  {"Grid", {"GridWidth" -> 3, "GridHeight" -> 3, "GridHoles" -> {{2, 2, 0.5}}}},
  {"Cylinder", {"GridWidth" -> 3, "GridHeight" -> 2}},
  {"Torus", {"GridWidth" -> 3}},
  {"Sphere", {"GridWidth" -> 3}},
  {"Klein", {"GridWidth" -> 3, "GridHeight" -> 2}},
  {"Mobius", {"GridWidth" -> 3}},
  {"Minkowski", {"SprinklingDensity" -> 8, "SprinklingSpatialDim" -> 1}},
  {"BrillLindquist", {"SprinklingDensity" -> 8}},
  {"Poisson", {"SprinklingDensity" -> 8}},
  {"Uniform", {"SprinklingDensity" -> 8}}};

genCase[seed_] := Module[{cls, arities, rules, init, nv, roots, steps, named = None, extra},
  SeedRandom[seed];
  cls = $classes[[Mod[seed, Length[$classes]] + 1]];
  arities = RandomChoice[{{2}, {2}, {1, 2}, {2, 3}, {3}, {1}, {1, 3}, {1, 2, 3}}];
  nv = RandomInteger[{1, 4}];
  extra = RandomChoice[{0, 0, 1, 1, 2}];
  steps = RandomChoice[{2, 3, 3, 4, 4}];
  rules = Table[genRule[arities, RandomChoice[{1, 1, 2, 2, 2, 3}], RandomChoice[{0, 1, 1, 2, 2, 2, 3, 3, 4}]],
    {RandomChoice[{1, 1, 1, 2, 2, 3}]}];
  Switch[cls,
    "arity",
      arities = RandomChoice[{{RandomInteger[{4, 16}]}, {2, RandomInteger[{4, 16}]},
        {1, RandomInteger[{5, 16}]}, RandomSample[Range[16], 3]}];
      rules = Table[genRule[arities, RandomChoice[{1, 1, 2}], RandomChoice[{0, 1, 2, 2}]], {RandomChoice[{1, 1, 2}]}];
      steps = RandomChoice[{2, 3, 3}],
    "unary",
      arities = RandomChoice[{{1}, {1, 2}, {1, 1, 2}}];
      rules = Table[genRule[arities, RandomChoice[{1, 2, 2, 3}], RandomChoice[{0, 1, 2, 3}]], {RandomChoice[{1, 2}]}],
    "selfloop",
      (* one left edge repeats a vertex; a right-side variable that leaves the left side is fresh *)
      rules = Table[Module[{r = genRule[Select[arities, # >= 2 &] /. {} -> {2}, RandomChoice[{1, 2}], RandomChoice[{1, 2, 3}]], k},
          k = RandomInteger[{1, Length[r[[1]]]}];
          r[[1, k]] = ConstantArray[First[r[[1, k]]], Length[r[[1, k]]]];
          r],
        {RandomChoice[{1, 2}]}],
    "disconnected",
      rules = Table[Module[{a = RandomChoice[arities], b = RandomChoice[arities], lhs, rhs},
          lhs = {Range[a], a + Range[b]};
          rhs = Table[RandomChoice[Range[a + b + 1], RandomChoice[arities]], {RandomChoice[{0, 1, 2, 3}]}];
          {lhs, rhs}], {RandomChoice[{1, 1, 2}]}],
    "deleteOnly",
      rules = Join[{{First[genRule[arities, RandomChoice[{1, 2}], 0]], {}}},
        If[RandomReal[] < 0.5, {genRule[arities, 1, RandomChoice[{1, 2}]]}, {}]],
    "identical",
      rules = Join[rules, If[RandomReal[] < 0.5, {First[rules]},
        (* the first rule with its variables renamed *)
        {First[rules] /. Thread[Union[Flatten[First[rules]]] -> (Union[Flatten[First[rules]]] + 100)]}]],
    "cyclic",
      rules = RandomChoice[{
        {{{{1, 2}}, {{2, 1}}}},
        {{{{1, 2, 3}}, {{2, 3, 1}}}},
        {{{{1, 2}, {2, 3}}, {{2, 3}, {3, 1}}}},
        {{{{1, 2}}, {{1, 3}}}, {{{1, 2}}, {{2, 1}}}},
        {{{{1, 2}, {2, 1}}, {{1, 1}, {2, 2}}}, {{{1, 1}, {2, 2}}, {{1, 2}, {2, 1}}}},
        {{{{1}, {2}}, {{2}, {1}}}}}];
      arities = Union[Length /@ Flatten[rules[[All, 1]], 1]];
      steps = RandomChoice[{3, 4, 5}],
    "nonconfluent",
      rules = Module[{r = genRule[arities, RandomChoice[{1, 2}], RandomChoice[{1, 2}]]},
        {r, {r[[1]], Reverse /@ Reverse[r[[2]]]}, If[RandomReal[] < 0.5, {r[[1]], Take[r[[2]], UpTo[1]]}, Nothing]}],
    "large",
      rules = Table[genRule[arities, 1, RandomChoice[{1, 2}]], {RandomChoice[{1, 2}]}];
      steps = RandomChoice[{1, 2, 2}]
  ];
  init = instantiate[rules, nv, extra, arities];
  If[RandomReal[] < 0.12, init = Table[RandomInteger[{1, nv}, RandomChoice[arities]], {RandomChoice[{1, 2, 3, 4}]}]];
  Switch[cls,
    "symmetric",
      init = symmetricState[];
      rules = {genRule[{2}, 1, RandomChoice[{1, 2}]]};
      steps = RandomChoice[{1, 2, 2, 3}],
    "large",
      init = Table[RandomInteger[{1, RandomInteger[{6, 20}]}, RandomChoice[arities]], {RandomInteger[{12, 30}]}],
    "named",
      named = RandomChoice[namedInits];
      rules = {RandomChoice[{
        {{{1, 2}}, {{1, 3}, {3, 2}}}, {{{1, 2}}, {{2, 1}}}, {{{1, 2}}, {}},
        {{{1, 2}, {2, 3}}, {{1, 3}}}, {{{1, 2}}, {{1, 2}, {2, 3}}}}]};
      steps = RandomChoice[{1, 1, 2}];
      init = Check[Quiet[HypergraphRewriting`Private`hgInitialEdges[named[[1]],
        Append[named[[2]], "RandomSeed" -> seed]]], {{1, 2}}]
  ];
  (* a generated condition with no edges is refused by HGEvolve (HGEvolve::emptyic) *)
  If[init === {} || !ListQ[init], init = {{1, 2}}; named = None];
  roots = {init};
  If[cls === "multiroot",
    Do[AppendTo[roots, Switch[RandomInteger[{1, 3}],
        1, init /. Thread[Union[Flatten[init]] -> RandomSample[Range[Length[Union[Flatten[init]]]]] + 10],
        2, If[Length[init] > 1, Delete[init, RandomInteger[{1, Length[init]}]], init],
        3, Table[RandomInteger[{1, nv}, RandomChoice[arities]], {RandomInteger[{1, 3}]}]]],
      {RandomInteger[{1, 2}]}]];
  <|"seed" -> seed, "class" -> cls, "rules" -> rules, "roots" -> roots, "maxSteps" -> steps,
    "named" -> named, "symbolic" -> (cls === "symbolic" || RandomReal[] < 0.08),
    "packed" -> RandomReal[] < 0.3|>
];

(* ---------------- reference ---------------- *)
cfCache = <||>;
cf[edges_] := Lookup[cfCache, Key[edges], cfCache[edges] = First[CanonicalForm[edges]]];

refRun[c_] := Module[{s = c["maxSteps"], ref = $Aborted, t, cap = If[c["class"] === "large", 3000, 1500]},
  While[s >= 1,
    {t, ref} = AbsoluteTiming[TimeConstrained[
      Quiet[MultiwayEvolve[c["rules"], First[c["roots"]], s, "Roots" -> c["roots"], "GenesisEvents" -> True]],
      30, $Aborted]];
    If[ref =!= $Aborted && ref["NumEventsRaw"] <= cap, Break[]];
    s--];
  If[s < 1, Return[$Aborted]];
  Module[{evs = ref["Events"], gen, real, contents, pairs, pairsNoGen, tr, eid, causalOf},
    gen = Select[evs, TrueQ[#["genesis"]] &];
    real = Select[evs, !TrueQ[#["genesis"]] &];
    eid = Association[(#["id"] -> #) & /@ evs];
    contents = Values[ref["StateContents"]];
    pairs = ref["Causal"];
    pairsNoGen = Select[pairs, !TrueQ[eid[#[[1]]]["genesis"]] &];
    tr[p_] := If[p === {}, {}, EdgeList[TransitiveReductionGraph[Graph[DirectedEdge @@@ p]]]];
    <|"steps" -> s, "refTime" -> t, "ref" -> ref, "real" -> real, "gen" -> gen,
      "statesNone" -> Length[contents],
      "statesAutomatic" -> Length[DeleteDuplicates[contents]],
      "statesFull" -> Length[DeleteDuplicates[cf /@ contents]],
      "canonNone" -> Sort[cf /@ contents],
      "canonAutomatic" -> Sort[cf /@ DeleteDuplicates[contents]],
      "canonFull" -> Union[cf /@ contents],
      "causalPairs" -> pairsNoGen, "causalTRPairs" -> tr[pairsNoGen],
      "causalPairsGen" -> pairs, "causalTRPairsGen" -> tr[pairs],
      "branchial" -> Length[ref["Branchial"]]|>
  ]
];

(* The event count of the reference under an identity, with or without the genesis events. *)
(* Memoised per case: evCache is cleared when a case starts. *)
evCache = <||>;
refEvents[r_, identity_, genesis_] := Lookup[evCache, Key[{identity, genesis}],
  evCache[{identity, genesis}] = Module[{res = r["ref"]},
    res["Events"] = If[genesis, Join[r["gen"], r["real"]], r["real"]];
    EventCount[res, identity]]];

authRun[c_, s_] := If[!useAuth, Missing["off"], TimeConstrained[Quiet[Module[{multi, mk},
  multi = Wolfram`Multicomputation`MultiwaySystem[(#[[1]] -> #[[2]]) & /@ c["rules"], c["roots"]];
  mk[ec_] := Module[{sg, ecbg},
    sg = multi["StatesGraph", s, "CanonicalStateFunction" -> "CanonicalHypergraph",
      "CanonicalEventFunction" -> ec];
    ecbg = multi["EvolutionCausalBranchialGraph", s,
      "CanonicalStateFunction" -> "CanonicalHypergraph", "CanonicalEventFunction" -> ec];
    {VertexCount[sg], Length[Cases[VertexList[ecbg], _DirectedEdge, {1}]]}];
  <|"Full" -> mk[Full], "Automatic" -> mk[Automatic]|>]], 15, Missing["timeout"]]];

(* ---------------- engine ---------------- *)
props = {"NumStates", "NumEvents", "NumCausalEdges", "NumBranchialEdges", "States"};
countKeys = {"states", "events", "causal", "branchial"};

(* The initial condition as passed: a named condition, symbols for vertices, a packed array. *)
engineInit[c_, roots_] := Which[
  c["named"] =!= None && roots === c["roots"], c["named"][[1]],
  TrueQ[c["symbolic"]], roots /. Thread[Union[Flatten[roots]] ->
    (Symbol["v" <> IntegerString[#]] & /@ Range[Length[Union[Flatten[roots]]]])],
  TrueQ[c["packed"]] && Length[roots] == 1 && Equal @@ (Length /@ First[roots]),
    Developer`ToPackedArray[First[roots]],
  Length[roots] == 1, First[roots],
  True, roots];
namedOpts[c_] := If[c["named"] === None, {}, Append[c["named"][[2]], "RandomSeed" -> c["seed"]]];

engRun[c_, s_, opts_, extraProps_ : {}] := engRunOn[c, c["rules"], c["roots"], s, opts, extraProps];
engRunOn[c_, rules_, roots_, s_, opts_, extraProps_ : {}] := Module[{ed, r, init = engineInit[c, roots]},
  ed = EvaluationData[TimeConstrained[
    HGEvolve[(#[[1]] -> #[[2]]) & /@ rules, init, s, Join[props, extraProps],
      Sequence @@ opts, Sequence @@ If[StringQ[init], namedOpts[c], {}]], 90, $Aborted]];
  r = ed["Result"];
  If[!AssociationQ[r], Return[<|"fail" -> ToString[Short[r, 5], InputForm], "msgs" -> ed["MessagesText"]|>]];
  Join[<|"states" -> r["NumStates"], "events" -> r["NumEvents"], "causal" -> r["NumCausalEdges"],
    "branchial" -> r["NumBranchialEdges"],
    (* one state per class: "States" lists every raw state under Automatic, each naming its class *)
    "canon" -> Sort[cf[Rest /@ #["Edges"]] & /@ DeleteDuplicatesBy[Values[r["States"]], #["CanonicalId"] &]],
    "msgs" -> ed["MessagesText"]|>, KeyTake[r, extraProps]]
];

(* The configurations: state mode, event mode, quotient, reduction, genesis, device. *)
keyNames = {"InputState", "OutputState", "Step", "Rule", "ConsumedEdges", "ProducedEdges"};
configsFor[c_] := Module[{base, keyLists},
  SeedRandom[c["seed"] + 77];
  keyLists = Table[RandomSample[keyNames, RandomInteger[{1, 6}]], {2}];
  base = Flatten[Table[
      If[q && sm =!= Full && em =!= None, Nothing, {sm, em, q, tr, False, dev}],
      {dev, If[useGPU, {"CPU", "GPU"}, {"CPU"}]}, {sm, {None, Automatic, Full}},
      {em, {None, Full, Automatic}}, {q, {False, True}}, {tr, {True, False}}], 4];
  Join[base,
    {{Full, "Positional", False, True, False, "CPU"}, {None, "Positional", False, True, False, "CPU"}},
    {Full, #, False, True, False, "CPU"} & /@ keyLists,
    {Full, #, True, False, False, "CPU"} & /@ keyLists,
    {{None, First[keyLists], False, True, False, "CPU"}},
    Flatten[Table[{sm, em, q, tr, True, dev},
      {dev, If[useGPU, {"CPU", "GPU"}, {"CPU"}]}, {sm, {None, Full}}, {em, {None, Full, Automatic}},
      {q, If[sm === Full, {False, True}, {False}]}, {tr, {True}}], 4],
    {{Full, First[keyLists], False, True, True, "CPU"}, {Full, First[keyLists], True, False, True, "CPU"}}]
];
cfgOpts[{sm_, em_, q_, tr_, g_, dev_}] := {"CanonicalizeStates" -> sm, "CanonicalizeEvents" -> em,
  "ExploreFromCanonicalStatesOnly" -> q, "CausalTransitiveReduction" -> tr,
  "ShowGenesisEvents" -> g, "TargetDevice" -> dev};

(* ---------------- comparison ---------------- *)
(* {hard differences, amber and known-class values} *)
compare[r_, auth_, cfg_, e_] := Module[{sm, em, q, tr, g, dev, exp = <||>, out = {}, amber = {}, lo, hi},
  {sm, em, q, tr, g, dev} = cfg;
  If[KeyExistsQ[e, "fail"], Return[{{{"fail", e["fail"], e["msgs"]}}, {}}]];
  exp["states"] = Switch[sm, Full, r["statesFull"], Automatic, r["statesAutomatic"], _, r["statesNone"]];
  exp["canon"] = Switch[sm, Full, r["canonFull"], Automatic, r["canonAutomatic"], _, r["canonNone"]];
  exp["causal"] = Length[If[g, If[tr, r["causalTRPairsGen"], r["causalPairsGen"]],
    If[tr, r["causalTRPairs"], r["causalPairs"]]]];
  exp["branchial"] = r["branchial"];
  Which[
    em === None, exp["events"] = refEvents[r, None, g],
    em === Full, exp["events"] = refEvents[r, "States", g],
    ListQ[em], exp["events"] = refEvents[r, em, g],
    True,
      (* Automatic and Positional: rank signatures (SPEC 4.2) *)
      lo = refEvents[r, "AutomaticCanonical", g]; hi = refEvents[r, None, g];
      If[!(lo <= e["events"] <= hi),
        AppendTo[out, {"eventsBound", "expected", {lo, hi}, "got", e["events"]}]];
      Module[{pos = refEvents[r, "AutomaticPositional", g], au},
        au = If[em === Automatic && sm === Full && !g && AssociationQ[auth] &&
            auth["Full"] === {r["statesFull"], refEvents[r, "States", False]}, auth["Automatic"][[2]], Missing[]];
        If[!MissingQ[au] && au =!= e["events"], AppendTo[amber, {"eventsAuthority", cfg, au, e["events"]}]];
        If[(em === "Positional" || sm =!= Full) && pos =!= e["events"],
          AppendTo[amber, {"eventsPosAmber", cfg, pos, e["events"]}]]];
      exp["events"] = Missing["bounded"]];
  Do[If[!MissingQ[exp[k]] && exp[k] =!= e[k],
      AppendTo[out, {k, "expected", If[k === "canon", Length[exp[k]], exp[k]],
        "got", If[k === "canon", Length[e[k]], e[k]]}]],
    {k, Append[countKeys, "canon"]}];
  {out, amber}
];

(* EdgeCount of the causal graph: under EdgeDeduplication, one edge per pair of event classes;
   without it, one per edge instance a (reduced) raw pair shares. *)
causalGraphCheck[c_, r_, s_, em_] := Module[{res = {}, cls, pairs, proj, ident, mult},
  ident = Switch[em, None, None, Full, "States", _, em];
  cls = Association[(#["id"] -> EventSignature[#, ident]) & /@ r["real"]];
  pairs = List @@@ r["causalTRPairs"];
  mult = Counts[r["ref"]["CausalInstances"]];
  proj = Join @@ (ConstantArray[{cls[#[[1]]], cls[#[[2]]]}, mult[#]] & /@ pairs);
  Do[
    Module[{g = Quiet[TimeConstrained[HGEvolve[(#[[1]] -> #[[2]]) & /@ c["rules"], engineInit[c, c["roots"]], s,
          "CausalGraph", "CanonicalizeStates" -> Full, "CanonicalizeEvents" -> em,
          "EdgeDeduplication" -> dd, Sequence @@ If[c["named"] =!= None, namedOpts[c], {}]], 90, $Aborted]], want},
      want = If[dd, Length[DeleteDuplicates[proj]], Length[proj]];
      If[!GraphQ[g], AppendTo[res, {"causalGraph", em, dd, "fail", Short[g]}],
        If[{VertexCount[g], EdgeCount[g]} =!= {Length[DeleteDuplicates[Values[cls]]], want},
          AppendTo[res, {"causalGraph", em, dd, "expected", {Length[DeleteDuplicates[Values[cls]]], want},
            "got", {VertexCount[g], EdgeCount[g]}}]]]],
    {dd, {True, False}}];
  res];

(* Caps and sampling. A named condition is passed as its edge list: "RandomSeed" seeds both the
   draws and the condition's generator. *)
capChecks[c0_, r_, s_] := Module[{c = c0, res = {}, base, wide, small, runs, sm = RandomChoice[{None, Automatic, Full}],
    em = RandomChoice[{None, Full}], seedOpt = "RandomSeed" -> RandomInteger[{0, 2^31}]},
  c["named"] = None;
  base = engRun[c, s, {"CanonicalizeStates" -> sm, "CanonicalizeEvents" -> em}];
  If[KeyExistsQ[base, "fail"], Return[{{"capBase", base}}]];
  (* a cap that keeps everything *)
  Do[
    wide = engRun[c, s, {"CanonicalizeStates" -> sm, "CanonicalizeEvents" -> em, opt, seedOpt}];
    If[Lookup[wide, countKeys] =!= Lookup[base, countKeys],
      AppendTo[res, {"capWide", opt, Lookup[base, countKeys], Lookup[wide, countKeys, wide]}]],
    {opt, {"MaxStatesPerStep" -> 10^6, "MaxSuccessorStatesPerParent" -> 10^6, "MatchesPerStateRule" -> 10^6,
      "TransitionRate" -> 1., "ExplorationProbability" -> 1.}}];
  (* a small cap: one value on every route *)
  small = RandomChoice[{"MaxStatesPerStep" -> RandomInteger[{1, 4}], "MaxSuccessorStatesPerParent" -> RandomInteger[{1, 3}],
    "MatchesPerStateRule" -> RandomInteger[{1, 2}], "TransitionRate" -> RandomReal[{0.2, 0.8}],
    "ExplorationProbability" -> RandomReal[{0.3, 0.9}]}];
  runs = Association[# -> engRun[c, s, Join[{"CanonicalizeStates" -> Full, "CanonicalizeEvents" -> em, small, seedOpt}, #]] & /@
    Join[{{}, {"ExploreFromCanonicalStatesOnly" -> True}}, If[useGPU, {{"TargetDevice" -> "GPU"}}, {}]]];
  Module[{full = engRun[c, s, {"CanonicalizeStates" -> Full, "CanonicalizeEvents" -> em}], vals},
    vals = Lookup[#, countKeys, #] & /@ runs;
    If[!AllTrue[Values[runs], !KeyExistsQ[#, "fail"] &] || Length[DeleteDuplicates[Values[vals]]] > 1,
      AppendTo[res, {"capRoutes", small, vals}]];
    If[AllTrue[Values[runs], !KeyExistsQ[#, "fail"] &] && !KeyExistsQ[full, "fail"] &&
       (runs[{}]["states"] > full["states"] || runs[{}]["events"] > full["events"]),
      AppendTo[res, {"capExceeds", small, vals[{}], Lookup[full, countKeys]}]]];
  res];

(* Sessions: one step at a time, then a step split over two subsets of the frontier. *)
sessionChecks[c_, r_, s_] := Module[{res = {}, sess, rows = r["ref"]["PerStep"], got, want, fr},
  If[c["named"] =!= None || Length[c["roots"]] > 1, Return[{}]];
  sess = Quiet[HGSessionOpen[(#[[1]] -> #[[2]]) & /@ c["rules"], engineInit[c, c["roots"]],
    {"NumStates", "NumEvents", "NumCausalEdges", "NumBranchialEdges"}, "CanonicalizeStates" -> Full]];
  If[Head[sess] =!= HGSessionObject, Return[{{"sessionOpen", Short[sess]}}]];
  Do[
    got = Quiet[HGSessionStep[sess, 1]];
    want = {rows[[k, "states"]], rows[[k, "eventsNone"]] - Length[r["gen"]], Null, rows[[k, "branchial"]]};
    If[!AssociationQ[got] || {got["NumStates"], got["NumEvents"], got["NumBranchialEdges"]} =!= want[[{1, 2, 4}]],
      AppendTo[res, {"sessionStep", k, want[[{1, 2, 4}]], If[AssociationQ[got], Lookup[got, {"NumStates", "NumEvents", "NumBranchialEdges"}], got]}]],
    {k, 1, s - 1}];
  got = Quiet[HGSessionStep[sess, 1]];
  want = {rows[[s, "states"]], rows[[s, "eventsNone"]] - Length[r["gen"]], rows[[s, "branchial"]]};
  If[!AssociationQ[got] || Lookup[got, {"NumStates", "NumEvents", "NumBranchialEdges"}] =!= want,
    AppendTo[res, {"sessionStep", s, want, If[AssociationQ[got], Lookup[got, {"NumStates", "NumEvents", "NumBranchialEdges"}], got]}]];
  Quiet[HGSessionClose[sess]];
  (* The last step from two halves of the frontier, then from an empty list. Under
     CanonicalizeStates -> None a frontier id names one raw state; under Full it names a class,
     and a class can have frontier states at two depths. *)
  sess = Quiet[HGSessionOpen[(#[[1]] -> #[[2]]) & /@ c["rules"], engineInit[c, c["roots"]],
    {"NumStates", "NumEvents", "NumCausalEdges", "NumBranchialEdges"}, "CanonicalizeStates" -> None]];
  If[Head[sess] =!= HGSessionObject, Return[Append[res, {"sessionOpen", Short[sess]}]]];
  If[s > 1, Quiet[HGSessionStep[sess, s - 1]]];
  fr = Quiet[HGSessionFrontier[sess]];
  If[ListQ[fr] && Length[fr] >= 2,
    Quiet[HGSessionStep[sess, 1, "From" -> Take[fr, Floor[Length[fr]/2]]]];
    Quiet[HGSessionStep[sess, 1, "From" -> Drop[fr, Floor[Length[fr]/2]]]],
    Quiet[HGSessionStep[sess, 1]]];
  got = Quiet[HGSessionStep[sess, 1, "From" -> {}]];
  want = {Length[c["roots"]] + rows[[s, "eventsNone"]] - Length[r["gen"]],
    rows[[s, "eventsNone"]] - Length[r["gen"]], rows[[s, "branchial"]]};
  If[!AssociationQ[got] || Lookup[got, {"NumStates", "NumEvents", "NumBranchialEdges"}] =!= want,
    AppendTo[res, {"sessionSplit", s, want, If[AssociationQ[got], Lookup[got, {"NumStates", "NumEvents", "NumBranchialEdges"}], got]}]];
  Quiet[HGSessionClose[sess]];
  res];

(* Presentation invariance: relabel root vertices, permute root edges, rename rule variables, and
   (outside Automatic event identity) permute each rule's left and right edges. Automatic takes
   consumed ranks in left-side order and produced ranks in right-side order (SPEC 4.2), so two
   rules that differ only in that order identify different events; Automatic state identity
   compares edge lists and is not checked here. *)
presentationChecks[c_, s_, res_] := Module[{c2 = c, perm, out = {}},
  If[c["named"] =!= None, Return[{}]];
  SeedRandom[c["seed"] + 1000003];
  perm = With[{vs = Union[Flatten[c["roots"]]], off = RandomChoice[{0, 20, 10^6, 2^31 - 100, 2^32 - 100}]},
    Thread[vs -> RandomSample[Range[off, off + Length[vs] - 1]]]];
  c2["roots"] = RandomSample[RandomSample /@ (c["roots"] /. perm)];
  c2["symbolic"] = False; c2["packed"] = False;
  Do[
    Module[{base = SelectFirst[res, #[[1]] === cfg &, None], alt, rules2},
      rules2 = Function[rr, Module[{vs = Union[Flatten[rr]], ren},
          ren = Thread[vs -> (RandomSample[Range[Length[vs]]] + 50)];
          If[cfg[[2]] === Automatic, rr /. ren,
            {RandomSample[rr[[1]] /. ren], RandomSample[rr[[2]] /. ren]}]]] /@ c["rules"];
      If[base =!= None && !KeyExistsQ[base[[2]], "fail"],
        alt = engRunOn[c2, rules2, c2["roots"], s, cfgOpts[cfg]];
        If[KeyExistsQ[alt, "fail"] || Lookup[alt, countKeys] =!= Lookup[base[[2]], countKeys],
          AppendTo[out, {"presentation", cfg, Lookup[base[[2]], countKeys], Lookup[alt, countKeys, alt]}]]]],
    {cfg, {{None, None, False, True, False, "CPU"}, {Full, Full, False, True, False, "CPU"},
      {Full, Automatic, False, True, False, "CPU"}, {Full, Automatic, True, True, False, "CPU"},
      {Full, None, True, False, True, "CPU"}, {None, Full, False, True, False, "GPU"}}}];
  out];

(* A named condition gives the numbers of its own edge list. *)
namedChecks[c_, s_, res_] := Module[{explicit, base = SelectFirst[res, #[[1]] === {Full, Full, False, True, False, "CPU"} &, None], alt},
  If[c["named"] === None || base === None, Return[{}]];
  explicit = c; explicit["named"] = None;
  alt = engRun[explicit, s, cfgOpts[{Full, Full, False, True, False, "CPU"}]];
  If[Lookup[alt, countKeys, alt] =!= Lookup[base[[2]], countKeys], {{"named", Lookup[base[[2]], countKeys], Lookup[alt, countKeys, alt]}}, {}]];

(* ---------------- main loop ---------------- *)
nHard = 0; nCases = 0;
Do[
  If[minutes > 0 && AbsoluteTime[] - tStart > 60 minutes, Print["budget reached before seed ", seed]; Break[]];
  Module[{c = genCase[seed], r, s, auth, res = {}, discs = {}, amber = {}, t0 = AbsoluteTime[]},
    evCache = <||>; cfCache = <||>;
    r = refRun[c];
    If[verbose, Print["  reference ", Round[AbsoluteTime[] - t0, 0.01], " s"]];
    If[r === $Aborted,
      logLine[resFile, ToString[<|"seed" -> seed, "skip" -> "reference too slow", "case" -> c|>, InputForm]];
      Print["seed ", seed, " ", c["class"], " SKIP"]; Continue[]];
    s = r["steps"];
    auth = authRun[c, s];
    Do[
      Module[{t1 = AbsoluteTime[], e = engRun[c, s, cfgOpts[cfg]], d, am},
        If[verbose, Print["  ", cfg, " ", Round[AbsoluteTime[] - t1, 0.01], " s ", Lookup[e, countKeys, e]]];
        {d, am} = compare[r, auth, cfg, e];
        amber = Join[amber, am];
        AppendTo[res, {cfg, KeyDrop[e, "canon"], d}];
        If[d =!= {}, AppendTo[discs, {cfg, d, e["msgs"]}]]],
      {cfg, configsFor[c]}];
    (* within one cell, quotient and device must agree *)
    Module[{cells = GatherBy[Select[res, !KeyExistsQ[#[[2]], "fail"] &], #[[1, {1, 2, 4, 5}]] &]},
      Do[
        Module[{vals = {#[[1]], Lookup[#[[2]], countKeys]} & /@ cell},
          If[Length[DeleteDuplicates[vals[[All, 2]]]] > 1,
            AppendTo[discs, {"inconsistent", cell[[1, 1, {1, 2, 4, 5}]], vals}]]],
        {cell, cells}]];
    Do[
      Module[{t1 = AbsoluteTime[]},
        discs = Join[discs, check[[2]][]];
        If[verbose, Print["  ", check[[1]], " ", Round[AbsoluteTime[] - t1, 0.01], " s"]]],
      {check, {{"causalGraph", causalGraphCheck[c, r, s, RandomChoice[{None, Full}]] &},
        {"caps", capChecks[c, r, s] &}, {"sessions", sessionChecks[c, r, s] &},
        {"presentation", presentationChecks[c, s, res] &}, {"named", namedChecks[c, s, res] &}}}];
    nCases++;
    If[discs =!= {}, nHard++];
    logLine[resFile, ToString[<|"seed" -> seed, "case" -> c, "steps" -> s, "rawEvents" -> Length[r["real"]],
      "auth" -> auth, "amber" -> amber, "engine" -> res, "discs" -> discs|>, InputForm]];
    If[discs =!= {},
      logLine[discFile, ToString[<|"seed" -> seed, "class" -> c["class"], "rules" -> c["rules"], "roots" -> c["roots"],
        "named" -> c["named"], "symbolic" -> c["symbolic"], "packed" -> c["packed"], "steps" -> s,
        "discs" -> discs|>, InputForm]]];
    Print["seed ", seed, " ", c["class"], " steps ", s, " raw ", Length[r["real"]], " st ", r["statesFull"],
      " amber ", Length[amber], " discs ", Length[discs], " t ", Round[AbsoluteTime[] - t0, 0.1],
      If[discs =!= {}, " " <> ToString[Short[DeleteDuplicates[
        If[ListQ[#[[1]]], {#[[1]], #[[2, All, 1]]}, #[[1 ;; Min[2, Length[#]]]]] & /@ discs], 6], InputForm], ""]];
  ],
  {seed, seedFrom, seedTo}];
Print["DONE cases ", nCases, " with hard differences ", nHard];
Exit[If[nHard > 0, 1, 0]]
