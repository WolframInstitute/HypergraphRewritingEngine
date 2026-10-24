(* ::Package:: *)

(* MultiwayReference.wl

   Independent brute-force reference implementation of multiway hypergraph
   rewriting, used as a ground-truth oracle for the C++/GPU engine.

   Design goals (deliberately simple, "no tricks, no indexes"):
     - Every match is found by exhaustive search over edge tuples.
     - State canonicalization is exact: color refinement to an equitable
       partition, then the lexicographic-minimum relabeling over all
       within-cell permutations (up to $ExhaustiveLabelings of them) or over the
       leaves of an individualization-refinement search without automorphism
       pruning (above it).
     - Multiway expansion is generation-based with global canonical dedup:
       step k expands the states first reached at generation k-1; equal
       (canonical) states are merged across the whole evolution.

   Data model (matches the engine):
     - A state is a multiset of ordered hyperedges.
     - An edge is a list of integer vertices; within-edge order is significant.
     - Duplicate edges (multiplicity) and self-loops/repeated vertices are allowed.
     - Rule = {lhs, rhs}; lhs/rhs are lists of edges whose entries are variables
       (any atoms, typically integers). Variables in rhs not in lhs become fresh
       vertices, allocated globally and per event. Vertex binding need not be
       injective across distinct variables (default Wolfram pattern semantics).

   Causal edge A -> B iff event B consumes an edge instance produced by event A.
   Branchial edge (default) connects sibling events that expand the SAME parent
   state and whose consumed-edge sets overlap. Transitive reduction is applied to
   the causal graph by default. *)

BeginPackage["MultiwayReference`"];

MultiwayEvolve::usage =
  "MultiwayEvolve[rules, init, steps] evolves the multiway system and returns an \
Association with per-step result tuples and the underlying graph data. rules is a \
list of {lhs, rhs}; lhs/rhs are lists of edges (an edge is a list of integer \
variables); init is a list of edges (the single initial state).";

CanonicalForm::usage =
  "CanonicalForm[edges] returns {canonicalEdges, labeling} for a list of ordered \
hyperedges: canonicalEdges uses vertex labels 0..n-1 in lexicographically minimal \
order; labeling is the chosen Association originalVertex -> label.";

EventSignature::usage =
  "EventSignature[event, identity] gives the identity value of one event of a MultiwayEvolve \
result. identity is None, \"States\", \"AutomaticPositional\", \"AutomaticCanonical\" or a \
list of the keys \"InputState\", \"OutputState\", \"Step\", \"Rule\", \"ConsumedEdges\", \
\"ProducedEdges\".";

EventCount::usage =
  "EventCount[result, identity, k] counts the distinct EventSignature values of the events of \
result at steps up to k.";

SingleHistoryEvolve::usage =
  "SingleHistoryEvolve[rules, init, n, ordering] applies up to n events one at a time. Each \
event is the match first in ordering, a list of SetReplace's \"EventOrderingFunction\" names \
in priority order. rules and init are as for MultiwayEvolve. The result is <|\"Events\" -> \
{{rule, inputs -> outputs}, ...}, \"Expressions\" -> edges|>, numbered from 1 as SetReplace's \
\"AllEventsList\" and \"AllEventsEdgesList\" are. An expression's index is its creation order.";

CanonicalForm::big =
  "The canonical labelling search passed `1` leaves.";

Begin["`Private`"];

(* ------------------------------------------------------------------ *)
(* Color refinement to an equitable, isomorphism-invariant partition.  *)
(* Returns an Association vertex -> integer color rank.                 *)
(* ------------------------------------------------------------------ *)

(* Equitable color refinement. `ecol` is a per-edge color (used to mark consumed/
   produced edges for the principled event canonicalization); pass all-equal for a
   plain hypergraph. Returns Association vertex -> integer color rank. *)
refineColors[edges_, verts_, ecol_] := refineColors[edges, verts, ecol, None];

(* `init` is None or an Association vertex -> orderable value that seeds the colouring; the
   individualisation search passes the current colouring with one vertex split off. *)
refineColors[edges_, verts_, ecol_, init_] := Module[
  {incid, color, sig, distinct, rmap, nColors, prevN = -1},

  (* incidences[v] = list of {position, arity, edgeColor, edgeIndex} for every slot v fills *)
  incid = AssociationThread[verts -> Table[{}, {Length[verts]}]];
  Do[
    With[{e = edges[[ei]], ar = Length[edges[[ei]]], ec = ecol[[ei]]},
      Do[ incid[e[[p]]] = Append[incid[e[[p]]], {p, ar, ec, ei}], {p, ar}]
    ],
    {ei, Length[edges]}
  ];

  (* initial color: sorted multiset of {position, arity, edgeColor} over incidences *)
  color = If[init === None,
    AssociationMap[Sort[incid[#][[All, {1, 2, 3}]]] &, verts],
    AssociationMap[{init[#], Sort[incid[#][[All, {1, 2, 3}]]]} &, verts]];

  (* iterate until the number of color classes stabilizes *)
  While[True,
    sig = AssociationMap[
      Function[v,
        {
          color[v],
          Sort[
            Function[inc,
              With[{p = inc[[1]], ar = inc[[2]], ec = inc[[3]], e = edges[[inc[[4]]]]},
                {p, ar, ec, Sort[Table[{q, color[e[[q]]]}, {q, ar}]]}
              ]
            ] /@ incid[v]
          ]
        }
      ],
      verts
    ];
    distinct = Sort[DeleteDuplicates[Values[sig]]];
    rmap = AssociationThread[distinct -> Range[Length[distinct]]];
    color = AssociationMap[rmap[sig[#]] &, verts];
    nColors = Length[distinct];
    If[nColors === prevN, Break[]];
    prevN = nColors;
  ];
  color
];

(* ------------------------------------------------------------------ *)
(* Exact canonical form: lex-min relabeling over within-cell perms.    *)
(* ------------------------------------------------------------------ *)

(* Up to this many within-cell labelings the form is the lex-minimum over all of them. Above
   it, the candidates are the leaves of the individualisation-refinement tree: individualise
   each vertex of the first non-singleton cell in turn, refine, recurse. Both candidate sets
   are isomorphism-invariant and the choice between them depends only on the refined cells, so
   either gives a complete invariant; the tree has n leaves on an n-cycle where the cell
   product has n!. *)
$ExhaustiveLabelings = 40320;
$MaxLeaves = 500000;

irLeaves[edges_, verts_, ecol_, color0_] := Module[{search, nLeaves = 0},
  search[col_] := Module[{cells = GatherBy[SortBy[verts, col], col], target},
    target = SelectFirst[cells, Length[#] > 1 &, None];
    If[target === None,
      nLeaves++;
      If[nLeaves === $MaxLeaves + 1, Message[CanonicalForm::big, $MaxLeaves]];
      {Flatten[cells]},
      Join @@ (Function[v,
        search[refineColors[edges, verts, ecol,
          AssociationMap[{col[#], If[# === v, 0, 1]} &, verts]]]] /@ target)]];
  search[color0]
];

CanonicalForm[edges_] := CanonicalForm[edges, ConstantArray[0, Length[edges]]];

CanonicalForm[edges_, ecol_] := Module[
  {verts, color, cells, nLab, candidates, best},

  If[edges === {}, Return[{{}, <||>}]];
  verts = Union[Flatten[edges]];
  color = refineColors[edges, verts, ecol];

  (* cells in increasing color order; permute within each cell only *)
  cells = GatherBy[SortBy[verts, color], color];
  nLab = Times @@ (Factorial[Length[#]] & /@ cells);

  (* each canonical edge is {edgeColor, relabeled vertices} *)
  candidates = Function[order,
    Module[{lmap, relabeled},
      lmap = AssociationThread[order -> Range[0, Length[order] - 1]];
      relabeled = Sort[MapThread[{#2, Map[lmap, #1]} &, {edges, ecol}]];
      {relabeled, lmap}
    ]
  ] /@ If[nLab <= $ExhaustiveLabelings,
    Flatten /@ Tuples[Permutations /@ cells],
    irLeaves[edges, verts, ecol, color]];

  (* min by canonical (colored) edge list, tie-broken by the label vector for determinism *)
  best = First[
    SortBy[candidates, {First, Lookup[#[[2]], Sort[verts]] &}]
  ];
  best
];

(* ------------------------------------------------------------------ *)
(* Exhaustive matching: all ordered injective edge assignments with a  *)
(* consistent (non-injective-allowed) vertex binding.                  *)
(* Returns a list of {matchedEdgeIndices, binding}.                    *)
(* ------------------------------------------------------------------ *)

findMatches[lhs_, state_] := Module[{res = {}, recurse},
  recurse[pos_, used_, binding_] := If[
    pos > Length[lhs],
    AppendTo[res, {used, binding}],
    With[{pe = lhs[[pos]], ar = Length[lhs[[pos]]]},
      Do[
        If[! MemberQ[used, ei] && Length[state[[ei]]] === ar,
          Module[{b = binding, ok = True, e = state[[ei]]},
            Do[
              With[{v = pe[[k]], a = e[[k]]},
                If[KeyExistsQ[b, v], If[b[v] =!= a, ok = False], b[v] = a]
              ];
              If[! ok, Break[]],
              {k, ar}
            ];
            If[ok, recurse[pos + 1, Append[used, ei], b]]
          ]
        ],
        {ei, Length[state]}
      ]
    ]
  ];
  recurse[1, {}, <||>];
  res
];

(* ------------------------------------------------------------------ *)
(* Multiway evolution.                                                  *)
(* ------------------------------------------------------------------ *)

(* Options (see reference/CANONICALIZATION.md for the full mapping to the engine
   and to the MultiwaySystem paclet). The per-step result always surfaces the event
   count under every convention; "EventCanonicalization"/"EventEdgeIdentity" select
   which one is echoed as the primary "events" field. *)
Options[MultiwayEvolve] = {
  "StateCanonicalization" -> "Canonical",      (* "Canonical" (merge isomorphic) | "Content" (merge equal edge lists, edges in creation order) | "None" *)
  "EventCanonicalization" -> "None",           (* "None" | "States" (=Full) | "Automatic" *)
  "EventEdgeIdentity" -> "Positional",         (* "Positional" (MultiwaySystem) | "Canonical" (IR merge) *)
  "TransitiveReduction" -> True,
  "FullCapture" -> True,
  "GenesisEvents" -> False,                    (* True: one genesis event per root, producing the root's edges *)
  "Roots" -> None                              (* None: single `init`. A list of edge-lists: seed each as a distinct root. *)
};

(* Event-canonicalization conventions (see reference/CANONICALIZATION.md):

   "States" (MultiwaySystem CanonicalEventFunction -> Full):
     identity = (canonical input state, canonical output state). Coarsest.

   "Automatic-Positional" (MultiwaySystem CanonicalEventFunction -> Automatic):
     States + step + consumed/produced edges identified by canonical edge RANK and
     kept ORDERED by match/rhs position. Does NOT quotient state automorphisms, so
     symmetric edge-role assignments stay DISTINCT (e.g. two {1,1} self-loop matches
     count as 2). Matches the MultiwaySystem paclet.

   "Automatic-Canonical" (principled IR merge; no paclet equivalent):
     States + step, where consumed/produced edges are marked as edge colors and the
     marked hypergraph is canonicalized. DOES quotient automorphisms, so genuinely
     isomorphic events merge (the two self-loop matches count as 1). *)

eventSigStates[ev_] := {First[CanonicalForm[ev["inContents"]]], First[CanonicalForm[ev["outContents"]]]};

eventSigAutomaticPositional[ev_] := Module[
  {cin, lin, cout, lout, inRelab, outRelab, rankIn, rankOut, np, producedPos},
  {cin, lin} = CanonicalForm[ev["inContents"]];
  {cout, lout} = CanonicalForm[ev["outContents"]];
  inRelab = Map[lin, ev["inContents"], {2}];
  outRelab = Map[lout, ev["outContents"], {2}];
  rankIn = InversePermutation[Ordering[Table[{inRelab[[i]], i}, {i, Length[inRelab]}]]];
  rankOut = InversePermutation[Ordering[Table[{outRelab[[i]], i}, {i, Length[outRelab]}]]];
  np = Length[ev["producedContents"]];
  producedPos = Range[Length[ev["outContents"]] - np + 1, Length[ev["outContents"]]];
  {cin, cout, ev["step"], rankIn[[ev["consumed"]]], rankOut[[producedPos]]}
];

eventSigAutomaticCanonical[ev_] := Module[{inN, outN, np, inCol, outCol},
  inN = Length[ev["inContents"]]; outN = Length[ev["outContents"]];
  np = Length[ev["producedContents"]];
  inCol = ReplacePart[ConstantArray[0, inN], (# -> 1) & /@ ev["consumed"]];
  outCol = Join[ConstantArray[0, outN - np], ConstantArray[1, np]];
  {First[CanonicalForm[ev["inContents"], inCol]], First[CanonicalForm[ev["outContents"], outCol]], ev["step"]}
];

(* A key list (SPEC 4.2): "ConsumedEdges" is the input state with the consumed edges marked,
   "ProducedEdges" the output state with the produced edges marked. *)
eventSigKeys[ev_, keys_List] := Module[{inN, outN, np, comp},
  inN = Length[ev["inContents"]]; outN = Length[ev["outContents"]];
  np = Length[ev["producedContents"]];
  comp["InputState"] := First[CanonicalForm[ev["inContents"]]];
  comp["OutputState"] := First[CanonicalForm[ev["outContents"]]];
  comp["Step"] := ev["step"];
  comp["Rule"] := ev["rule"];
  comp["ConsumedEdges"] := First[CanonicalForm[ev["inContents"],
    ReplacePart[ConstantArray[0, inN], (# -> 1) & /@ ev["consumed"]]]];
  comp["ProducedEdges"] := First[CanonicalForm[ev["outContents"],
    Join[ConstantArray[0, outN - np], ConstantArray[1, np]]]];
  {#, comp[#]} & /@ Sort[DeleteDuplicates[keys]]
];

(* The signature of an event under an identity: None, "States", "AutomaticPositional",
   "AutomaticCanonical" or a key list. A genesis event is its own event under every identity
   (SPEC 5.2: one genesis event per initial state). *)
EventSignature[ev_, identity_] := Which[
  TrueQ[ev["genesis"]], {"genesis", ev["id"]},
  identity === None, ev["id"],
  identity === "States", eventSigStates[ev],
  identity === "AutomaticPositional", eventSigAutomaticPositional[ev],
  identity === "AutomaticCanonical", eventSigAutomaticCanonical[ev],
  ListQ[identity], eventSigKeys[ev, identity],
  True, $Failed];

(* The number of events of a MultiwayEvolve result up to step k (default: all) under an identity. *)
EventCount[res_Association, identity_, k_ : Infinity] :=
  Length[DeleteDuplicates[EventSignature[#, identity] & /@ Select[res["Events"], #["step"] <= k &]]];

MultiwayEvolve[rules0_, init_, steps_, OptionsPattern[]] := Module[
  {rules, stateMode, trCausal, fullCapture, evCanon, evEdge, primaryKey, genesis,
   eventCounter = 0, sidCounter = 0, freshCounter,
   statesByCanon = <||>, repEdges = <||>, sidStep = <||>, canonOf = <||>, currentStep,
   events = {}, causal = {}, branchial = {}, frontier, mkState, eStep, perStep},

  stateMode = Switch[OptionValue["StateCanonicalization"],
    "None" | False | None, "None",
    "Content", "Content",
    _, "Canonical"];
  genesis = TrueQ[OptionValue["GenesisEvents"]];
  trCausal = OptionValue["TransitiveReduction"] =!= False;
  fullCapture = OptionValue["FullCapture"] =!= False;
  evCanon = OptionValue["EventCanonicalization"];
  evEdge = OptionValue["EventEdgeIdentity"];
  primaryKey = Switch[{evCanon, evEdge},
    {"States", _}, "eventsStates",
    {"Automatic", "Canonical"}, "eventsAutomaticCanonical",
    {"Automatic", _}, "eventsAutomaticPositional",
    _, "eventsNone"
  ];

  (* normalize rules to {lhs, rhs, newVars} *)
  rules = Function[r,
    Module[{lhs = r[[1]], rhs = r[[2]]},
      {lhs, rhs, Complement[DeleteDuplicates[Flatten[rhs]], DeleteDuplicates[Flatten[lhs]]]}
    ]
  ] /@ rules0;

  (* fresh vertices are integers above every integer vertex of the roots; a symbolic vertex
     cannot equal an integer *)
  freshCounter = With[{allInit = If[OptionValue["Roots"] === None, {init}, OptionValue["Roots"]]},
    Max[Append[Cases[Flatten[allInit], _Integer], 0]] + 1];

  (* create a state node from an instance list {{contents, producerEvent}, ...}.
     With FullCapture, every output is its own node (the multiway forest of raw
     states); num_states then counts distinct canonical classes among nodes.
     Without FullCapture, isomorphic states are merged into one expanded node
     (the deduplicated multiway graph). *)
  mkState[instList_] := Module[{contents = instList[[All, 1]], cf, sid},
    cf = Switch[stateMode,
      "Canonical", First[CanonicalForm[contents]],
      "Content", contents,
      _, {"raw", sidCounter + 1}];
    If[! fullCapture && KeyExistsQ[statesByCanon, cf],
      {statesByCanon[cf], False},
      sidCounter++; sid = sidCounter;
      repEdges[sid] = instList; sidStep[sid] = currentStep; canonOf[sid] = cf;
      If[! fullCapture, statesByCanon[cf] = sid];
      {sid, True}
    ]
  ];

  (* genesis: seed the initial state(s). Each provided root is a distinct entry
     point (reference MultiwaySystem semantics); under state canonicalization
     isomorphic roots merge iff FullCapture is off. *)
  currentStep = 0;
  With[{roots = If[OptionValue["Roots"] === None, {init}, OptionValue["Roots"]]},
    Do[
      If[genesis,
        (* the genesis event produces the root's edges; its id is the producer of each *)
        Module[{eid = ++eventCounter, sid},
          sid = First[mkState[{#, eid} & /@ root]];
          AppendTo[events, <|"id" -> eid, "genesis" -> True, "rule" -> 0, "in" -> 0, "out" -> sid,
            "consumed" -> {}, "step" -> 0, "inContents" -> {}, "outContents" -> root,
            "consumedContents" -> {}, "producedContents" -> root|>]],
        mkState[{#, 0} & /@ root]],
      {root, roots}]
  ];
  frontier = DeleteDuplicates[Range[sidCounter]];

  Do[
    currentStep = stepNum;
    Module[{newFrontier = {}},
      Do[
        Module[{inst = repEdges[fsid], contents, evs = {}},
          contents = inst[[All, 1]];
          Do[
            Module[{lhs = rules[[ri, 1]], rhs = rules[[ri, 2]], newVars = rules[[ri, 3]], matches},
              matches = findMatches[lhs, contents];
              Do[
                Module[{used = m[[1]], binding = m[[2]], eid, fresh, full,
                        produced, surviving, outInst, outSid, isNew},
                  eventCounter++; eid = eventCounter;
                  fresh = AssociationThread[newVars -> Table[freshCounter++, {Length[newVars]}]];
                  full = Join[binding, fresh];
                  produced = Map[full, rhs, {2}];
                  surviving = Delete[inst, List /@ used];
                  outInst = Join[surviving, {#, eid} & /@ produced];
                  {outSid, isNew} = mkState[outInst];
                  If[isNew, AppendTo[newFrontier, outSid]];
                  AppendTo[events, <|
                    "id" -> eid, "rule" -> ri, "in" -> fsid, "out" -> outSid,
                    "consumed" -> used, "step" -> stepNum,
                    "inContents" -> contents, "outContents" -> outInst[[All, 1]],
                    "consumedContents" -> inst[[used, 1]], "producedContents" -> produced
                  |>];
                  Do[
                    With[{p = inst[[ci, 2]]}, If[p =!= 0, AppendTo[causal, {p, eid}]]],
                    {ci, used}
                  ];
                  AppendTo[evs, <|"id" -> eid, "consumed" -> used|>];
                ],
                {m, matches}
              ]
            ],
            {ri, Length[rules]}
          ];
          (* branchial: sibling events of this parent with overlapping consumed sets *)
          Do[
            If[Intersection[evs[[a, "consumed"]], evs[[b, "consumed"]]] =!= {},
              AppendTo[branchial, Sort[{evs[[a, "id"]], evs[[b, "id"]]}]]
            ],
            {a, Length[evs]}, {b, a + 1, Length[evs]}
          ];
        ],
        {fsid, frontier}
      ];
      frontier = DeleteDuplicates[newFrontier];
    ],
    {stepNum, steps}
  ];

  eStep = Association[(#["id"] -> #["step"]) & /@ events];

  perStep = Table[
    Module[{evK = Select[events, #["step"] <= k &], pairs, g, caK, brK, row},
      pairs = DeleteDuplicates[Select[causal, eStep[#[[2]]] <= k &]];
      caK = If[pairs === {}, 0,
        g = Graph[DirectedEdge @@@ pairs];
        EdgeCount[If[trCausal, TransitiveReductionGraph[g], g]]
      ];
      brK = Length[DeleteDuplicates[Select[branchial, eStep[#[[1]]] <= k &]]];
      row = <|
        "step" -> k,
        "states" -> Length[DeleteDuplicates[canonOf /@ Select[Range[sidCounter], sidStep[#] <= k &]]],
        (* event counts under each canonicalization convention, surfaced side by side *)
        "eventsNone" -> Length[evK],
        "eventsStates" -> Length[DeleteDuplicates[EventSignature[#, "States"] & /@ evK]],
        "eventsAutomaticPositional" -> Length[DeleteDuplicates[EventSignature[#, "AutomaticPositional"] & /@ evK]],
        "eventsAutomaticCanonical" -> Length[DeleteDuplicates[EventSignature[#, "AutomaticCanonical"] & /@ evK]],
        "causal" -> caK,
        "branchial" -> brK
      |>;
      Append[row, "events" -> row[primaryKey]]
    ],
    {k, 1, steps}
  ];

  <|
    "PerStep" -> perStep,
    "NumStates" -> sidCounter,
    "NumEventsRaw" -> eventCounter,
    (* one entry per state node: the node's identity key under the state mode, and its edges *)
    "StateKeys" -> canonOf,
    "StateContents" -> (#[[All, 1]] & /@ repEdges),
    "Events" -> events,
    "Causal" -> DeleteDuplicates[causal],
    (* one {producer, consumer} entry per consumed edge instance *)
    "CausalInstances" -> causal,
    "Branchial" -> DeleteDuplicates[branchial]
  |>
];

(* ------------------------------------------------------------------ *)
(* Single history under SetReplace's event ordering functions          *)
(* (libSetReplace/HypergraphMatcher.cpp, MatchComparator).             *)
(* A match is {rule, tokens}: its rule index and its expression indices *)
(* in pattern order. Each name compares one list of the two matches:    *)
(* the indices sorted ascending (OldestEdge, reversed LeastOldEdge),    *)
(* sorted descending (LeastRecentEdge, reversed NewestEdge), in pattern *)
(* order (RuleOrdering, reversed ReverseRuleOrdering), or the rule      *)
(* (RuleIndex, reversed ReverseRuleIndex). Lists compare by their first *)
(* differing entry, and a prefix comes before a longer list.            *)
(* ------------------------------------------------------------------ *)

srOrderList[name_, {rule_, tokens_}] := Switch[name,
  "OldestEdge" | "LeastOldEdge", Sort[tokens],
  "LeastRecentEdge" | "NewestEdge", ReverseSort[tokens],
  "RuleOrdering" | "ReverseRuleOrdering", tokens,
  "RuleIndex" | "ReverseRuleIndex", {rule}];

srReversed[name_] := MemberQ[{"LeastOldEdge", "NewestEdge", "ReverseRuleOrdering", "ReverseRuleIndex"}, name];

srCompareLists[a_, b_] := Module[{i = 1},
  While[i <= Min[Length[a], Length[b]] && a[[i]] === b[[i]], i++];
  Which[
    i <= Min[Length[a], Length[b]], If[a[[i]] < b[[i]], -1, 1],
    Length[a] == Length[b], 0,
    Length[a] < Length[b], -1,
    True, 1]];

srBefore[ordering_, m1_, m2_] := Catch[
  Do[With[{c = srCompareLists[srOrderList[name, m1], srOrderList[name, m2]]},
      If[c != 0, Throw[If[srReversed[name], c > 0, c < 0]]]],
    {name, ordering}];
  False];

SingleHistoryEvolve[rules0_, init_, n_Integer, ordering_List] := Module[
  {rules, exprs = init, alive = Range[Length[init]], events = {}, fresh, matches, best,
   rule, binding, outs},
  rules = Function[r, {r[[1]], r[[2]],
    Complement[DeleteDuplicates[Flatten[r[[2]]]], DeleteDuplicates[Flatten[r[[1]]]]]}] /@ rules0;
  fresh = Max[Append[Cases[Flatten[init], _Integer], 0]] + 1;
  Do[
    matches = Flatten[Table[
      {{ri, alive[[#[[1]]]]}, #[[2]]} & /@ findMatches[rules[[ri, 1]], exprs[[alive]]],
      {ri, Length[rules]}], 1];
    If[matches === {}, Break[]];
    best = First[matches];
    Do[If[srBefore[ordering, m[[1]], best[[1]]], best = m], {m, Rest[matches]}];
    rule = best[[1, 1]];
    binding = best[[2]];
    Do[binding[v] = fresh++, {v, rules[[rule, 3]]}];
    outs = Replace[rules[[rule, 2]], binding, {2}];
    AppendTo[events, {rule, best[[1, 2]] -> Range[Length[exprs] + 1, Length[exprs] + Length[outs]]}];
    exprs = Join[exprs, outs];
    alive = Join[DeleteCases[alive, Alternatives @@ best[[1, 2]]],
      Range[Length[exprs] - Length[outs] + 1, Length[exprs]]],
    {n}];
  <|"Events" -> events, "Expressions" -> exprs|>
];

End[];

EndPackage[];
