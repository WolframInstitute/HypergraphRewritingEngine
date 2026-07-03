(* ::Package:: *)

(* MultiwayReference.wl

   Independent brute-force reference implementation of multiway hypergraph
   rewriting, used as a ground-truth oracle for the C++/GPU engine.

   Design goals (deliberately simple, "no tricks, no indexes"):
     - Every match is found by exhaustive search over edge tuples.
     - State canonicalization is exact: color refinement to an equitable
       partition, then exhaustive lexicographic-minimum relabeling over all
       within-cell permutations. This is at least as capable as the engine's
       nauty-style individualization-refinement (it finds the true canonical
       form; it is only slower on highly symmetric graphs).
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

CanonicalForm::big =
  "Automorphism cell product `1` exceeds the search cap; canonical form may not be exact.";

Begin["`Private`"];

(* ------------------------------------------------------------------ *)
(* Color refinement to an equitable, isomorphism-invariant partition.  *)
(* Returns an Association vertex -> integer color rank.                 *)
(* ------------------------------------------------------------------ *)

(* Equitable color refinement. `ecol` is a per-edge color (used to mark consumed/
   produced edges for the principled event canonicalization); pass all-equal for a
   plain hypergraph. Returns Association vertex -> integer color rank. *)
refineColors[edges_, verts_, ecol_] := Module[
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
  color = AssociationMap[Sort[incid[#][[All, {1, 2, 3}]]] &, verts];

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

CanonicalForm[edges_] := CanonicalForm[edges, ConstantArray[0, Length[edges]]];

CanonicalForm[edges_, ecol_] := Module[
  {verts, color, cells, nLab, candidates, best},

  If[edges === {}, Return[{{}, <||>}]];
  verts = Union[Flatten[edges]];
  color = refineColors[edges, verts, ecol];

  (* cells in increasing color order; permute within each cell only *)
  cells = GatherBy[SortBy[verts, color], color];
  nLab = Times @@ (Factorial[Length[#]] & /@ cells);
  If[nLab > 500000, Message[CanonicalForm::big, nLab]];

  (* each canonical edge is {edgeColor, relabeled vertices} *)
  candidates = Function[choice,
    Module[{order = Flatten[choice], lmap, relabeled},
      lmap = AssociationThread[order -> Range[0, Length[order] - 1]];
      relabeled = Sort[MapThread[{#2, Map[lmap, #1]} &, {edges, ecol}]];
      {relabeled, lmap}
    ]
  ] /@ Tuples[Permutations /@ cells];

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
  "StateCanonicalization" -> "Canonical",      (* "Canonical" (merge isomorphic) | "None" *)
  "EventCanonicalization" -> "None",           (* "None" | "States" (=Full) | "Automatic" *)
  "EventEdgeIdentity" -> "Positional",         (* "Positional" (MultiwaySystem) | "Canonical" (IR merge) *)
  "TransitiveReduction" -> True,
  "FullCapture" -> True
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

MultiwayEvolve[rules0_, init_, steps_, OptionsPattern[]] := Module[
  {rules, canonStates, trCausal, fullCapture, evCanon, evEdge, primaryKey,
   eventCounter = 0, sidCounter = 0, freshCounter,
   statesByCanon = <||>, repEdges = <||>, sidStep = <||>, canonOf = <||>, currentStep,
   events = {}, causal = {}, branchial = {}, frontier, mkState, eStep, perStep},

  canonStates = OptionValue["StateCanonicalization"] =!= "None" && OptionValue["StateCanonicalization"] =!= False;
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

  freshCounter = If[init === {}, 1, Max[Flatten[init]] + 1];

  (* create a state node from an instance list {{contents, producerEvent}, ...}.
     With FullCapture, every output is its own node (the multiway forest of raw
     states); num_states then counts distinct canonical classes among nodes.
     Without FullCapture, isomorphic states are merged into one expanded node
     (the deduplicated multiway graph). *)
  mkState[instList_] := Module[{contents = instList[[All, 1]], cf, sid},
    cf = If[canonStates, First[CanonicalForm[contents]], contents];
    If[! fullCapture && KeyExistsQ[statesByCanon, cf],
      {statesByCanon[cf], False},
      sidCounter++; sid = sidCounter;
      repEdges[sid] = instList; sidStep[sid] = currentStep; canonOf[sid] = cf;
      If[! fullCapture, statesByCanon[cf] = sid];
      {sid, True}
    ]
  ];

  (* genesis: the single initial state (producer 0 = uncaused) *)
  currentStep = 0;
  mkState[{#, 0} & /@ init];
  frontier = {1};

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
        "eventsStates" -> Length[DeleteDuplicates[eventSigStates /@ evK]],
        "eventsAutomaticPositional" -> Length[DeleteDuplicates[eventSigAutomaticPositional /@ evK]],
        "eventsAutomaticCanonical" -> Length[DeleteDuplicates[eventSigAutomaticCanonical /@ evK]],
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
    "Events" -> events,
    "Causal" -> DeleteDuplicates[causal],
    "Branchial" -> DeleteDuplicates[branchial]
  |>
];

End[];

EndPackage[];
