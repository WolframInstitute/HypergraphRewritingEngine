#!/usr/bin/env wolframscript
(* Smoke-test the ASSEMBLED .paclet exactly as a user receives it, and verify CPU/GPU
   parity across every property. Install the archive, load the installed paclet, then:
     - CPU/GPU NumStates parity across CanonicalizeStates None/Full (modes are distinct)
     - CPU/GPU structural parity for EVERY graph property (GraphData): same vertex and
       edge counts and isomorphic graphs -- and neither returns $Failed
     - CPU/GPU raw-count parity (NumStates/NumEvents/NumCausalEdges/NumBranchialEdges)
     - the fallback DLL loads and runs (performHausdorffAnalysis)
   Argv[1] is the .paclet path (a Windows path under the Windows kernel). *)

pacletFile = $ScriptCommandLine[[2]];
Print["==> installing ", pacletFile];
inst = PacletInstall[pacletFile, ForceVersionInstall -> True];
If[FailureQ[inst], Print["INSTALL FAILED"]; Exit[2]];
Needs["HypergraphRewriting`"];
Print["==> loaded: ", PacletObject["WolframInstitute/HypergraphRewriteEngine"]["Location"]];

rule  = {{1, 2}, {1, 3}} -> {{1, 2}, {1, 3}, {2, 3}};
init  = {{1, 2}, {1, 3}};
steps = 3;

ok = True;
check[label_, cond_] := (Print[If[TrueQ[cond], "PASS  ", "FAIL  "], label]; ok = ok && TrueQ[cond]);

(* ---- counts across modes ---- *)
ns[dev_, mode_] := HGEvolve[rule, init, steps, "NumStates", "CanonicalizeStates" -> mode, "TargetDevice" -> dev];
cpuNone = ns["CPU", None]; cpuFull = ns["CPU", Full];
gpuNone = ns["GPU", None]; gpuFull = ns["GPU", Full];
Print["==> NumStates  CPU None=", cpuNone, " Full=", cpuFull, "   GPU None=", gpuNone, " Full=", gpuFull];
check["CPU modes distinct (None > Full)", IntegerQ[cpuNone] && IntegerQ[cpuFull] && cpuNone > cpuFull];
check["GPU None == CPU None", gpuNone === cpuNone];
check["GPU Full == CPU Full", gpuFull === cpuFull];
check["GPU honors modes (None > Full)", IntegerQ[gpuNone] && IntegerQ[gpuFull] && gpuNone > gpuFull];

(* ---- raw counts CPU vs GPU (Full) ---- *)
counts[dev_] := HGEvolve[rule, init, steps, "Debug", "CanonicalizeStates" -> Full, "TargetDevice" -> dev];
cc = counts["CPU"]; cg = counts["GPU"];
Print["==> Debug counts  CPU=", cc, "  GPU=", cg];
check["CPU/GPU Debug counts identical", cc === cg];

(* ---- structural graph parity for EVERY graph property ---- *)
graphProps = {"StatesGraph", "CausalGraph", "BranchialGraph", "EvolutionGraph",
   "EvolutionCausalGraph", "EvolutionBranchialGraph", "EvolutionCausalBranchialGraph",
   "StatesGraphStructure", "CausalGraphStructure", "BranchialGraphStructure"};

(* IsomorphicGraphQ returns unevaluated on multigraphs (StatesGraph / Branchial / Evolution
   graphs have multi-edges), so compare multigraph-safe structural invariants: vertex count,
   edge count, and the sorted in/out degree sequences (which count edge multiplicity). *)
degSeq[g_] := {Sort[VertexInDegree[g]], Sort[VertexOutDegree[g]]};
cmp[gc_, gg_] := Which[
  ! GraphQ[gc], "CPU not a Graph (" <> ToString[Head[gc]] <> ")",
  ! GraphQ[gg], "GPU not a Graph (" <> ToString[Head[gg]] <> ")",
  VertexCount[gc] =!= VertexCount[gg],
    "vertex count " <> ToString[VertexCount[gc]] <> " vs " <> ToString[VertexCount[gg]],
  EdgeCount[gc] =!= EdgeCount[gg],
    "edge count " <> ToString[EdgeCount[gc]] <> " vs " <> ToString[EdgeCount[gg]],
  degSeq[gc] =!= degSeq[gg], "degree sequence differs",
  True, "OK"];

Print["==> graph-property parity (CPU vs GPU, CanonicalizeStates -> Full):"];
Do[
  Module[{gc, gg, res},
    gc = Quiet@Check[HGEvolve[rule, init, steps, p, "CanonicalizeStates" -> Full, "TargetDevice" -> "CPU"], $Failed];
    gg = Quiet@Check[HGEvolve[rule, init, steps, p, "CanonicalizeStates" -> Full, "TargetDevice" -> "GPU"], $Failed];
    res = cmp[gc, gg];
    check[StringPadRight[p, 32] <> If[GraphQ[gc], "V=" <> ToString[VertexCount[gc]] <> " E=" <> ToString[EdgeCount[gc]] <> "  ", ""] <> res, res === "OK"]
  ],
  {p, graphProps}];

(* ---- fallback DLL path ---- *)
haus = HGHausdorffAnalysis[{{1, 2}, {2, 3}, {3, 4}, {4, 1}, {1, 3}, {2, 4}}];
check["fallback DLL loaded + ran (performHausdorffAnalysis)", AssociationQ[haus] && Length[haus] >= 3];

Print[If[ok, "\nSMOKE TEST PASSED", "\nSMOKE TEST FAILED"]];
Exit[If[ok, 0, 1]];
