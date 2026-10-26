#!/usr/bin/env wolframscript
(* The parity fixtures of the hgevolve Python package: the HGEvolve calls the examples of
   docs/en/ReferencePages/Symbols/HGEvolve.md make, and the reply the paclet gets for each.

   Two steps, so that no job of unknown size runs in an engine process outside a memory cap:

     wolframscript -file generate_fixtures.wls capture DIR
         Evaluates the page's fenced blocks in order with the engine call replaced by $Failed,
         so no engine runs. Every rule-carrying job HGEvolve builds is recorded with the rules and
         initial states the call was given. Each distinct job, changed to ask for every list and
         count (and "StepStatistics" when the call did), with one thread, the canonical hashes
         and no graphs, is written to DIR/job<i>.wxf, and the inputs to DIR/inputs.json.

     wolframscript -file generate_fixtures.wls reply DIR
         Sends the jobs DIR/selected.txt names, comma-separated (each measured beforehand with
         hg_evolve under a memory cap by measure_fixture_jobs.py), to the paclet engine, and writes
         their replies with the inputs to python/tests/fixtures/HGEvolve.json, which
         `gzip -9 -n` turns into the HGEvolve.json.gz the tests read.

   The job of HGEvolve[chain, {{1, 2}}, 20, ..., "ExploreFromCanonicalStatesOnly" -> True] is not
   a fixture: with "States" and "Events" it unfolds every raw event of 20 steps, which exhausts any
   memory (measure_fixture_jobs.py reports its reduced reply under the cap).

   Run each step under flock /tmp/hg_wolfram.lock. The paclet is loaded from this repository. *)

repo = ParentDirectory[DirectoryName[$InputFileName], 2];
PacletDirectoryUnload[FileNameJoin[{repo, "paclet"}]];
PacletDirectoryLoad[FileNameJoin[{repo, "paclet"}]];
Needs["HypergraphRewriting`"];
Off[General::stop];

args = Rest[$ScriptCommandLine];
If[Length[args] < 2, Print["usage: capture DIR | reply DIR"]; Exit[2]];
{mode, dir} = args[[;; 2]];

allData = {"States", "Events", "CausalEdges", "BranchialEdges", "NumStates", "NumEvents",
           "NumCausalEdges", "NumBranchialEdges", "GlobalEdges", "StateBitvectors"};

(* The job a fixture is made from: the call's options with every list and count, StepStatistics
   when the call asked for it, one thread, the canonical hashes, and no graphs. *)
fixtureJob[job_] := Module[{o = job["Options"]},
  o["RequestedData"] = Join[allData, Intersection[o["RequestedData"], {"StepStatistics"}]];
  o["GraphProperties"] = {};
  o["Threads"] = 1;
  o["IncludeCanonicalHashes"] = True;
  o["ShowProgress"] = False;
  Append[job, "Options" -> o]];

(* JSON form: symbols as strings (True, False and None as JSON's), association keys as their
   InputForm, sparse arrays dense, reals that are not finite as strings. *)
json[a_Association] := Association[KeyValueMap[jsonKey[#1] -> json[#2] &, a]];
json[l_List] := json /@ l;
json[s_SparseArray] := json[Normal[s]];
json[x_Integer] := x;
json[x_Real] := If[NumericQ[x] && Abs[x] < Infinity, x, ToString[x, InputForm]];
json[x_Rational] := N[x];
json[s_String] := s;
json[True] := True;
json[False] := False;
json[None] := Null;
json[s_Symbol] := SymbolName[s];
json[r_Rule] := {json[r[[1]]], json[r[[2]]]};
json[other_] := ToString[other, InputForm];
jsonKey[k_String] := k;
jsonKey[k_] := ToString[k, InputForm];

(* A rule side's or a state's vertex: an integer as itself, anything else marked as a label. *)
label[v_Integer] := v;
label[v_] := <|"label" -> ToString[v, InputForm]|>;
rawRules[rules_] := Map[{Map[label, #[[1]], {2}], Map[label, #[[2]], {2}]} &, Flatten[{rules}]];
rawInitial[init_] := Map[label, init, {-1}];

If[mode === "capture",
  page = FileNameJoin[{repo, "docs", "en", "ReferencePages", "Symbols", "HGEvolve.md"}];
  blocks = Module[{lines, cur = {}, inside = False, res = {}},
    lines = StringSplit[ByteArrayToString[ReadByteArray[page], "UTF-8"], "\n", All];
    Do[If[StringStartsQ[StringTrim[l], "```"],
         If[inside, AppendTo[res, StringRiffle[cur, "\n"]]; cur = {}; inside = False, inside = True],
         If[inside, AppendTo[cur, l]]], {l, lines}];
    res];
  calls = {};
  lastInput = None;
  Internal`InheritedBlock[{HypergraphRewriting`Private`hgCheckInput,
                           HypergraphRewriting`Private`hgSendJob},
    Unprotect[HypergraphRewriting`Private`hgCheckInput, HypergraphRewriting`Private`hgSendJob];
    PrependTo[DownValues[HypergraphRewriting`Private`hgCheckInput],
      HoldPattern[HypergraphRewriting`Private`hgCheckInput[r_, i_]] /; !TrueQ[$inCheck] :>
        Block[{$inCheck = True}, lastInput = {r, i}; HypergraphRewriting`Private`hgCheckInput[r, i]]];
    (* No engine runs: every job is recorded and answered with $Failed. *)
    DownValues[HypergraphRewriting`Private`hgSendJob] = {
      HoldPattern[HypergraphRewriting`Private`hgSendJob[job_Association, ___]] :>
        (If[KeyExistsQ[job, "Rules"] && Lookup[job, "Op", "Evolve"] === "Evolve",
           AppendTo[calls, {lastInput, job}]]; $Failed)};
    Block[{$Context = "DocExamples`", $ContextPath = {"DocExamples`", "HypergraphRewriting`", "System`"}},
      Do[Quiet@Check[ToExpression[b], Null], {b, blocks}]]];
  unique = DeleteDuplicatesBy[calls, Last];
  CreateDirectory[dir, CreateIntermediateDirectories -> True];
  Do[BinaryWrite[FileNameJoin[{dir, "job" <> ToString[i] <> ".wxf"}],
                 BinarySerialize[fixtureJob[unique[[i, 2]]]]];
     Close[FileNameJoin[{dir, "job" <> ToString[i] <> ".wxf"}]], {i, Length[unique]}];
  Export[FileNameJoin[{dir, "inputs.json"}],
    Table[<|"Index" -> i,
            "Rules" -> If[unique[[i, 1]] === None, Null, rawRules[unique[[i, 1, 1]]]],
            "Initial" -> If[unique[[i, 1]] === None, Null, rawInitial[unique[[i, 1, 2]]]]|>,
          {i, Length[unique]}], "RawJSON", "Compact" -> True];
  Print["calls ", Length[calls], ", distinct jobs ", Length[unique]];
  Exit[0]];

If[mode === "reply",
  wanted = ToExpression /@ StringSplit[StringTrim[Import[FileNameJoin[{dir, "selected.txt"}], "Text"]], ","];
  inputs = Association[#["Index"] -> # & /@ Import[FileNameJoin[{dir, "inputs.json"}], "RawJSON"]];
  fixtures = Table[
    Module[{job, reply},
      job = BinaryDeserialize[ReadByteArray[FileNameJoin[{dir, "job" <> ToString[i] <> ".wxf"}]]];
      reply = HypergraphRewriting`Private`hgSendJob[job, "CPU", False];
      If[!AssociationQ[reply], Print["no reply for job ", i]; Nothing,
        <|"Index" -> i,
          "Rules" -> inputs[i]["Rules"],
          "Initial" -> inputs[i]["Initial"],
          "Job" -> json[KeyDrop[job, {"Rules"}]],
          "JobRules" -> json[Values[job["Rules"]]],
          "ReplyBytes" -> HypergraphRewriting`Private`$hgLastReplyBytes,
          "Reply" -> json[Join[reply, <|
            "States" -> Values[Lookup[reply, "States", <||>]],
            "Events" -> Values[Lookup[reply, "Events", <||>]],
            "StateBitvectors" -> KeyValueMap[<|"State" -> #1, "Edges" -> #2|> &,
                                             Lookup[reply, "StateBitvectors", <||>]]|>]]|>]],
    {i, wanted}];
  out = FileNameJoin[{repo, "python", "tests", "fixtures", "HGEvolve.json"}];
  Export[out, <|"Page" -> "HGEvolve", "Fixtures" -> fixtures|>, "RawJSON", "Compact" -> True];
  Print["wrote ", Length[fixtures], " fixtures to ", out];
  Exit[0]];

Print["unknown mode ", mode]; Exit[2];
