(* hgNoSocketCheck[pacletDir]: loads the paclet in pacletDir, runs HGEvolve and every session verb
   (HGSessionOpen, HGSessionStep, HGSessionQuery, HGSessionFrontier, HGSessionClose) on the CPU
   and on the GPU, and compares Sockets[] before and after. Prints NOSOCK_OK when no call opened a
   socket and every call returned its usual result, otherwise NOSOCK_FAIL with the calls that
   opened a socket and the sockets they left, or the calls that failed.
   The paclet's engine worker opens no socket: on Windows, Wolfram 15.0 faults inside SocketConnect
   (libzeromqlink.dll) in about 1 of 6 fresh kernels on
   Windows. PacletTest.NoCallOpensASocket runs this on the repository's paclet. A GPU call on a
   machine without a usable GPU runs on the CPU with HGEvolve::gpudev, which is quieted here. *)
hgNoSocketCheck[pacletDir_String] := Module[
  {rule = {{1, 2}, {2, 3}} -> {{3, 2}, {2, 1}, {1, 4}}, init = {{1, 2}, {2, 3}},
   opts = Sequence["Tuning" -> <|"MemoryLimit" -> 3 2^30|>], before, opened = {}, results = {}, failed,
   check, s},
  PacletDirectoryLoad[pacletDir];
  Needs["HypergraphRewriting`"];
  If[!MemberQ[$Packages, "HypergraphRewriting`"], Print["NOSOCK_LOAD_FAIL"]; Return[$Failed]];
  check[label_, expr_] := Module[{r, new},
    before = Sockets[];
    r = Quiet[expr, {HypergraphRewriting`HGEvolve::gpudev}];
    new = Complement[Sockets[], before];
    AppendTo[results, label -> Head[r]];
    If[new =!= {}, AppendTo[opened, label -> new]];
    r];
  SetAttributes[check, HoldRest];
  Do[
    check[{"HGEvolve", dev},
      HypergraphRewriting`HGEvolve[rule, init, 3, "NumStates", "TargetDevice" -> dev, opts]];
    s = check[{"HGSessionOpen", dev},
      HypergraphRewriting`HGSessionOpen[rule, init, "NumStates", "TargetDevice" -> dev, opts]];
    check[{"HGSessionStep", dev}, HypergraphRewriting`HGSessionStep[s, 2]];
    check[{"HGSessionQuery", dev}, HypergraphRewriting`HGSessionQuery[s]];
    check[{"HGSessionFrontier", dev}, HypergraphRewriting`HGSessionFrontier[s]];
    check[{"HGSessionClose", dev}, HypergraphRewriting`HGSessionClose[s]],
    {dev, {"CPU", "GPU"}}];
  (* Every call has to have run: a session verb that failed opens no socket either. *)
  failed = Select[results, !MatchQ[#, ({"HGEvolve", _} -> Integer) | ({"HGSessionOpen", _} ->
    HypergraphRewriting`HGSessionObject) | ({"HGSessionStep" | "HGSessionQuery", _} -> Integer) |
    ({"HGSessionFrontier", _} -> List) | ({"HGSessionClose", _} -> _)] &];
  Which[
    opened =!= {}, Print["NOSOCK_FAIL opened ", opened, " results ", results],
    failed =!= {}, Print["NOSOCK_FAIL calls failed ", failed],
    True, Print["NOSOCK_OK ", results]]
]
