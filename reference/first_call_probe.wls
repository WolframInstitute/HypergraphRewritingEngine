(* One fresh kernel's first engine call, for tools/dev/first_call_gate.sh.
   Usage: first_call_probe.wls PACLETDIR MARKERFILE DEVICE
   Writes "start <kernel process id>" to MARKERFILE, makes the kernel's first HGEvolve call on
   DEVICE ("CPU" or "GPU": Wolfram's rule from {{0,0},{0,0}}, 3 steps, "MemoryLimit" 3 GB), and
   then writes "returned <result>" and "messages <names>". A marker with "start" and no "returned"
   is a kernel that ended during its first call. *)
PacletDirectoryLoad[$ScriptCommandLine[[2]]];
marker = $ScriptCommandLine[[3]];
device = $ScriptCommandLine[[4]];
Needs["HypergraphRewriting`"];
Export[marker, "start " <> ToString[$ProcessID], "Text"];
d = EvaluationData[HGEvolve[{{x, y}, {x, z}} -> {{x, y}, {x, w}, {y, w}, {z, w}}, {{0, 0}, {0, 0}},
  3, "NumStates", "TargetDevice" -> device, "Tuning" -> <|"MemoryLimit" -> 3 2^30|>]];
Export[marker, "start " <> ToString[$ProcessID] <> "\nreturned " <> ToString[d["Result"]] <>
  "\nmessages " <> ToString[d["MessagesText"]],
  "Text"];
Exit[0];
