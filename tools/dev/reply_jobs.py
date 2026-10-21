#!/usr/bin/env python3
"""Write the hg_evolve jobs that docs/ARCHITECTURE.md ("Where each analysis runs") and
docs/TUNING.md ("Quiet-box queue") measure, as WXF files in a directory.

    python3 tools/dev/reply_jobs.py .scratch/jobs
    hg_evolve < .scratch/jobs/wpp_d7_q_graphs.wxf > reply.wxf
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wxf_mini import encode, Sym, Fn  # noqa: E402

WPP = [[[0, 1], [0, 2]], [[0, 1], [0, 3], [1, 3], [2, 3]]]
INIT = [[[0, 1], [0, 2]]]
Q = {"CanonicalizeStates": Sym("Full"), "ExploreFromCanonicalStatesOnly": Sym("True")}
RECORDS = ["States", "Events", "CausalEdges", "BranchialEdges", "StateBitvectors", "GlobalEdges",
           "NumStates", "NumEvents", "NumCausalEdges", "NumBranchialEdges"]


def job(steps, opts):
    return encode({"InitialStates": INIT, "Rules": {"r0": Fn("Rule", WPP)}, "Steps": steps,
                   "Options": opts})


JOBS = {
    "wpp_d7_q_graphs": job(7, dict(Q, RequestedData=[],
                                   GraphProperties=["StatesGraph", "CausalGraph"])),
    "wpp_d7_q_structure": job(7, dict(Q, RequestedData=[],
                                      GraphProperties=["StatesGraphStructure",
                                                       "CausalGraphStructure"])),
    "wpp_d6_full_evograph": job(6, {"CanonicalizeStates": Sym("Full"), "RequestedData": [],
                                    "GraphProperties": ["EvolutionCausalBranchialGraph"]}),
    "wpp_d6_full_all": job(6, {"CanonicalizeStates": Sym("Full"), "RequestedData": RECORDS,
                               "IncludeCanonicalHashes": Sym("True")}),
    "wpp_d6_auto_stats_all": job(6, {"CanonicalizeStates": Sym("Automatic"),
                                     "RequestedData": ["StepStatistics"],
                                     "StepStatisticsBranchial": ["Graph", "Overlap"]}),
    "wpp_d7_q_stats": job(7, dict(Q, RequestedData=["StepStatistics"])),
    "wpp_d8_q_stats": job(8, dict(Q, RequestedData=["StepStatistics"])),
}

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
for name, b in JOBS.items():
    with open(os.path.join(out, name + ".wxf"), "wb") as f:
        f.write(b)
print(" ".join(sorted(JOBS)), flush=True)
