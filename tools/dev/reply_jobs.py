#!/usr/bin/env python3
"""Write the hg_evolve jobs that docs/ARCHITECTURE.md ("Where each analysis runs") and
docs/TUNING.md ("Quiet-box queue") measure, and the wpp depth 8 quotient jobs whose peak RSS
the explored-branchial store is measured on, as WXF files in a directory. The jobs are encoded by
the hgevolve package (python/), so it must be built (HG_BUILD_PYTHON) or installed.

    PYTHONPATH=python python3 tools/dev/reply_jobs.py .scratch/jobs
    hg_evolve < .scratch/jobs/wpp_d7_q_graphs.wxf > reply.wxf
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
import hgevolve  # noqa: E402

WPP = ([[0, 1], [0, 2]], [[0, 1], [0, 3], [1, 3], [2, 3]])
INIT = [[0, 1], [0, 2]]
Q = {"canonicalize_states": "Full", "explore_from_canonical_states_only": True}
RECORDS = ["states", "events", "causal_edges", "branchial_edges", "state_bitvectors",
           "global_edges", "num_states", "num_events", "num_causal_edges", "num_branchial_edges"]
COUNTS = ["num_states", "num_events", "num_causal_edges", "num_branchial_edges"]


def job(steps, data, graphs=None, **options):
    # Seed 0 and every branchial step: the engine's defaults, which these measurements use.
    return hgevolve.encode_job(WPP, INIT, steps, data=data, graph_properties=graphs,
                               random_seed=0, branchial_step="all", **options)


JOBS = {
    "wpp_d7_q_graphs": job(7, [], ["StatesGraph", "CausalGraph"], **Q),
    "wpp_d7_q_structure": job(7, [], ["StatesGraphStructure", "CausalGraphStructure"], **Q),
    "wpp_d6_full_evograph": job(6, [], ["EvolutionCausalBranchialGraph"],
                                canonicalize_states="Full"),
    "wpp_d6_full_all": job(6, RECORDS, canonicalize_states="Full", include_canonical_hashes=True),
    "wpp_d6_auto_stats_all": job(6, ["step_statistics"], canonicalize_states="Automatic",
                                 step_statistics_branchial=["Graph", "Overlap"]),
    "wpp_d7_q_stats": job(7, ["step_statistics"], **Q),
    "wpp_d7_q_stats_geodesic": job(7, ["step_statistics"],
                                   step_statistics_measures=["GeodesicAnalysis"], **Q),
    "wpp_d6_none_curv": job(6, ["states"], canonicalize_states="None",
                            include_vertex_curvature=True),
    "wpp_d7_full_curv": job(7, ["states"], canonicalize_states="Full",
                            include_vertex_curvature=True),
    "wpp_d8_q_stats": job(8, ["step_statistics"], **Q),
    "wpp_d8_q_counts": job(8, COUNTS, **Q),
    "wpp_d8_q_branchial": job(8, ["num_branchial_edges"], ["BranchialGraphStructure"], **Q),
}

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
for name, b in JOBS.items():
    with open(os.path.join(out, name + ".wxf"), "wb") as f:
        f.write(b)
print(" ".join(sorted(JOBS)), flush=True)
