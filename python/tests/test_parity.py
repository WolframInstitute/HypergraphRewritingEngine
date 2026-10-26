"""Parity with HGEvolve: every job the examples of docs/en/ReferencePages/Symbols/HGEvolve.md send
(fixtures/HGEvolve.json.gz, written by generate_fixtures.wls and gzip) gives the same reply through
hgevolve.evolve.

Each fixture is run from the rules and initial states the example gave HGEvolve, with the options
HGEvolve sent, so the label numbering and the option names are tested as well as the engine. The
reply is compared exactly (ids and all) and, where the ids differ, up to the documented id
conventions (docs/SPEC.md): state and event ids are run-local, so states are compared by their
"CanonicalHash" and events, causal and branchial pairs through the states they join."""

import collections
import gzip
import json
import math
import os
import re
import warnings

import numpy as np
import pytest

import hgevolve as hg
from hgevolve._options import DATA, OPTIONS

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "HGEvolve.json.gz")

with gzip.open(FIXTURES, "rt") as f:
    _ALL = json.load(f)["Fixtures"]

WIRE_TO_OPTION = {wire: name for name, (wire, _) in OPTIONS.items()}
WIRE_TO_DATA = {wire: name for name, wire in DATA.items()}
# Options HGEvolve sends that the Python package has no use for: graph building only.
GRAPH_ONLY = {"GraphProperties", "EdgeDeduplication"}


def _from_wire(name, v):
    """The Python value of HGEvolve's option value `v` (in fixture JSON form)."""
    if name in ("canonicalize_states", "canonicalize_events"):
        return None if v in (None, "None") else v
    if name == "event_ordering_function":
        return None if v == [] else v
    if name in ("step_statistics_measures", "step_statistics_branchial", "step_statistics_matrices"):
        return None if v == [] else v
    if name == "step_statistics_matrix_limit":
        return None if v == 0 else v
    if name == "state_cost":
        return None if not v.get("Code") else v
    return v


def python_call(fx):
    job = fx["Job"]
    options, data = {}, []
    for wire, v in job["Options"].items():
        if wire in GRAPH_ONLY:
            continue
        if wire == "RequestedData":
            data = [WIRE_TO_DATA[d] for d in v]
            continue
        name = WIRE_TO_OPTION[wire]
        options[name] = _from_wire(name, v)
    return data, options


def _label(v):
    return v["label"] if isinstance(v, dict) else v


def call_inputs(fx):
    """The rules and initial states as the example gave them to HGEvolve, labels as strings; the
    job's own when the example's input was not recorded."""
    if fx["Rules"] is not None:
        rules = [([[_label(v) for v in e] for e in lhs], [[_label(v) for v in e] for e in rhs])
                 for lhs, rhs in fx["Rules"]]
        initial = [[[_label(v) for v in e] for e in s] for s in fx["Initial"]] \
            if isinstance(fx["Initial"][0][0], list) else \
            [[_label(v) for v in e] for e in fx["Initial"]]
        return rules, initial
    return [tuple(r) for r in fx["JobRules"]], fx["Job"]["InitialStates"]


def run(fx, **overrides):
    rules, initial = call_inputs(fx)
    data, options = python_call(fx)
    options.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", hg.EngineWarning)
        return hg.evolve(rules, initial, fx["Job"]["Steps"], data=data, **options)


# -------------------------------------------------------------------------------------------
# Comparison


def _key(k):
    """An association key in fixture form (generate_fixtures.wls jsonKey)."""
    if isinstance(k, str):
        return k
    if isinstance(k, bool):
        return "True" if k else "False"
    if isinstance(k, int):
        return str(k)
    if isinstance(k, tuple):
        return "{" + ", ".join(_key(x) for x in k) + "}"
    return str(k)


def jsonish(x):
    """A Python reply value in fixture JSON form."""
    if isinstance(x, dict):
        return {_key(k): jsonish(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return jsonish(x.tolist())
    if isinstance(x, (list, tuple)):
        return [jsonish(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return {math.inf: "Infinity", -math.inf: "-Infinity"}.get(x, "Indeterminate")
    return x


_REAL = re.compile(r"-?\d+\.\d*(`\d*\.?\d*)?(\*\^-?\d+)?|-?\d+\.\d+(e-?\d+)?")


def _numeric_keys(d):
    """[(key as a float, value)] when some key is a real number written as text (WL InputForm
    "2." or Python "2.0") and every key is a number; else None."""
    out, real = [], False
    for k, v in d.items():
        if isinstance(k, str) and _REAL.fullmatch(k):
            m = k.split("`")[0].replace("*^", "e")
            out.append((float(m + ("0" if m.endswith(".") else "")), v))
            real = True
        elif isinstance(k, str) and re.fullmatch(r"-?\d+", k):
            out.append((float(k), v))
        else:
            return None
    return out if real else None


def same(a, b, path="", rel=1e-9):
    """None when `a` (fixture) and `b` (Python) agree, reals to `rel`; else where they differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        na, nb = _numeric_keys(a), _numeric_keys(b)
        if na is not None and nb is not None:
            # Real keys (a histogram's bins), compared as numbers in order.
            if len(na) != len(nb):
                return f"{path}: {len(na)} keys vs {len(nb)}"
            for (ka, va), (kb, vb) in zip(sorted(na), sorted(nb)):
                d = same(ka, kb, f"{path}<key>", rel) or same(va, vb, f"{path}/{ka}", rel)
                if d:
                    return d
            return None
        if set(a) != set(b):
            return f"{path}: keys {sorted(set(a) ^ set(b))[:5]}"
        for k in a:
            d = same(a[k], b[k], f"{path}/{k}", rel)
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = same(x, y, f"{path}[{i}]", rel)
            if d:
                return d
        return None
    if isinstance(a, bool) or isinstance(b, bool):
        return None if a == b else f"{path}: {a!r} vs {b!r}"
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b or math.isclose(a, b, rel_tol=rel, abs_tol=1e-12):
            return None
        return f"{path}: {a!r} vs {b!r}"
    return None if a == b else f"{path}: {a!r} vs {b!r}"


def exact_lists(fx_reply, r):
    """The reply's lists in fixture form, from the Python result."""
    out = {}
    if r.states is not None:
        out["States"] = r.states.records()
    if r.events is not None:
        out["Events"] = r.events.records()
    if r.causal_edges is not None:
        out["CausalEdges"] = r.causal_edges.records()
    if r.branchial_edges is not None:
        out["BranchialEdges"] = r.branchial_edges.records()
    if r.global_edges is not None:
        out["GlobalEdges"] = [[rec["Id"]] + rec["Vertices"] for rec in r.global_edges.records()]
    if r.state_bitvectors is not None:
        out["StateBitvectors"] = r.state_bitvectors.records()
    return out


def _sorted(records, key):
    return sorted(records, key=lambda x: x[key])


def invariant_view(reply):
    """The reply up to run-local ids: multisets keyed by canonical hashes."""
    states = reply.get("States", [])
    h = {s["Id"]: s["CanonicalHash"] for s in states}
    sv = collections.Counter(
        (s["CanonicalHash"], s["Step"], s["IsInitial"], len(s["Edges"]),
         h.get(s["CanonicalId"]), h.get(s["ContentStateId"])) for s in states)
    events = reply.get("Events", [])
    sig = {}
    for e in events:
        sig[e["Id"]] = (e["RuleIndex"], h.get(e["InputState"]), h.get(e["OutputState"]),
                        h.get(e["CanonicalInputState"]), h.get(e["CanonicalOutputState"]),
                        len(e.get("ConsumedEdges", [])), len(e.get("ProducedEdges", [])))
    ev = collections.Counter(sig.values())
    canon = collections.Counter(sig.get(e["CanonicalId"]) for e in events)
    causal = collections.Counter((sig.get(c["From"]), sig.get(c["To"]))
                                 for c in reply.get("CausalEdges", []))
    branchial = collections.Counter(frozenset([sig.get(c["From"]), sig.get(c["To"])])
                                    for c in reply.get("BranchialEdges", []))
    glob = collections.Counter(len(e) for e in reply.get("GlobalEdges", []))
    bits = collections.Counter((h.get(b["State"]), len(b["Edges"]))
                               for b in reply.get("StateBitvectors", []))
    return {"states": sv, "events": ev, "canonical_events": canon, "causal": causal,
            "branchial": branchial, "global_edges": glob, "bitvectors": bits}


COUNTS = ["NumStates", "NumEvents", "NumCausalEdges", "NumBranchialEdges"]


def compare(fx, r):
    """("exact" | "invariant", None) when the result matches the fixture, else a message."""
    want = fx["Reply"]
    for k in COUNTS:
        if want.get(k) != r.values.get(k):
            return None, f"{k}: {want.get(k)} vs {r.values.get(k)}"
    if "StepStatistics" in want:
        d = same(want["StepStatistics"], jsonish(r.step_statistics), "StepStatistics")
        if d:
            return None, d
    got = exact_lists(want, r)
    exact = True
    for k, v in got.items():
        w = want.get(k, [])
        if k in ("States", "Events"):
            w, v = _sorted(w, "Id"), _sorted(v, "Id")
        if same(w, v, k):
            exact = False
    if exact:
        return "exact", None
    a = invariant_view(want)
    b = invariant_view(got)
    for k in a:
        if a[k] != b[k]:
            return None, f"{k} differ up to ids"
    return "invariant", None


@pytest.mark.parametrize("fx", _ALL, ids=[f"job{fx['Index']}" for fx in _ALL])
def test_parity(fx):
    r = run(fx)
    kind, message = compare(fx, r)
    assert message is None, message


def test_fixture_count():
    # Every example job of the page is a fixture; the page has well over a hundred calls.
    assert len(_ALL) >= 50


def test_state_cost_expression():
    # The page's "StateCost" -> (Abs[#BallDimension - 1.5] &) example, given as Python.
    fx = [f for f in _ALL if f["Job"]["Options"].get("StateCost", {}).get("Code")]
    assert fx, "no StateCost example among the fixtures"
    for f in fx:
        program = f["Job"]["Options"]["StateCost"]
        if "BallDimension" not in program["Code"]:
            continue
        r = run(f, state_cost="abs(BallDimension - 1.5)")
        kind, message = compare(f, r)
        assert message is None, message
