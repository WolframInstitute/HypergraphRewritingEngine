"""hgevolve: multiway hypergraph rewriting with the HypergraphRewriting engine, in-process.

    import hgevolve as hg
    r = hg.evolve([([[1, 2]], [[1, 2], [2, 3]])], [[1, 2]], 3)
    r.num_states, r.states.step, r.events.consumed_edges, r.causal_edges.array

evolve() takes HGEvolve's arguments and options (docs/API.md lists them under both names) and
returns a Result whose arrays are NumPy views of the engine's reply. The engine runs in this
process: a crash of the engine ends the Python process. Run work that may crash, or work in
parallel, in worker processes (examples/parallel_sweep.py).
"""

import warnings as _warnings

from . import _core
from ._options import DATA, DEFAULT_DATA, OPTIONS, Function, Symbol, initial_states, \
    job_options, rules_association
from ._result import EdgeLists, Ragged, Relation, Result, Table, VertexReals
from ._cost import state_cost_program

__all__ = ["evolve", "Session", "encode_job", "Result", "Table", "Relation", "Ragged",
           "EdgeLists", "VertexReals", "EngineError", "EngineWarning", "OPTIONS", "DATA",
           "DEFAULT_DATA", "state_cost_program", "build_stamp"]


class EngineError(RuntimeError):
    """The engine refused the job (HGEvolve's message for the same job)."""


class EngineWarning(UserWarning):
    """An entry of the reply's "Warnings" (HGEvolve::warn and HGEvolve::overflow)."""


def build_stamp():
    """The engine build's stamp: commit, variant and configuration."""
    return _core.build_stamp()


def _job(rules, initial, steps, data, options):
    if isinstance(steps, bool) or int(steps) != steps or steps < 0:
        raise ValueError(f"steps must be a non-negative integer, got {steps!r}")
    return {"InitialStates": initial_states(initial), "Rules": rules_association(rules),
            "Steps": int(steps), "Options": job_options(options, data)}


def _run(job):
    try:
        reply = _core.run(job)
    except RuntimeError as e:
        raise EngineError(str(e)) from None
    result = Result(reply)
    for w in result.warnings:
        _warnings.warn(f"{w.get('Kind')}: {w.get('Context')}", EngineWarning, stacklevel=3)
    return result


def evolve(rules, initial, steps, *, data=None, **options):
    """HGEvolve[rules, initial, steps] in this process.

    rules    a (lhs, rhs) pair or a list of them; a side is a list of edges, an edge a list of
             vertex labels. Labels are numbered as HGEvolve numbers them.
    initial  a hypergraph (a list of edges) or a list of hypergraphs.
    steps    the number of steps.
    data     what the reply holds: a list of names from hgevolve.DATA (default DEFAULT_DATA).
    options  HGEvolve's options under the names in hgevolve.OPTIONS.
    """
    return _run(_job(rules, initial, steps, data, options))


def encode_job(rules, initial, steps, *, data=None, graph_properties=None, **options):
    """The WXF bytes of the job evolve() runs, as hg_evolve reads them on stdin.
    `graph_properties` (HGEvolve's graph property names) adds the "GraphData" the paclet builds
    its graphs from to hg_evolve's reply; evolve() builds no graphs."""
    job = _job(rules, initial, steps, data, options)
    if graph_properties:
        job["Options"]["GraphProperties"] = [str(p) for p in graph_properties]
    return _core.encode(job)


class Session:
    """HGSessionOpen: an evolution that can be continued. One session is open per process.

        with hg.Session(rules, initial, canonicalize_states="Full") as s:
            r1 = s.step(2)
            r2 = s.step(1, start=r1.frontier[:1])
            r3 = s.query(data=["num_states"])

    The identity options (canonicalize_states, canonicalize_events, show_genesis_events,
    explore_from_canonical_states_only) are fixed when the session opens. Arrays of a Result stay
    valid after the session is closed."""

    def __init__(self, rules, initial, *, data=None, **options):
        self._options = dict(options)
        self._options.setdefault("random_seed", None)
        self._options["random_seed"] = OPTIONS["random_seed"][1](self._options["random_seed"],
                                                                 "random_seed")
        job = _job(rules, initial, 0, data, self._options)
        job["Op"] = "Open"
        self.result = _run(job)
        self.handle = self.result.session
        self.frontier = self.result.frontier

    def _verb(self, op, steps, start, data):
        if self.handle is None:
            raise EngineError("the session is closed")
        job = {"Steps": int(steps), "Op": op, "Session": self.handle,
               "Options": job_options(self._options, data)}
        if start is not None:
            job["From"] = [int(x) for x in start]
        result = _run(job)
        self.frontier = result.frontier
        return result

    def step(self, steps=1, *, start=None, data=None):
        """HGSessionStep: `steps` more steps from the frontier, or from the frontier states
        `start` names (the others stay on the frontier)."""
        return self._verb("Step", steps, start, data)

    def query(self, *, data=None):
        """HGSessionQuery: the session's graph, explored no further."""
        return self._verb("Query", 0, None, data)

    def close(self):
        """HGSessionClose. Closing a closed session does nothing."""
        if self.handle is None:
            return
        handle, self.handle = self.handle, None
        try:
            _core.run({"Op": "Close", "Session": handle, "Options": {"Op": "Close"}})
        except RuntimeError as e:
            raise EngineError(str(e)) from None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
