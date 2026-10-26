"""The options of evolve() and Session, each named after its HGEvolve option, and the job envelope
they make: the association HGEvolve sends to the engine (hgJobOptions in
paclet/Kernel/HypergraphRewriting.wl), as Python objects."""

import random

from . import _core
from ._cost import state_cost_program


class Symbol(str):
    """A Wolfram symbol in a job (None, Full, Automatic, True, ...)."""


class Function:
    """head[args...] in a job. Rules are Function("Rule", [lhs, rhs])."""

    def __init__(self, head, args):
        self.head, self.args = head, list(args)


_core.register_types(Symbol, Function)


def _enum(*names):
    def convert(value, option):
        if value is None:
            value = "None"
        if value not in names:
            raise ValueError(f"{option} must be one of {', '.join(map(repr, names))}, got {value!r}")
        return Symbol(value)
    return convert


def _bool(value, option):
    if not isinstance(value, bool):
        raise TypeError(f"{option} must be True or False, got {value!r}")
    return value


def _count(value, option):
    if isinstance(value, bool) or int(value) != value or value < 0:
        raise ValueError(f"{option} must be a non-negative integer, got {value!r}")
    return int(value)


def _real(value, option):
    return float(value)


def _reals(value, option):
    return [float(x) for x in value]


def _seed(value, option):
    if value is None:
        return random.randint(0, 2 ** 62)
    return int(value)


def _threads(value, option):
    return 0 if value is None else _count(value, option)


def _tuning(value, option):
    return {str(k): int(v) for k, v in dict(value).items()}


def _names(all_names):
    def convert(value, option):
        if value is None:
            return []
        if isinstance(value, str):
            if value == "all":
                return list(all_names())
            return [value]
        return [str(x) for x in value]
    return convert


def _canonicalize_events(value, option):
    if value is None or value in ("None", "Full", "Automatic"):
        return Symbol("None" if value is None else value)
    if value == "Positional":
        return "Positional"
    if isinstance(value, str):
        raise ValueError(f"{option} must be None, 'Full', 'Automatic', 'Positional' or a list of "
                         f"components, got {value!r}")
    return [str(x) for x in value]


def _event_ordering(value, option):
    if value is None:
        return []
    if value == "Automatic":
        return Symbol("Automatic")
    if isinstance(value, str):
        return [value]
    return [str(x) for x in value]


def _branchial_step(value, option):
    if value == "all":
        return 0
    return int(value)


def _matrix_limit(value, option):
    return 0 if value is None else _count(value, option)


# Python name -> (HGEvolve option, conversion of the value). The order is the documentation's.
OPTIONS = {
    "canonicalize_states": ("CanonicalizeStates", _enum("None", "Automatic", "Full")),
    "canonicalize_events": ("CanonicalizeEvents", _canonicalize_events),
    "causal_transitive_reduction": ("CausalTransitiveReduction", _bool),
    "show_genesis_events": ("ShowGenesisEvents", _bool),
    "explore_from_canonical_states_only": ("ExploreFromCanonicalStatesOnly", _bool),
    "undirected_edges": ("UndirectedEdges", _bool),
    "generational": ("Generational", _bool),
    "event_ordering_function": ("EventOrderingFunction", _event_ordering),
    "max_successor_states_per_parent": ("MaxSuccessorStatesPerParent", _count),
    "max_states_per_step": ("MaxStatesPerStep", _count),
    "matches_per_state_rule": ("MatchesPerStateRule", _count),
    "uniform_random": ("UniformRandom", _bool),
    "matches_per_step": ("MatchesPerStep", _count),
    "exploration_probability": ("ExplorationProbability", _real),
    "transition_rate": ("TransitionRate", _real),
    "rule_weights": ("RuleWeights", _reals),
    "state_cost": ("StateCost", state_cost_program),
    "random_seed": ("RandomSeed", _seed),
    "branchial_step": ("BranchialStep", _branchial_step),
    "include_canonical_hashes": ("IncludeCanonicalHashes", _bool),
    "include_vertex_curvature": ("IncludeVertexCurvature", _bool),
    "step_statistics_weighting": ("StepStatisticsWeighting", _enum("States", "Classes")),
    "statistics_graph": ("StatisticsGraph", _enum("Consecutive", "AllPairs")),
    "step_statistics_measures": ("StepStatisticsMeasures", _names(_core.measure_names)),
    "step_statistics_branchial": ("StepStatisticsBranchial", _names(lambda: ["Graph", "Overlap"])),
    "step_statistics_matrices": ("StepStatisticsMatrices", _names(_core.matrix_names)),
    "step_statistics_matrix_limit": ("StepStatisticsMatrixLimit", _matrix_limit),
    "threads": ("Threads", _threads),
    "tuning": ("Tuning", _tuning),
    "show_progress": ("ShowProgress", _bool),
}

# Options HGEvolve always sends, with the value it sends when the caller gives none: HGEvolve's
# RandomSeed -> Automatic draws a seed, and BranchialStep -> Automatic is -1 for every property
# but the evolution graphs.
ALWAYS = {"random_seed": None, "branchial_step": -1}

# Python name -> HGEvolve "RequestedData" name.
DATA = {
    "states": "States",
    "events": "Events",
    "events_minimal": "EventsMinimal",
    "causal_edges": "CausalEdges",
    "branchial_edges": "BranchialEdges",
    "branchial_state_edges": "BranchialStateEdges",
    "branchial_state_edges_all_siblings": "BranchialStateEdgesAllSiblings",
    "num_states": "NumStates",
    "num_events": "NumEvents",
    "num_causal_edges": "NumCausalEdges",
    "num_branchial_edges": "NumBranchialEdges",
    "step_statistics": "StepStatistics",
    "global_edges": "GlobalEdges",
    "state_bitvectors": "StateBitvectors",
    "device": "Device",
}

DEFAULT_DATA = ("states", "events", "causal_edges", "branchial_edges", "num_states", "num_events",
                "num_causal_edges", "num_branchial_edges")


def requested_data(data):
    if data is None:
        data = DEFAULT_DATA
    elif isinstance(data, str):
        data = [data]
    out = []
    for name in data:
        if name not in DATA:
            raise ValueError(f"{name!r} is not one of {', '.join(DATA)}")
        out.append(DATA[name])
    return out


def job_options(options, data):
    """The "Options" association of a job: every option given, the ALWAYS ones, and the data."""
    unknown = sorted(set(options) - set(OPTIONS))
    if unknown:
        raise TypeError(f"unknown option(s) {', '.join(unknown)}; the options are "
                        f"{', '.join(OPTIONS)}")
    given = dict(ALWAYS)
    given.update(options)
    out = {}
    for name, (wire, convert) in OPTIONS.items():
        if name in given:
            out[wire] = convert(given[name], name)
    out["RequestedData"] = requested_data(data)
    return out


def _is_edge(e):
    return isinstance(e, (list, tuple)) and len(e) > 0 and \
        all(not isinstance(v, (list, tuple)) for v in e)


def _as_list(x):
    return x.tolist() if hasattr(x, "tolist") else x


def rules_association(rules):
    """{"Rule1": Rule[lhs, rhs], ...}, each rule's labels numbered from 0 by first appearance in
    the left side and then the right side, as HGEvolve's normalizeRule numbers them."""
    rules = _as_list(rules)
    if isinstance(rules, tuple) and len(rules) == 2 and all(
            isinstance(side, (list, tuple)) and all(_is_edge(_as_list(e)) for e in side)
            for side in rules):
        rules = [rules]
    out = {}
    for i, rule in enumerate(rules):
        if not (isinstance(rule, (list, tuple)) and len(rule) == 2):
            raise ValueError(f"rule {i} is not a (lhs, rhs) pair: {rule!r}")
        lhs, rhs = (_as_list(side) for side in rule)
        if not lhs or not all(_is_edge(_as_list(e)) for e in lhs) or \
                not all(_is_edge(_as_list(e)) for e in rhs):
            raise ValueError(f"rule {i}: a side is a list of edges, an edge a non-empty list of "
                             f"vertices, and the left side has at least one edge: {rule!r}")
        number = {}
        for e in list(lhs) + list(rhs):
            for v in _as_list(e):
                number.setdefault(v, len(number))
        out[f"Rule{i + 1}"] = Function("Rule", [[[number[v] for v in _as_list(e)] for e in lhs],
                                                [[number[v] for v in _as_list(e)] for e in rhs]])
    return out


def initial_states(initial):
    """The initial states as lists of edges of integers. One hypergraph is one state; a list of
    hypergraphs is several. Vertices that are not all integers are numbered from 0 by first
    appearance across the states, as HGEvolve's normalizeInitialStates numbers them."""
    initial = _as_list(initial)
    if not isinstance(initial, (list, tuple)) or not initial:
        raise ValueError(_INITIAL)
    first = _as_list(initial[0])
    one = isinstance(first, (list, tuple)) and len(first) > 0 and \
        not isinstance(_as_list(first[0]), (list, tuple))
    states = [initial] if one else [_as_list(s) for s in initial]
    out, labels = [], set()
    for s in states:
        if not isinstance(s, (list, tuple)) or not s:
            raise ValueError(_INITIAL)
        edges = [e if type(e) is list else list(_as_list(e)) for e in s]
        for e in edges:
            if not e:
                raise ValueError(_INITIAL)
            labels.update(map(type, e))
        out.append(edges)
    if labels <= {int}:
        return out
    if any(issubclass(t, (list, tuple)) for t in labels):
        raise ValueError(_INITIAL)
    number = {}
    return [[[number.setdefault(v, len(number)) for v in e] for e in s] for s in out]


_INITIAL = ("initial is a hypergraph (a non-empty list of edges, each a non-empty list of "
            "vertices) or a non-empty list of them")
