"""The reply of a job: tables of records, relations and plain values, every array a read-only NumPy
view of the engine's reply."""

import re

import numpy as np

_EMPTY = np.zeros(0, dtype=np.int64)
_EMPTY.flags.writeable = False


def snake(name):
    """"ContentStateId" -> "content_state_id"."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


class Ragged:
    """Lists of numbers in CSR form: list i is values[offsets[i]:offsets[i + 1]]."""

    def __init__(self, values, offsets):
        self.values, self.offsets = values, offsets

    def __len__(self):
        return max(len(self.offsets) - 1, 0)

    def __getitem__(self, i):
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(i)
        return self.values[self.offsets[i]:self.offsets[i + 1]]

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def lengths(self):
        return np.diff(self.offsets)

    def tolist(self):
        return [x.tolist() for x in self]

    def __repr__(self):
        return f"Ragged({len(self)} lists, {len(self.values)} values)"


class EdgeLists:
    """Each record's edges: record i has edges offsets[i] to offsets[i + 1] - 1, edge j has id
    ids[j] and vertices vertices[j] (a Ragged over vertex_values and vertex_offsets)."""

    def __init__(self, ids, offsets, vertex_values, vertex_offsets):
        self.ids, self.offsets = ids, offsets
        self.vertices = Ragged(vertex_values, vertex_offsets)

    def __len__(self):
        return max(len(self.offsets) - 1, 0)

    def __getitem__(self, i):
        """Record i's edges as (edge id, vertex array) pairs."""
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(i)
        return [(int(self.ids[j]), self.vertices[j])
                for j in range(self.offsets[i], self.offsets[i + 1])]

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def lengths(self):
        """The number of edges of each record."""
        return np.diff(self.offsets)

    def tolist(self):
        """Record i's edges as lists [id, v1, v2, ...], HGEvolve's form."""
        return [[[eid] + v.tolist() for eid, v in rec] for rec in self]

    def __repr__(self):
        return f"EdgeLists({len(self)} records, {len(self.ids)} edges)"


class VertexReals:
    """Each record's vertex -> real map: record i's entries are vertices[offsets[i]:offsets[i+1]]
    with values[...] at the same positions."""

    def __init__(self, vertices, offsets, values):
        self.vertices, self.offsets, self.values = vertices, offsets, values

    def __len__(self):
        return max(len(self.offsets) - 1, 0)

    def __getitem__(self, i):
        a, b = self.offsets[i], self.offsets[i + 1]
        return dict(zip(self.vertices[a:b].tolist(), self.values[a:b].tolist()))

    def __iter__(self):
        return (self[i] for i in range(len(self)))


def _column(kind, parts):
    if kind == "int":
        return parts["values"]
    if kind == "lists":
        return Ragged(parts["values"], parts["offsets"])
    if kind == "edges":
        return EdgeLists(parts["values"], parts["offsets"], parts["vertices"],
                         parts["vertex_offsets"])
    return VertexReals(parts["values"], parts["offsets"], parts["reals"])


class Table:
    """Records as columns. t["CanonicalId"] and t.canonical_id are the same column. A column of
    integers is a 1-D array; a column of lists a Ragged; "Edges" an EdgeLists; "VertexCurvature" a
    VertexReals. A table of no records has no columns; any name then gives an empty array."""

    def __init__(self, rows, columns):
        self._rows = rows
        self._columns = {name: _column(kind, parts) for name, kind, parts in columns}
        self._snake = {snake(name): name for name in self._columns}

    def __len__(self):
        return self._rows

    @property
    def columns(self):
        return list(self._columns)

    def __getitem__(self, name):
        if name in self._columns:
            return self._columns[name]
        if self._rows == 0:
            return _EMPTY
        raise KeyError(f"{name!r} is not a column; the columns are {', '.join(self._columns)}")

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._snake:
            return self._columns[self._snake[name]]
        if self._rows == 0:
            return _EMPTY
        raise AttributeError(f"{name!r} is not a column; the columns are "
                             f"{', '.join(self._snake)}")

    def __contains__(self, name):
        return name in self._columns or name in self._snake

    def records(self):
        """The records as dicts with HGEvolve's keys, edge lists as [id, v1, v2, ...]."""
        cols = {name: (c.tolist() if hasattr(c, "tolist") else list(c))
                for name, c in self._columns.items()}
        return [{name: cols[name][i] for name in cols} for i in range(self._rows)]

    def __repr__(self):
        return f"Table({self._rows} records: {', '.join(self._columns)})"


class Relation:
    """Pairs (or rows) of ids as an n x k array, rows in ascending order. r["From"] and r.from_
    are column views of the array."""

    def __init__(self, names, array):
        self.names, self.array = list(names), array

    def __len__(self):
        return self.array.shape[0]

    def __getitem__(self, name):
        return self.array[:, self.names.index(name)]

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        for n in self.names:
            s = snake(n)
            if name in (s, s + "_"):
                return self.array[:, self.names.index(n)]
        raise AttributeError(name)

    def records(self):
        return [dict(zip(self.names, row)) for row in self.array.tolist()]

    def __repr__(self):
        return f"Relation({len(self)} rows: {', '.join(self.names)})"


class Result:
    """The reply of evolve() or of a Session verb. Each attribute is None when the job did not ask
    for it:

        states, events, global_edges, state_bitvectors            Table
        causal_edges, branchial_edges, branchial_state_edges      Relation
        branchial_state_vertices                                  1-D array
        num_states, num_events, num_causal_edges, num_branchial_edges   int
        step_statistics     list of dicts, one per step
        device              dict
        warnings            list of dicts (Kind, Count, Context, Partial)
        session, frontier   the session handle and its frontier's state ids (Session verbs)
        reduced             True when the evolution ran out of memory and the reply carries the
                            counts only

    `values` holds every key the reply has that is not a table or a relation, under HGEvolve's
    names."""

    def __init__(self, reply):
        self._reply = reply
        tables = reply.tables()
        relations = reply.relations()
        self.values = reply.values()
        t = {name: Table(rows, cols) for name, (rows, cols) in tables.items()}
        r = {name: Relation(names, array) for name, (names, array) in relations.items()}
        self.states = t.get("States")
        self.events = t.get("Events")
        self.global_edges = t.get("GlobalEdges")
        self.state_bitvectors = t.get("StateBitvectors")
        self.causal_edges = r.get("CausalEdges")
        self.branchial_edges = r.get("BranchialEdges")
        self.branchial_state_edges = r.get("BranchialStateEdges")
        v = r.get("BranchialStateVertices")
        self.branchial_state_vertices = None if v is None else v["Vertex"]
        get = self.values.get
        self.num_states = get("NumStates")
        self.num_events = get("NumEvents")
        self.num_causal_edges = get("NumCausalEdges")
        self.num_branchial_edges = get("NumBranchialEdges")
        self.step_statistics = get("StepStatistics")
        self.device = get("Device")
        self.warnings = get("Warnings", [])
        self.session = get("Session")
        self.frontier = get("Frontier")
        self.reduced = get("Reduced", 0) == 1

    def __repr__(self):
        parts = []
        for name in ("states", "events", "causal_edges", "branchial_edges"):
            x = getattr(self, name)
            if x is not None:
                parts.append(f"{name}={len(x)}")
        for name in ("num_states", "num_events", "num_causal_edges", "num_branchial_edges"):
            x = getattr(self, name)
            if x is not None:
                parts.append(f"{name}={x}")
        return f"Result({', '.join(parts)})"
