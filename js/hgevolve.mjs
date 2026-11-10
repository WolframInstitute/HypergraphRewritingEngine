// hgevolve: multiway hypergraph rewriting with the HypergraphRewriting engine, compiled to
// WebAssembly. The same job code runs here as in HGEvolve and in the Python package hgevolve.
//
//   import { load } from "hgevolve";
//   const hg = await load();
//   const r = hg.evolve([[[1, 2]], [[1, 2], [2, 3]]], [[1, 2]], 3);
//   r.numStates, r.states.column("Step"), r.causalEdges.column("From")
//
// Options take HGEvolve's names in camelCase ("CanonicalizeStates" is canonicalizeStates) and
// HGEvolve's values: strings for symbols ("Full", "Automatic"), true and false, null for None.
// The engine checks each option and refuses a job with HGEvolve's message (EngineError).

const SINGLE = "./dist/hgevolve.mjs";
const THREADS = "./dist/hgevolve-threads.mjs";

export class EngineError extends Error {
    constructor(message) {
        super(message);
        this.name = "EngineError";
    }
}

// HGEvolve "RequestedData" names under evolve()'s `data`, and the default.
export const DATA = ["states", "events", "eventsMinimal", "causalEdges", "branchialEdges",
    "branchialStateEdges", "branchialStateEdgesAllSiblings", "numStates", "numEvents",
    "numCausalEdges", "numBranchialEdges", "stepStatistics", "globalEdges", "stateBitvectors",
    "device"];
export const DEFAULT_DATA = ["states", "events", "causalEdges", "branchialEdges", "numStates",
    "numEvents", "numCausalEdges", "numBranchialEdges"];

// True where the threaded module can run: SharedArrayBuffer exists, and in a browser the page is
// cross-origin isolated (served with COOP and COEP headers).
export function threadsAvailable() {
    if (typeof SharedArrayBuffer === "undefined") return false;
    if (typeof process !== "undefined" && process.versions?.node) return true;
    return globalThis.crossOriginIsolated === true;
}

// Loads the engine. threads: "auto" (the threaded module where threadsAvailable()), true or
// false. In a browser the threaded module blocks the thread that calls it, so call it from a
// Web Worker there.
export async function load({ threads = "auto" } = {}) {
    const useThreads = threads === "auto" ? threadsAvailable() : Boolean(threads);
    if (useThreads && !threadsAvailable())
        throw new Error("the threaded module needs SharedArrayBuffer and, in a browser, a " +
                        "cross-origin isolated page");
    const { default: create } = await import(useThreads ? THREADS : SINGLE);
    const module = await create();
    module.HgEngineError = EngineError;
    module.hgCallCost = (fn, parent, child) => {
        try {
            return { ok: true, value: Number(fn(parent, child)) };
        } catch (error) {
            return { ok: false, error };
        }
    };
    return new Engine(module);
}

// ---------------------------------------------------------------------------------------------
// The job envelope (hgJobOptions in paclet/Kernel/HypergraphRewriting.wl)

const sym = (name) => ({ $symbol: name });
const pascal = (name) => name.charAt(0).toUpperCase() + name.slice(1);

function isInt(x) {
    return (typeof x === "number" && Number.isInteger(x)) || typeof x === "bigint";
}

function isEdge(e) {
    return Array.isArray(e) && e.length > 0 && e.every(isInt);
}

function isHypergraph(h) {
    return Array.isArray(h) && h.length > 0 && h.every(isEdge);
}

// {Rule1: Rule[lhs, rhs], ...}. A rule is [lhs, rhs] or {lhs, rhs}, a side a list of edges of
// integer vertex labels; `rules` is one rule or an array of them.
function rulesAssociation(rules) {
    const one = (r) => (r && !Array.isArray(r) && "lhs" in r) ||
        (Array.isArray(r) && r.length === 2 && r.every((side) => Array.isArray(side) &&
            side.every(isEdge)));
    const list = one(rules) ? [rules] : rules;
    if (!Array.isArray(list) || list.length === 0) throw new TypeError("rules is a rule or a non-empty array of rules");
    const out = {};
    list.forEach((r, i) => {
        if (!one(r)) throw new TypeError(`rule ${i} is not [lhs, rhs]: ${JSON.stringify(r)}`);
        const [lhs, rhs] = Array.isArray(r) ? r : [r.lhs, r.rhs];
        if (lhs.length === 0) throw new TypeError(`rule ${i}: the left side has no edge`);
        out[`Rule${i + 1}`] = { $head: "Rule", args: [lhs, rhs] };
    });
    return out;
}

function initialStates(engine, initial, seed) {
    if (typeof initial === "string" || (initial && !Array.isArray(initial))) {
        const spec = typeof initial === "string" ? { Type: initial } : { ...initial };
        if (!("Seed" in spec)) spec.Seed = seed;
        return [engine.initialCondition(spec).Edges];
    }
    if (isHypergraph(initial)) return [initial];
    if (Array.isArray(initial) && initial.length > 0 && initial.every(isHypergraph)) return initial;
    throw new TypeError("initial is a hypergraph (a non-empty array of edges, each a non-empty " +
                        "array of integer vertices), an array of them, or an initial condition " +
                        "name or spec");
}

function names(value, all) {
    if (value === null || value === undefined) return [];
    if (value === "all") return all();
    return typeof value === "string" ? [value] : [...value];
}

function randomSeed() {
    return Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
}

// The job's "Options": each option under its HGEvolve name, RandomSeed and BranchialStep with the
// values HGEvolve sends when the caller gives none, and RequestedData.
function jobOptions(engine, options, data) {
    const out = { RandomSeed: randomSeed(), BranchialStep: -1 };
    for (const [name, value] of Object.entries(options)) {
        switch (name) {
            case "transitionCost":
                out.TransitionCost = typeof value === "function" ? { Field: "Function" }
                    : value === null ? sym("None") : value;
                break;
            case "branchialStep":
                out.BranchialStep = value === "all" ? 0 : value;
                break;
            case "stepStatisticsMeasures":
                out.StepStatisticsMeasures = names(value, () => engine.measureNames);
                break;
            case "stepStatisticsMatrices":
                out.StepStatisticsMatrices = names(value, () => engine.matrixNames);
                break;
            case "stepStatisticsBranchial":
                out.StepStatisticsBranchial = names(value, () => ["Graph", "Overlap"]);
                break;
            case "eventOrderingFunction":
                out.EventOrderingFunction = value === null || value === "None" ? []
                    : value === "Automatic" ? sym("Automatic")
                    : typeof value === "string" ? [value] : value;
                break;
            case "randomSeed":
                out.RandomSeed = value === null ? randomSeed() : value;
                break;
            default:
                out[pascal(name)] = value;
        }
    }
    const wanted = data === undefined || data === null ? DEFAULT_DATA
        : typeof data === "string" ? [data] : data;
    for (const d of wanted)
        if (!DATA.includes(d)) throw new TypeError(`data "${d}" is not one of ${DATA.join(", ")}`);
    out.RequestedData = wanted.map(pascal);
    return out;
}

function costFunction(options) {
    return typeof options.transitionCost === "function" ? options.transitionCost : undefined;
}

// ---------------------------------------------------------------------------------------------
// The reply

// A reply integer column: a Float64Array when every value is exact as a number, a BigInt64Array
// otherwise (canonical hashes).
function integers(a) {
    for (let i = 0; i < a.length; ++i)
        if (a[i] > 9007199254740991n || a[i] < -9007199254740991n) return a;
    return Float64Array.from(a, Number);
}

// Records in compressed rows: the values of record i are values[offsets[i]..offsets[i+1]).
export class Ragged {
    constructor(values, offsets) {
        this.values = values;
        this.offsets = offsets;
    }
    get length() { return this.offsets.length - 1; }
    at(i) { return Array.from(this.values.subarray(this.offsets[i], this.offsets[i + 1])); }
    toArray() { return Array.from({ length: this.length }, (_, i) => this.at(i)); }
}

// Each record's edges: the ids of record i, and each edge's vertices.
export class EdgeLists {
    constructor(ids, offsets, vertices, vertexOffsets) {
        this.ids = ids;
        this.offsets = offsets;
        this.vertices = vertices;
        this.vertexOffsets = vertexOffsets;
    }
    get length() { return this.offsets.length - 1; }
    // The edges of record i as arrays of vertices.
    at(i) {
        const out = [];
        for (let e = this.offsets[i]; e < this.offsets[i + 1]; ++e)
            out.push(Array.from(this.vertices.subarray(this.vertexOffsets[e], this.vertexOffsets[e + 1])));
        return out;
    }
    toArray() { return Array.from({ length: this.length }, (_, i) => this.at(i)); }
}

// A real per vertex of each record: record i's vertices and their values.
export class VertexReals {
    constructor(vertices, offsets, values) {
        this.vertices = vertices;
        this.offsets = offsets;
        this.values = values;
    }
    get length() { return this.offsets.length - 1; }
    at(i) {
        const a = this.offsets[i], b = this.offsets[i + 1];
        return { vertices: Array.from(this.vertices.subarray(a, b)), values: Array.from(this.values.subarray(a, b)) };
    }
    toArray() { return Array.from({ length: this.length }, (_, i) => this.at(i)); }
}

// Records as named columns ("States", "Events", ...). column(name) is an integer array for a
// one-integer column, and a Ragged, EdgeLists or VertexReals otherwise.
export class Table {
    constructor(raw) {
        this.length = raw.rows;
        this.columns = raw.columns.map((c) => c.name);
        this._columns = new Map();
        for (const c of raw.columns) {
            const values = integers(c.values);
            let col = values;
            if (c.kind === "lists") col = new Ragged(values, integers(c.offsets));
            else if (c.kind === "edges")
                col = new EdgeLists(values, integers(c.offsets), integers(c.vertices), integers(c.vertexOffsets));
            else if (c.kind === "vertex_reals") col = new VertexReals(values, integers(c.offsets), c.reals);
            this._columns.set(c.name, col);
        }
    }
    has(name) { return this._columns.has(name); }
    column(name) {
        const c = this._columns.get(name);
        if (c === undefined) throw new RangeError(`no column "${name}"; the columns are ${this.columns.join(", ")}`);
        return c;
    }
    // Record i as an object of its columns.
    record(i) {
        const o = {};
        for (const [name, c] of this._columns) o[name] = c.at ? c.at(i) : c[i];
        return o;
    }
    records() { return Array.from({ length: this.length }, (_, i) => this.record(i)); }
}

// Rows of integers under named columns (CausalEdges: From, To, RawFrom, RawTo, ...).
export class Relation {
    constructor(raw) {
        this.names = raw.names;
        this.length = raw.rows;
        this.cells = integers(raw.cells);
    }
    // Column `name` of every row.
    column(name) {
        const k = this.names.indexOf(name);
        if (k < 0) throw new RangeError(`no column "${name}"; the columns are ${this.names.join(", ")}`);
        const w = this.names.length;
        const out = new this.cells.constructor(this.length);
        for (let i = 0; i < this.length; ++i) out[i] = this.cells[i * w + k];
        return out;
    }
    row(i) {
        const w = this.names.length;
        return Array.from(this.cells.subarray(i * w, (i + 1) * w));
    }
    records() {
        return Array.from({ length: this.length }, (_, i) => {
            const r = this.row(i);
            return Object.fromEntries(this.names.map((n, k) => [n, r[k]]));
        });
    }
}

// The reply of evolve() or of a Session verb. A field is undefined when the job did not ask for
// it. `values` holds every key of the reply that is not a table or a relation, under HGEvolve's
// names.
export class Result {
    constructor(raw) {
        const t = (n) => (raw.tables[n] ? new Table(raw.tables[n]) : undefined);
        const r = (n) => (raw.relations[n] ? new Relation(raw.relations[n]) : undefined);
        this.values = raw.values;
        this.states = t("States");
        this.events = t("Events");
        this.globalEdges = t("GlobalEdges");
        this.stateBitvectors = t("StateBitvectors");
        this.causalEdges = r("CausalEdges");
        this.branchialEdges = r("BranchialEdges");
        this.branchialStateEdges = r("BranchialStateEdges");
        const v = r("BranchialStateVertices");
        this.branchialStateVertices = v ? v.column("Vertex") : undefined;
        const g = raw.values;
        this.numStates = g.NumStates;
        this.numEvents = g.NumEvents;
        this.numCausalEdges = g.NumCausalEdges;
        this.numBranchialEdges = g.NumBranchialEdges;
        this.stepStatistics = g.StepStatistics;
        this.device = g.Device;
        this.warnings = g.Warnings ?? [];
        this.session = g.Session;
        this.frontier = g.Frontier;
        this.pathCosts = g.PathCosts;
        this.reduced = g.Reduced === 1;
    }
}

// ---------------------------------------------------------------------------------------------

export class Engine {
    constructor(module) {
        this._m = module;
        // True for the threaded module.
        this.threaded = module.threaded();
        this.measureNames = module.measureNames();
        this.matrixNames = module.matrixNames();
    }

    // The engine build's stamp: commit, variant and configuration.
    buildStamp() { return this._m.buildStamp(); }

    _run(job, cost) {
        return new Result(this._m.runJob(job, cost));
    }

    _job(rules, initial, steps, options) {
        const { data, ...rest } = options;
        if (!Number.isInteger(steps) || steps < 0)
            throw new TypeError(`steps must be a non-negative integer, got ${steps}`);
        const opts = jobOptions(this, rest, data);
        return {
            InitialStates: initialStates(this, initial, opts.RandomSeed),
            Rules: rulesAssociation(rules),
            Steps: steps,
            Options: opts,
        };
    }

    // HGEvolve[rules, initial, steps]. options: `data` (names from DATA, default DEFAULT_DATA)
    // and HGEvolve's options in camelCase. transitionCost may be a function
    // (parentEdges, childEdges) => number in the single-threaded module.
    evolve(rules, initial, steps, options = {}) {
        return this._run(this._job(rules, initial, steps, options), costFunction(options));
    }

    // The WXF bytes of the job evolve() runs, as hg_evolve reads them on stdin.
    encodeJob(rules, initial, steps, options = {}) {
        return this._m.encode(this._job(rules, initial, steps, options));
    }

    // HGSessionOpen: an evolution that can be continued. One session is open per engine.
    session(rules, initial, options = {}) {
        return new Session(this, rules, initial, options);
    }

    // A generated initial condition ("Grid", "Torus", "BrillLindquist", ...): spec is a type name
    // or an object with Type and the generator's keys under HGEvolve's names. Returns the
    // generator's values: Edges, VertexCount, VertexCoordinates and the others it gives.
    initialCondition(spec) {
        const s = typeof spec === "string" ? { Type: spec } : spec;
        if (!s || typeof s !== "object" || !("Type" in s))
            throw new TypeError("an initial condition spec is a type name or an object with Type");
        const values = this._m.runJob({ Op: "InitialCondition", InitialCondition: s, Options: {} }, undefined).values;
        if (!values.Edges || values.Edges.length === 0)
            throw new EngineError(`the ${s.Type} initial condition generated no edges`);
        return values;
    }

    // The engine's "TransitionCost" residual of `field` from `parent` to `child` (edge arrays).
    heatResidual(parent, child, { field = "Degree", timeStep = 0, statisticsGraph = "Consecutive" } = {}) {
        return this._m.heatResidual(parent, child, field, timeStep, statisticsGraph);
    }
}

export class Session {
    constructor(engine, rules, initial, options = {}) {
        this._engine = engine;
        this._options = { ...options };
        if (this._options.randomSeed === undefined || this._options.randomSeed === null)
            this._options.randomSeed = randomSeed();
        const job = engine._job(rules, initial, 0, this._options);
        job.Op = "Open";
        this.result = engine._run(job, costFunction(this._options));
        this.handle = this.result.session;
        this.frontier = this.result.frontier;
    }

    _verb(op, steps, start, data) {
        if (this.handle === undefined) throw new EngineError("the session is closed");
        const { data: _, ...rest } = this._options;
        const job = { Steps: steps, Op: op, Session: this.handle,
            Options: jobOptions(this._engine, rest, data ?? this._options.data) };
        if (start !== undefined) job.From = [...start];
        const result = this._engine._run(job, costFunction(this._options));
        this.frontier = result.frontier;
        return result;
    }

    // HGSessionStep: `steps` more steps from the frontier, or from the frontier states `start`.
    step(steps = 1, { start, data } = {}) { return this._verb("Step", steps, start, data); }

    // HGSessionQuery: the session's graph, explored no further.
    query({ data } = {}) { return this._verb("Query", 0, undefined, data); }

    // HGSessionClose. Closing a closed session does nothing.
    close() {
        if (this.handle === undefined) return;
        const handle = this.handle;
        this.handle = undefined;
        this._engine._m.runJob({ Op: "Close", Session: handle, Options: {} }, undefined);
    }
}
