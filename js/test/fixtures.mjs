// The HGEvolve parity fixtures of the Python package (python/tests/fixtures/HGEvolve.json.gz):
// the jobs the examples of docs/en/ReferencePages/Symbols/HGEvolve.md send, with the reply the
// paclet got for each. Integers beyond 2^53 (canonical hashes) are read as BigInt.
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";

export const FIXTURES = new URL("../../python/tests/fixtures/HGEvolve.json.gz", import.meta.url);

export function loadFixtures() {
    const text = gunzipSync(readFileSync(FIXTURES)).toString();
    return JSON.parse(text, (_, v, ctx) =>
        typeof v === "number" && !Number.isSafeInteger(v) && /^-?\d+$/.test(ctx?.source ?? "")
            ? BigInt(ctx.source) : v).Fixtures;
}

// Options HGEvolve sends that the package has no use for: graph building only.
const GRAPH_ONLY = new Set(["GraphProperties", "EdgeDeduplication"]);
const camel = (name) => name.charAt(0).toLowerCase() + name.slice(1);

// evolve()'s arguments for a fixture: the job's rules and initial states, and its options in
// camelCase with `data`.
export function evolveCall(fx) {
    const options = {};
    let data = [];
    for (const [wire, v] of Object.entries(fx.Job.Options)) {
        if (GRAPH_ONLY.has(wire)) continue;
        if (wire === "RequestedData") { data = v.map(camel); continue; }
        options[camel(wire)] = v;
    }
    options.data = data;
    return { rules: fx.JobRules, initial: fx.Job.InitialStates, steps: fx.Job.Steps, options };
}

// A table's records in fixture form: an edge list as [id, vertex, ...] per edge.
function tableRecords(t) {
    return t.records().map((rec) => {
        const out = {};
        for (const name of t.columns) {
            const c = t.column(name);
            if (c.constructor.name === "EdgeLists") {
                const ids = Array.from(c.ids.subarray(c.offsets[rec.__i], c.offsets[rec.__i + 1]));
                out[name] = rec[name].map((verts, k) => [ids[k], ...verts]);
            } else out[name] = rec[name];
        }
        return out;
    });
}

// A Result in fixture form, under HGEvolve's reply keys.
export function fixtureForm(result) {
    const out = {};
    const tables = { States: result.states, Events: result.events, StateBitvectors: result.stateBitvectors };
    for (const [k, t] of Object.entries(tables)) if (t) out[k] = records(t);
    if (result.globalEdges) out.GlobalEdges = records(result.globalEdges).map((r) => [r.Id, ...r.Vertices]);
    for (const [k, r] of Object.entries({ CausalEdges: result.causalEdges, BranchialEdges: result.branchialEdges }))
        if (r) out[k] = r.records();
    for (const [k, v] of Object.entries(result.values)) out[k] = v;
    return out;
}

function records(t) {
    const out = [];
    for (let i = 0; i < t.length; ++i) {
        const rec = {};
        for (const name of t.columns) {
            const c = t.column(name);
            if (c.constructor.name === "EdgeLists") {
                const ids = Array.from(c.ids.subarray(c.offsets[i], c.offsets[i + 1]));
                rec[name] = c.at(i).map((verts, k) => [ids[k], ...verts]);
            } else if (c.at) rec[name] = c.at(i);
            else rec[name] = c[i];
        }
        out.push(rec);
    }
    return out;
}

// An object key written as a number ("2.", Wolfram Language's form of the real 2, and "2", the
// JavaScript form) as one key.
const NUMERIC_KEY = /^-?\d+\.?\d*(e-?\d+)?$/;
function keyed(o) {
    const out = {};
    for (const [k, v] of Object.entries(o)) {
        // A list key: "{2, 2}" in the fixture, "2,2" in the reply.
        const key = /^\{.*\}$/.test(k) ? k.slice(1, -1).replace(/\s+/g, "") : k;
        out[NUMERIC_KEY.test(key) ? String(Number(key)) : key] = v;
    }
    return out;
}

// Deep equality with numbers and BigInts compared by value, and numeric keys by value.
export function same(a, b) {
    if (typeof a === "bigint" || typeof b === "bigint") {
        try { return BigInt(a) === BigInt(b); } catch { return false; }
    }
    // Reals within 1e-9 relative (1e-12 absolute), the tolerance of the Python parity test: the
    // WebAssembly maths library rounds some functions to a different last bit.
    if (typeof a === "number" && typeof b === "number")
        return a === b || (Number.isNaN(a) && Number.isNaN(b)) ||
            Math.abs(a - b) <= Math.max(1e-9 * Math.max(Math.abs(a), Math.abs(b)), 1e-12);
    if (Array.isArray(a) || Array.isArray(b)) {
        if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
        return a.every((x, i) => same(x, b[i]));
    }
    if (a && b && typeof a === "object" && typeof b === "object") {
        a = keyed(a);
        b = keyed(b);
        const ka = Object.keys(a), kb = Object.keys(b);
        if (ka.length !== kb.length) return false;
        return ka.every((k) => k in b && same(a[k], b[k]));
    }
    return a === b;
}

// The first differing path of two values, for a failure message.
export function firstDifference(a, b, path = "") {
    if (same(a, b)) return null;
    if (a && b && typeof a === "object" && typeof b === "object") {
        a = keyed(a);
        b = keyed(b);
        const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
        for (const k of keys) {
            const d = firstDifference(a[k], b[k], `${path}/${k}`);
            if (d) return d;
        }
    }
    const show = (x) => JSON.stringify(x, (_, v) => (typeof v === "bigint" ? `${v}n` : v))?.slice(0, 200);
    return `${path || "/"}: ${show(a)} != ${show(b)}`;
}

// A multiset as a Map from each element's JSON text to its count.
function counter(items) {
    const m = new Map();
    for (const x of items) {
        const k = JSON.stringify(x, (_, v) => (typeof v === "bigint" ? `${v}n` : v));
        m.set(k, (m.get(k) ?? 0) + 1);
    }
    return m;
}

function groupBy(items, key) {
    const m = new Map();
    for (const x of items) {
        const k = key(x);
        if (!m.has(k)) m.set(k, []);
        m.get(k).push(x);
    }
    return m;
}

function sameCounter(a, b) {
    if (a.size !== b.size) return false;
    for (const [k, n] of a) if (b.get(k) !== n) return false;
    return true;
}

// The reply up to run-local ids (docs/SPEC.md): multisets keyed by canonical hashes, events by
// the states they join, causal and branchial pairs by their events.
function invariantView(reply) {
    const states = reply.States ?? [];
    const h = new Map(states.map((s) => [String(s.Id), s.CanonicalHash]));
    const H = (id) => h.get(String(id));
    // A raw state that the reply does not list (an isomorphic duplicate under Full states) stands
    // for its class, the event's canonical state: which copy of a class is listed, and which
    // copy's children are found first, depends on the order of exploration.
    const sig = new Map();
    for (const e of reply.Events ?? [])
        sig.set(String(e.Id), [e.RuleIndex, H(e.InputState) ?? H(e.CanonicalInputState),
            H(e.OutputState) ?? H(e.CanonicalOutputState), H(e.CanonicalInputState),
            H(e.CanonicalOutputState), (e.ConsumedEdges ?? []).length, (e.ProducedEdges ?? []).length]);
    const json = (x) => JSON.stringify(x, (_, v) => (typeof v === "bigint" ? `${v}n` : v));
    const S = (id) => json(sig.get(String(id)));
    // The partition of the events into canonical events: each class as the sorted signatures of
    // its events. A CanonicalId numbers the classes and is not a raw event's id; a causal or
    // branchial pair's From and To are canonical ids, its RawFrom and RawTo raw ids.
    const classes = groupBy(reply.Events ?? [], (e) => String(e.CanonicalId));
    const classSig = new Map([...classes].map(([k, members]) => [k, json(members.map((e) => S(e.Id)).sort())]));
    const C = (id) => classSig.get(String(id));
    const pairs = (rs, sorted) => counter(rs.map((c) => {
        const p = [C(c.From), C(c.To), S(c.RawFrom), S(c.RawTo)];
        return sorted ? [[p[0], p[1]].sort(), [p[2], p[3]].sort()] : p;
    }));
    return {
        states: counter(states.map((s) => [s.CanonicalHash, s.Step, s.IsInitial, s.Edges.length, H(s.CanonicalId), H(s.ContentStateId)])),
        events: counter([...sig.values()]),
        canonicalEvents: counter([...classSig.values()]),
        causal: pairs(reply.CausalEdges ?? [], false),
        branchial: pairs(reply.BranchialEdges ?? [], true),
        globalEdges: counter((reply.GlobalEdges ?? []).map((e) => e.length)),
        bitvectors: counter((reply.StateBitvectors ?? []).map((b) => [H(b.State), b.Edges.length])),
    };
}

const COUNTS = ["NumStates", "NumEvents", "NumCausalEdges", "NumBranchialEdges"];
const LISTS = ["States", "Events", "CausalEdges", "BranchialEdges", "GlobalEdges", "StateBitvectors"];

// {kind: "exact" | "invariant"} when the result matches the fixture's reply, else {message}.
export function compareFixture(fx, result) {
    const want = fx.Reply;
    const got = fixtureForm(result);
    for (const k of COUNTS)
        if (!same(want[k], got[k])) return { message: `${k}: ${want[k]} vs ${got[k]}` };
    if ("StepStatistics" in want) {
        const d = firstDifference(got.StepStatistics, want.StepStatistics, "/StepStatistics");
        if (d) return { message: d };
    }
    const byId = (rs) => [...(rs ?? [])].sort((x, y) => Number(x.Id) - Number(y.Id));
    let exact = true;
    for (const k of LISTS) {
        const w = k === "States" || k === "Events" ? byId(want[k]) : want[k] ?? [];
        const g = k === "States" || k === "Events" ? byId(got[k]) : got[k] ?? [];
        if (!same(w, g)) exact = false;
    }
    if (exact) return { kind: "exact" };
    const a = invariantView(want), b = invariantView(got);
    for (const k of Object.keys(a)) if (!sameCounter(a[k], b[k])) return { message: `${k} differ up to ids` };
    return { kind: "invariant" };
}
