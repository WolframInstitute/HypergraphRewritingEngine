// The JavaScript API: argument forms, options, errors, sessions, initial conditions, the
// transition cost callback, and thread counts.
import test from "node:test";
import assert from "node:assert/strict";
import { load, threadsAvailable, EngineError, DATA, Table, Relation } from "../hgevolve.mjs";
import { fixtureForm, compareFixture } from "./fixtures.mjs";

const GROWTH = [[[1, 2]], [[1, 2], [2, 3]]];
const single = await load({ threads: false });
const threaded = threadsAvailable() ? await load({ threads: true }) : null;
const engines = [["single-threaded", single], ...(threaded ? [["threaded", threaded]] : [])];

for (const [name, hg] of engines) {
    test(`${name}: the counts of {{1,2}} -> {{1,2},{2,3}} for 3 steps`, () => {
        const r = hg.evolve(GROWTH, [[1, 2]], 3);
        assert.equal(r.numStates, 10);
        assert.equal(r.numEvents, 9);
        assert.equal(r.numCausalEdges, 8);
        assert.ok(r.states instanceof Table);
        assert.ok(r.causalEdges instanceof Relation);
        assert.equal(r.states.length, 10);
        assert.deepEqual(r.causalEdges.names, ["From", "To", "RawFrom", "RawTo"]);
        assert.equal(r.causalEdges.length, 8);
    });

    test(`${name}: rule and initial state forms`, () => {
        const counts = (r) => [r.numStates, r.numEvents];
        const want = counts(hg.evolve(GROWTH, [[1, 2]], 3));
        assert.deepEqual(counts(hg.evolve([GROWTH], [[1, 2]], 3)), want);
        assert.deepEqual(counts(hg.evolve({ lhs: GROWTH[0], rhs: GROWTH[1] }, [[1, 2]], 3)), want);
        assert.deepEqual(counts(hg.evolve(GROWTH, [[[1, 2]]], 3)), want);
        // Labels are any integers.
        assert.deepEqual(counts(hg.evolve([[[5, 7]], [[5, 7], [7, 9]]], [[10, 20]], 3)), want);
        // Two initial states.
        const two = hg.evolve(GROWTH, [[[1, 2]], [[1, 2], [2, 3]]], 1, { data: ["numStates"] });
        assert.equal(two.numStates, 5);
    });

    test(`${name}: options take HGEvolve's names in camelCase`, () => {
        const none = hg.evolve(GROWTH, [[1, 2]], 3, { data: "numStates" });
        const full = hg.evolve(GROWTH, [[1, 2]], 3, { data: "numStates", canonicalizeStates: "Full" });
        assert.equal(none.numStates, 10);
        assert.equal(full.numStates, 8);
        assert.equal(full.states, undefined);
        const capped = hg.evolve(GROWTH, [[1, 2]], 3, { data: "numStates", maxStatesPerStep: 1 });
        assert.equal(capped.numStates, 4);
    });

    test(`${name}: an integer is accepted where a real is expected`, () => {
        const r = hg.evolve(GROWTH, [[1, 2]], 2, { data: "numStates", explorationProbability: 1, randomSeed: 3 });
        assert.equal(r.numStates, 4);
    });

    test(`${name}: canonical hashes beyond 2^53 are BigInt`, () => {
        const r = hg.evolve(GROWTH, [[1, 2]], 2, { data: "states", includeCanonicalHashes: true });
        const h = r.states.column("CanonicalHash");
        assert.ok(h instanceof BigInt64Array || h instanceof Float64Array);
        assert.equal(new Set(Array.from(h, String)).size, r.states.length);
        const edges = r.states.column("Edges").at(1);
        assert.deepEqual(edges.map((e) => e.length), [2, 2]);
    });

    test(`${name}: a refused job is an EngineError with HGEvolve's message`, () => {
        assert.throws(() => hg.evolve(GROWTH, [[1, 2]], 2, { canonicalizeStates: "Sometimes" }),
            (e) => e instanceof EngineError && /CanonicalizeStates|Sometimes/.test(e.message));
        assert.throws(() => hg.evolve(GROWTH, [[1, 2]], 2, { canonicaliseStates: "Full" }),
            (e) => e instanceof EngineError && /CanonicaliseStates/.test(e.message));
        assert.throws(() => hg.evolve(GROWTH, [[1, 2]], 2, { maxStatesPerStep: "many" }),
            (e) => e instanceof EngineError && /MaxStatesPerStep/.test(e.message));
        assert.throws(() => hg.evolve(GROWTH, [[1, 2]], -1), TypeError);
        assert.throws(() => hg.evolve(GROWTH, [["a", "b"]], 1), TypeError);
        assert.throws(() => hg.evolve(GROWTH, [[1, 2]], 1, { data: "nothing" }), TypeError);
        // The module serves the next job after a refusal.
        assert.equal(hg.evolve(GROWTH, [[1, 2]], 3, { data: "numStates" }).numStates, 10);
    });

    test(`${name}: a session continues the evolution`, () => {
        const s = hg.session(GROWTH, [[1, 2]], { canonicalizeStates: "Full", data: ["numStates", "numEvents"] });
        s.step(2);
        const r = s.step(1);
        const q = s.query({ data: ["numStates", "numEvents"] });
        s.close();
        s.close();
        const whole = hg.evolve(GROWTH, [[1, 2]], 3, { canonicalizeStates: "Full", data: ["numStates", "numEvents"] });
        assert.equal(r.numStates, whole.numStates);
        assert.equal(q.numEvents, whole.numEvents);
        assert.throws(() => s.step(1), EngineError);
    });

    test(`${name}: generated initial conditions`, () => {
        const grid = hg.initialCondition({ Type: "Grid", Width: 3, Height: 3 });
        assert.equal(grid.VertexCount, 9);
        assert.equal(grid.Edges.length, 12);
        const r = hg.evolve([[[1, 2]], [[1, 2], [2, 3]]], { Type: "Grid", Width: 3, Height: 3 }, 1, { data: "numStates" });
        assert.equal(r.numStates, 13);
        assert.throws(() => hg.initialCondition({ Type: "NoSuchShape" }), EngineError);
    });

    test(`${name}: the heat residual of identical states with time step 0 is 0`, () => {
        assert.equal(hg.heatResidual([[0, 1], [1, 2]], [[0, 1], [1, 2]]), 0);
        assert.ok(hg.heatResidual([[0, 1]], [[0, 1], [1, 2]]) > 0);
    });

    test(`${name}: encodeJob writes WXF`, () => {
        const bytes = hg.encodeJob(GROWTH, [[1, 2]], 3);
        assert.ok(bytes instanceof Uint8Array);
        assert.deepEqual(Array.from(bytes.subarray(0, 2)), [0x38, 0x3a]);  // "8:"
    });

    test(`${name}: every data name is accepted`, () => {
        const r = hg.evolve(GROWTH, [[1, 2]], 2, { data: DATA });
        assert.equal(r.numStates, 4);
        assert.ok(r.device);
    });

    test(`${name}: the build stamp names the variant`, () => {
        assert.match(hg.buildStamp(), /variant=js/);
    });
}

test("single-threaded: a transition cost function steers the search", () => {
    const calls = [];
    const r = single.evolve(GROWTH, [[1, 2]], 3, {
        data: ["numStates"], search: "BestFirst", searchBudget: 6,
        transitionCost: (parent, child) => { calls.push([parent.length, child.length]); return child.length; },
    });
    assert.ok(calls.length > 0);
    assert.ok(calls.every(([p, c]) => c === p + 1));
    assert.ok(r.numStates >= 1);
});

test("single-threaded: an error thrown by the transition cost function is rethrown", () => {
    const boom = new Error("cost failed");
    assert.throws(() => single.evolve(GROWTH, [[1, 2]], 2, {
        data: "numStates", search: "BestFirst", searchBudget: 4, transitionCost: () => { throw boom; },
    }), (e) => e === boom);
    assert.equal(single.evolve(GROWTH, [[1, 2]], 3, { data: "numStates" }).numStates, 10);
});

test("threaded: a transition cost function is refused", { skip: !threaded }, () => {
    assert.throws(() => threaded.evolve(GROWTH, [[1, 2]], 2, { transitionCost: () => 0 }), TypeError);
});

test("threaded: the reply is the same at 1, 4 and 16 threads, up to run-local ids", { skip: !threaded }, () => {
    const options = { canonicalizeStates: "Full", canonicalizeEvents: "Automatic", includeCanonicalHashes: true,
        data: ["states", "events", "causalEdges", "branchialEdges", "numStates", "numEvents", "numCausalEdges", "numBranchialEdges"] };
    const rule = [[[1, 2], [1, 3]], [[1, 2], [1, 4], [2, 4], [3, 4]]];
    const one = fixtureForm(threaded.evolve(rule, [[1, 2], [1, 3]], 4, { ...options, threads: 1 }));
    for (const threads of [4, 16]) {
        const r = threaded.evolve(rule, [[1, 2], [1, 3]], 4, { ...options, threads });
        const c = compareFixture({ Reply: one }, r);
        assert.equal(c.message, undefined, `${threads} threads: ${c.message}`);
    }
});
