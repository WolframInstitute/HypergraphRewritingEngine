// Parity with HGEvolve: each job sent by the examples of docs/en/ReferencePages/Symbols/HGEvolve.md
// (python/tests/fixtures/HGEvolve.json.gz) gives the paclet's reply through evolve(), on both
// modules. The threaded module runs a job's "Threads" -> 1 on one worker, as the paclet did, and
// its replies are compared exactly, ids included. The single-threaded module runs the job on the
// calling thread, which numbers states and events in another order: its replies are compared up
// to run-local ids (docs/SPEC.md).
import test from "node:test";
import assert from "node:assert/strict";
import { load, threadsAvailable } from "../hgevolve.mjs";
import { loadFixtures, evolveCall, compareFixture } from "./fixtures.mjs";

const FIXTURES = loadFixtures();

test("the fixture set holds the documentation's jobs", () => {
    assert.ok(FIXTURES.length >= 200, `${FIXTURES.length} fixtures`);
});

for (const threads of [false, true]) {
    test(`every fixture matches HGEvolve (${threads ? "threaded" : "single-threaded"} module)`,
        { skip: threads && !threadsAvailable() }, async () => {
            const hg = await load({ threads });
            assert.equal(hg.threaded, threads);
            const failed = [];
            let exact = 0;
            for (const fx of FIXTURES) {
                const c = evolveCall(fx);
                const r = compareFixture(fx, hg.evolve(c.rules, c.initial, c.steps, c.options));
                if (r.message) failed.push(`fixture ${fx.Index}: ${r.message}`);
                else if (r.kind === "exact") exact++;
            }
            assert.deepEqual(failed, []);
            if (threads) assert.equal(exact, FIXTURES.length, "the threaded module's replies are exact");
        });
}

// The comparison reports a reply that differs from the fixture's: a state's hash, an event moved
// to another canonical event, and a causal pair pointed at another event.
test("the comparison refuses a changed reply", async () => {
    const hg = await load({ threads: false });
    const fx = FIXTURES.find((f) => f.Job.Options.CanonicalizeEvents === "Automatic" &&
        f.Reply.CausalEdges.length > 1 && new Set(f.Reply.Events.map((e) => e.CanonicalId)).size > 1);
    assert.ok(fx, "a fixture with canonical events and causal edges");
    const c = evolveCall(fx);
    const run = () => hg.evolve(c.rules, c.initial, c.steps, c.options);
    assert.equal(compareFixture(fx, run()).message, undefined);
    const changed = (edit) => {
        const reply = structuredClone(fx.Reply);
        edit(reply);
        return compareFixture({ ...fx, Reply: reply }, run()).message;
    };
    assert.ok(changed((r) => { r.States[0].CanonicalHash = 12345n; }));
    assert.ok(changed((r) => {
        const ids = [...new Set(r.Events.map((e) => e.CanonicalId))];
        const e = r.Events.find((x) => x.CanonicalId === ids[0]);
        e.CanonicalId = ids[1];
    }));
    // The pair's consumer moved to an event that leaves another state class: a pair between
    // events of the same signatures is the same pair up to ids.
    assert.ok(changed((r) => {
        const pair = r.CausalEdges[0];
        const to = r.Events.find((e) => e.Id === pair.RawTo);
        const other = r.Events.find((e) => e.CanonicalInputState !== to.CanonicalInputState);
        pair.RawTo = other.Id;
        pair.To = other.CanonicalId;
    }));
});
