// Runs in a module Web Worker: loads the module asked for and evolves the growth rule.
import { load } from "../../hgevolve.mjs";

self.onmessage = async ({ data: { threads } }) => {
    try {
        const hg = await load({ threads });
        const t0 = performance.now();
        const r = hg.evolve([[[1, 2]], [[1, 2], [2, 3]]], [[1, 2]], 3);
        const big = hg.evolve([[[1, 2], [1, 3]], [[1, 2], [1, 4], [2, 4], [3, 4]]], [[1, 2], [1, 3]], 5,
            { canonicalizeStates: "Full", data: ["numStates", "numEvents"] });
        self.postMessage({ threaded: hg.threaded, numStates: r.numStates, numEvents: r.numEvents,
            causal: r.causalEdges.length, bigStates: big.numStates, bigEvents: big.numEvents,
            ms: performance.now() - t0 });
    } catch (e) {
        self.postMessage({ error: String(e && e.stack || e) });
    }
};
