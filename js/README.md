# hypergraph-rewriting-engine for JavaScript

The package hypergraph-rewriting-engine runs the CPU engine of the HypergraphRewriting paclet in WebAssembly, in Node.js and in browsers. It runs the same job code as `HGEvolve` and the Python package (`hgevolve`), and it gives the same replies.

```js
import { load } from "hypergraph-rewriting-engine";

const hg = await load();
const r = hg.evolve([[[1, 2]], [[1, 2], [2, 3]]], [[1, 2]], 3);
r.numStates;                      // 10
r.states.column("Step");          // Float64Array of each state's step
r.causalEdges.records();          // [{From, To, RawFrom, RawTo}, ...]
```

## Two modules

The package holds two WebAssembly modules:

- `hgevolve.mjs` runs on one thread and works everywhere.
- `hgevolve-threads.mjs` runs the engine's workers on Web Workers. It needs `SharedArrayBuffer`. In a browser, the page must be served with the headers `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`.

`load()` picks the threaded module where it can run, and `load({ threads: false })` or `load({ threads: true })` picks one. A call into the threaded module blocks the calling thread until the evolution ends, so in a browser call it from a Web Worker. The single-threaded module can run on a page's main thread.

## Arguments

`hg.evolve(rules, initial, steps, options)`:

- `rules` is one rule or an array of rules. A rule is `[lhs, rhs]` or `{lhs, rhs}`, and each side is an array of edges. An edge is an array of integer vertex labels.
- `initial` is a hypergraph (an array of edges), an array of hypergraphs, or a generated initial condition: a name such as `"Grid"`, or an object such as `{Type: "BrillLindquist", Density: 400}`.
- `steps` is the number of steps.
- `options` holds HGEvolve's options under their names in camelCase (`canonicalizeStates` for `"CanonicalizeStates"`), with HGEvolve's values: strings for symbols (`"Full"`, `"Automatic"`), `true` and `false`, and `null` for `None`. `data` lists the properties of the reply (`hgevolve.DATA`; default `DEFAULT_DATA`).

The engine checks every option. A job with an option the engine does not recognise, or a value the engine cannot read, is refused with an `EngineError` before it runs.

## The reply

`evolve()` returns a `Result`. Tables (`states`, `events`, `globalEdges`, `stateBitvectors`) have `column(name)`, `record(i)` and `records()`. Relations (`causalEdges`, `branchialEdges`, `branchialStateEdges`) have `names`, `column(name)`, `row(i)` and `records()`. Integer columns are `Float64Array`, or `BigInt64Array` when a value exceeds 2^53, as canonical hashes do. `values` holds the rest of the reply under HGEvolve's names, and `warnings` holds the reply's warnings.

State and event ids are local to a run. The canonical hash of a state is the same in every run. The threaded module numbers states and events in the same order as the paclet at the same thread count. The single-threaded module numbers them in another order, and the states, events and relations are the same up to their ids.

## Sessions

```js
const s = hg.session(rules, initial, { canonicalizeStates: "Full" });
s.step(2);
s.step(1, { start: s.frontier.slice(0, 1) });
s.query({ data: ["numStates"] });
s.close();
```

One session is open per loaded engine.

## Transition costs

`transitionCost` takes `{Field, TimeStep}`, or in the single-threaded module a function `(parentEdges, childEdges) => number`. The threaded module refuses a function, because a JavaScript function can only be called from the thread that created it.

## Building

The modules are built from the repository with an Emscripten SDK:

```sh
source <emsdk>/emsdk_env.sh
js/build.sh            # writes js/dist/
cd js && npm test      # Node.js 21 or later
```

`PLAYWRIGHT_CORE=<dir>/node_modules/playwright-core node test/browser.mjs` runs both modules in headless Chromium.

## Limits

- The GPU engine is not part of the package.
- A module's memory grows to at most 4 GB.
