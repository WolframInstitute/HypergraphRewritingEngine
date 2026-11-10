// Types of the hgevolve JavaScript package (hgevolve.mjs).

/** An edge: its vertices, as integers. */
export type Edge = number[];
/** A hypergraph: its edges. */
export type Hypergraph = Edge[];
/** A rule: the left and right sides, each a list of edges. */
export type Rule = [Hypergraph, Hypergraph] | { lhs: Hypergraph; rhs: Hypergraph };

/** HGEvolve "RequestedData" names under evolve()'s `data`. */
export type DataName =
    | "states" | "events" | "eventsMinimal" | "causalEdges" | "branchialEdges"
    | "branchialStateEdges" | "branchialStateEdgesAllSiblings" | "numStates" | "numEvents"
    | "numCausalEdges" | "numBranchialEdges" | "stepStatistics" | "globalEdges"
    | "stateBitvectors" | "device";

export const DATA: DataName[];
export const DEFAULT_DATA: DataName[];

/** A generated initial condition: a type name, or an object with Type and the generator's keys. */
export type InitialConditionSpec = string | ({ Type: string } & Record<string, unknown>);

/**
 * evolve()'s options: `data`, and HGEvolve's options in camelCase with HGEvolve's values
 * (strings for symbols, true and false, null for None). docs/API.md lists them.
 */
export interface EvolveOptions {
    data?: DataName | DataName[];
    canonicalizeStates?: "None" | "Automatic" | "Full" | null;
    canonicalizeEvents?: "None" | "Automatic" | "Full" | "Positional" | string[] | null;
    causalTransitiveReduction?: boolean;
    showGenesisEvents?: boolean;
    exploreFromCanonicalStatesOnly?: boolean;
    undirectedEdges?: boolean;
    generational?: boolean;
    eventOrderingFunction?: "Automatic" | string | string[] | null;
    maxSuccessorStatesPerParent?: number;
    maxStatesPerStep?: number;
    matchesPerStateRule?: number;
    uniformRandom?: boolean;
    matchesPerStep?: number;
    explorationProbability?: number;
    transitionRate?: number;
    ruleWeights?: number[];
    /** {Field, TimeStep}, or (single-threaded module) a function of the parent's and child's edges. */
    transitionCost?: { Field: string; TimeStep?: number } | ((parent: Hypergraph, child: Hypergraph) => number) | null;
    search?: "Automatic" | "BestFirst";
    searchBudget?: number;
    beamCost?: "Transition" | "Path";
    randomSeed?: number | bigint | null;
    branchialStep?: number | "all";
    includeCanonicalHashes?: boolean;
    includeVertexCurvature?: boolean;
    stepStatisticsWeighting?: "States" | "Classes";
    statisticsGraph?: "Consecutive" | "AllPairs";
    stepStatisticsMeasures?: string | string[] | "all" | null;
    stepStatisticsBranchial?: string | string[] | "all" | null;
    stepStatisticsMatrices?: string | string[] | "all" | null;
    stepStatisticsMatrixLimit?: number;
    /** Worker threads of the threaded module; the single-threaded module runs one. */
    threads?: number;
    tuning?: Record<string, number>;
    showProgress?: boolean;
    [option: string]: unknown;
}

/** The engine refused the job (HGEvolve's message for the same job). */
export class EngineError extends Error {}

/** Integers of a reply: Float64Array, or BigInt64Array when a value exceeds 2^53 (hashes). */
export type Integers = Float64Array | BigInt64Array;

export class Ragged {
    readonly values: Integers;
    readonly offsets: Integers;
    readonly length: number;
    at(i: number): (number | bigint)[];
    toArray(): (number | bigint)[][];
}

export class EdgeLists {
    readonly ids: Integers;
    readonly offsets: Integers;
    readonly vertices: Integers;
    readonly vertexOffsets: Integers;
    readonly length: number;
    /** Record i's edges, each as its vertices. */
    at(i: number): (number | bigint)[][];
    toArray(): (number | bigint)[][][];
}

export class VertexReals {
    readonly vertices: Integers;
    readonly offsets: Integers;
    readonly values: Float64Array;
    readonly length: number;
    at(i: number): { vertices: (number | bigint)[]; values: number[] };
    toArray(): { vertices: (number | bigint)[]; values: number[] }[];
}

export type Column = Integers | Ragged | EdgeLists | VertexReals;

export class Table {
    readonly length: number;
    readonly columns: string[];
    has(name: string): boolean;
    column(name: string): Column;
    record(i: number): Record<string, unknown>;
    records(): Record<string, unknown>[];
}

export class Relation {
    readonly names: string[];
    readonly length: number;
    /** Row-major: row i is cells[i * names.length ... (i + 1) * names.length). */
    readonly cells: Integers;
    column(name: string): Integers;
    row(i: number): (number | bigint)[];
    records(): Record<string, number | bigint>[];
}

export class Result {
    /** Every key of the reply that is not a table or a relation, under HGEvolve's names. */
    readonly values: Record<string, unknown>;
    readonly states?: Table;
    readonly events?: Table;
    readonly globalEdges?: Table;
    readonly stateBitvectors?: Table;
    readonly causalEdges?: Relation;
    readonly branchialEdges?: Relation;
    readonly branchialStateEdges?: Relation;
    readonly branchialStateVertices?: Integers;
    readonly numStates?: number;
    readonly numEvents?: number;
    readonly numCausalEdges?: number;
    readonly numBranchialEdges?: number;
    readonly stepStatistics?: Record<string, unknown>[];
    readonly device?: Record<string, unknown>;
    readonly warnings: { Kind: string; Count: number; Context: string; Partial: number }[];
    readonly session?: number;
    readonly frontier?: number[];
    readonly pathCosts?: Record<string, number>;
    /** True when the evolution ran out of memory and the reply carries the counts only. */
    readonly reduced: boolean;
}

export class Session {
    readonly result: Result;
    readonly handle?: number;
    frontier?: number[];
    step(steps?: number, options?: { start?: number[]; data?: DataName | DataName[] }): Result;
    query(options?: { data?: DataName | DataName[] }): Result;
    close(): void;
}

export class Engine {
    /** True for the threaded module. */
    readonly threaded: boolean;
    readonly measureNames: string[];
    readonly matrixNames: string[];
    buildStamp(): string;
    evolve(rules: Rule | Rule[], initial: Hypergraph | Hypergraph[] | InitialConditionSpec,
           steps: number, options?: EvolveOptions): Result;
    encodeJob(rules: Rule | Rule[], initial: Hypergraph | Hypergraph[] | InitialConditionSpec,
              steps: number, options?: EvolveOptions): Uint8Array;
    session(rules: Rule | Rule[], initial: Hypergraph | Hypergraph[] | InitialConditionSpec,
            options?: EvolveOptions): Session;
    initialCondition(spec: InitialConditionSpec): Record<string, unknown> & { Edges: number[][] };
    heatResidual(parent: Hypergraph, child: Hypergraph,
                 options?: { field?: string; timeStep?: number; statisticsGraph?: string }): number;
}

/** True where the threaded module can run. */
export function threadsAvailable(): boolean;

/** Loads the engine: the threaded module where threadsAvailable() unless `threads` says. */
export function load(options?: { threads?: boolean | "auto" }): Promise<Engine>;
