#!/usr/bin/env python3
"""Random job-sequence fuzz of hg_evolve (CPU, one-shot) against one hg_evolve_gpu --serve worker.

Each case is a random rule set, initial states, step count and option set. Per case:

  (a) worker    The GPU worker, one process for the whole run, answers the full reply. Its
                fingerprint (counts, canonical state and event multisets, causal and branchial
                structure) must equal the CPU one-shot reply's at 4 threads. A mismatch is rerun
                in a fresh GPU process: if the fresh process agrees with the CPU, the finding is a
                worker leak and the job sequence is minimised; otherwise the case is minimised.
  (b) stats     When "StepStatistics" is requested, the CPU and GPU-worker StepStatistics must be
                equal. On the CPU, the full fingerprint and StepStatistics at 1 and 16 threads
                must equal the 4-thread ones.
  (c) counts    A counts-only request (RequestedData = Num* only) must give the Num* of the full
                reply, on the CPU and on the GPU worker.

Every child runs under RLIMIT_AS (--cpu-va-mb, --gpu-va-mb). GPU processes are serialised with a
flock on <out>/gpu.lock. The CPU worker count is set through an LD_PRELOAD shim that overrides
get_nprocs and sysconf; the tool compiles it into <out> with cc on first use.

A finding of a known class is recorded with the defect name and does not fail the run. An unknown
finding is shrunk (initial edges, rule edges, rules, options, steps, vertex merges) to a minimal
case, written to <out>/min_<seed>_<kind>.json and .wxf, and makes the exit status 1.

    python3 tools/dev/worker_fuzz.py                       # 200 cases from seed 1000
    python3 tools/dev/worker_fuzz.py --seed 5000 --cases 1000 --minutes 60
    python3 tools/dev/worker_fuzz.py --replay <out>/min_1234_worker.json
"""
import argparse, copy, fcntl, hashlib, json, os, random, resource, struct, subprocess, sys, threading, time, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wxf_mini import encode, decode, Sym, Fn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN_DIR = os.path.join(ROOT, "paclet", "LibraryResources", "Linux-x86-64")
NUMS = ["NumStates", "NumEvents", "NumCausalEdges", "NumBranchialEdges"]
REQ = ["States", "Events", "CausalEdges", "BranchialEdges"] + NUMS
KEYS = ["InputState", "OutputState", "Step", "Rule", "ConsumedEdges", "ProducedEdges"]
CAPS = ("MaxStatesPerStep", "MaxSuccessorStatesPerParent", "MatchesPerStateRule")

SHIM_C = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <unistd.h>
static int fuzz_nprocs(void) { const char* s = getenv("HG_FUZZ_NPROCS"); return s ? atoi(s) : 0; }
int get_nprocs(void) {
    int n = fuzz_nprocs(); if (n > 0) return n;
    int (*real)(void) = (int (*)(void))dlsym(RTLD_NEXT, "get_nprocs"); return real();
}
int get_nprocs_conf(void) {
    int n = fuzz_nprocs(); if (n > 0) return n;
    int (*real)(void) = (int (*)(void))dlsym(RTLD_NEXT, "get_nprocs_conf"); return real();
}
long sysconf(int name) {
    int n = fuzz_nprocs();
    if (n > 0 && (name == _SC_NPROCESSORS_ONLN || name == _SC_NPROCESSORS_CONF)) return n;
    long (*real)(int) = (long (*)(int))dlsym(RTLD_NEXT, "sysconf"); return real(name);
}
"""


class Cfg:
    cpu = gpu = out = shim = None
    cpu_va = 8000 << 20
    gpu_va = 16000 << 20
    cpu_timeout = 30.0
    gpu_timeout = 60.0


def build_shim(out):
    src, so = os.path.join(out, "nprocs_shim.c"), os.path.join(out, "nprocs_shim.so")
    if not os.path.exists(so):
        with open(src, "w") as f:
            f.write(SHIM_C)
        subprocess.run(["cc", "-shared", "-fPIC", "-O2", "-o", so, src, "-ldl"], check=True)
    return so


# ---------------------------------------------------------------- generation
def gen_rule(rng):
    nl = rng.choice([1, 1, 2, 2, 2, 3])
    pool = list(range(1, rng.randint(1, 5) + 1))
    lhs = [[rng.choice(pool) for _ in range(rng.choice([1, 2, 2, 2, 3, 3, 4]))] for _ in range(nl)]
    lv = sorted({v for e in lhs for v in e})
    fresh = list(range(max(lv) + 1, max(lv) + 1 + rng.randint(0, 2)))
    rhs = [[rng.choice(lv + fresh) for _ in range(rng.choice([1, 2, 2, 2, 3, 3, 4]))]
           for _ in range(rng.choice([0, 1, 1, 2, 2, 3, 3, 4]))]
    return [lhs, rhs]


def gen_state(rng, arities, large):
    n_edges = rng.randint(15, 50) if large else rng.randint(1, 6)
    nv = max(1, int(n_edges * rng.uniform(0.5, 1.5))) if large else rng.randint(1, 5)
    return [[rng.randint(1, nv) for _ in range(rng.choice(arities) if rng.random() < 0.85 else rng.randint(1, 4))]
            for _ in range(n_edges)]


def gen_options(rng):
    o = {"IncludeCanonicalHashes": Sym("True"), "RequestedData": list(REQ)}
    cs = rng.choice(["None", "Automatic", "Full", "Full"])
    o["CanonicalizeStates"] = Sym(cs)
    r = rng.random()
    if r < 0.3:
        o["CanonicalizeEvents"] = Sym("None")
    elif r < 0.5:
        o["CanonicalizeEvents"] = Sym("Full")
    elif r < 0.75:
        o["CanonicalizeEvents"] = Sym("Automatic")
    elif r < 0.95:
        o["CanonicalizeEvents"] = rng.sample(KEYS, rng.randint(1, 6))
    else:
        o["CanonicalizeEvents"] = "Positional"
    if rng.random() < (0.4 if cs == "Full" else 0.08):
        o["ExploreFromCanonicalStatesOnly"] = Sym("True")
    if rng.random() < 0.3:
        o["CausalTransitiveReduction"] = Sym("False")
    sampled = False
    if rng.random() < 0.2:
        o["MaxStatesPerStep"] = rng.randint(1, 20); sampled = True
    if rng.random() < 0.2:
        o["MaxSuccessorStatesPerParent"] = rng.randint(1, 5); sampled = True
    if rng.random() < 0.2:
        o["MatchesPerStateRule"] = rng.randint(1, 4); sampled = True
    if rng.random() < 0.12:
        o["TransitionRate"] = round(rng.uniform(0.1, 0.9), 3); sampled = True
    if rng.random() < 0.08:
        o["ExplorationProbability"] = round(rng.uniform(0.2, 0.9), 3); sampled = True
    if rng.random() < 0.05:
        o["UniformRandom"] = Sym("True"); o["MatchesPerStep"] = rng.randint(1, 20); sampled = True
    if sampled and rng.random() < 0.8:
        o["RandomSeed"] = rng.randint(0, 2**31)
    return o


def gen_case(seed):
    rng = random.Random(seed)
    rules = [gen_rule(rng) for _ in range(rng.choice([1, 1, 2, 2, 3]))]
    arities = [len(e) for r in rules for e in r[0]]
    large = rng.random() < 0.12
    inits = [gen_state(rng, arities, large) for _ in range(1 if rng.random() < 0.85 else rng.randint(2, 3))]
    steps = rng.randint(1, 2) if large else rng.randint(1, 5)
    o = gen_options(rng)
    if "TransitionRate" in o and rng.random() < 0.3:
        o["RuleWeights"] = [round(rng.uniform(0, 1), 2) for _ in rules]
    rng2 = random.Random(seed * 31 + 5)
    if o.get("CanonicalizeStates") == "Full" and rng2.random() < 0.35:
        o["ExploreFromCanonicalStatesOnly"] = Sym("True")
    if rng2.random() < 0.6:
        o["RequestedData"] = list(o["RequestedData"]) + ["StepStatistics"]
        o["StepStatisticsWeighting"] = rng2.choice(["States", "Classes"])
        o["StepStatisticsBranchial"] = rng2.choice([[], ["Graph"], ["Overlap"], ["Graph", "Overlap"]])
        if o.get("CanonicalizeStates") == "Full" and rng2.random() < 0.6:
            o["CanonicalizeStates"] = Sym(rng2.choice(["None", "Automatic"]))
            o.pop("ExploreFromCanonicalStatesOnly", None)
    return {"seed": seed, "rules": rules, "inits": inits, "steps": steps, "opts": o}


def job_bytes(case, opts=None):
    return encode({"InitialStates": case["inits"],
                   "Rules": {f"r{i}": Fn("Rule", r) for i, r in enumerate(case["rules"])},
                   "Steps": case["steps"], "Options": opts if opts is not None else case["opts"]})


def counts_only(opts):
    o = dict(opts)
    o["RequestedData"] = list(NUMS)
    return o


def to_json(v):
    if isinstance(v, dict):
        return {k: to_json(x) for k, x in v.items()}
    if isinstance(v, list):
        return [to_json(x) for x in v]
    if isinstance(v, Sym):
        return "sym:" + str(v)
    return v


def from_json(v):
    if isinstance(v, dict):
        return {k: from_json(x) for k, x in v.items()}
    if isinstance(v, list):
        return [from_json(x) for x in v]
    if isinstance(v, str) and v.startswith("sym:"):
        return Sym(v[4:])
    return v


# ---------------------------------------------------------------- processes
def _limit(va):
    def f():
        resource.setrlimit(resource.RLIMIT_AS, (va, va))
        os.nice(10)
    return f


class GpuLock:
    """Exclusive across processes, counted within this one: the run's worker holds it while a
    fresh worker for a rerun takes it again."""
    depth = 0
    f = None

    def __enter__(self):
        if GpuLock.depth == 0:
            GpuLock.f = open(os.path.join(Cfg.out, "gpu.lock"), "w")
            fcntl.flock(GpuLock.f, fcntl.LOCK_EX)
        GpuLock.depth += 1

    def __exit__(self, *a):
        GpuLock.depth -= 1
        if GpuLock.depth == 0:
            fcntl.flock(GpuLock.f, fcntl.LOCK_UN)
            GpuLock.f.close()


def run_cpu(payload, threads):
    env = dict(os.environ, HG_FUZZ_NPROCS=str(threads), LD_PRELOAD=Cfg.shim)
    t0 = time.time()
    try:
        p = subprocess.run([Cfg.cpu], input=payload, capture_output=True, env=env,
                           timeout=Cfg.cpu_timeout, preexec_fn=_limit(Cfg.cpu_va))
    except subprocess.TimeoutExpired:
        return {"rc": "timeout", "time": time.time() - t0}
    res = {"rc": p.returncode, "time": time.time() - t0, "err": p.stderr.decode(errors="replace")[-1500:]}
    if p.returncode == 0:
        try:
            res["reply"] = decode(p.stdout)
        except Exception as e:
            res["rc"] = "bad reply"; res["err"] = repr(e)
    return res


def read_exact(f, k):
    buf = b""
    while len(buf) < k:
        c = f.read(k - len(buf))
        if not c:
            raise EOFError("worker exited")
        buf += c
    return buf


class Worker:
    """One hg_evolve_gpu --serve process. Frames are u64-length-prefixed; bit 63 marks a progress
    frame and bit 62 an error frame."""

    def __init__(self, tag):
        self.errf = open(os.path.join(Cfg.out, f"gpu_worker_{tag}.err"), "ab")
        self.lock = GpuLock()
        self.lock.__enter__()
        self.p = subprocess.Popen([Cfg.gpu, "--serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.errf, preexec_fn=_limit(Cfg.gpu_va))
        self.history = []

    def run(self, payload):
        self.history.append(payload)
        timer = threading.Timer(Cfg.gpu_timeout, self.p.kill)
        timer.start()
        t0 = time.time()
        try:
            self.p.stdin.write(struct.pack("<Q", len(payload)) + payload)
            self.p.stdin.flush()
            while True:
                n = struct.unpack("<Q", read_exact(self.p.stdout, 8))[0]
                if n & (1 << 63):
                    read_exact(self.p.stdout, n & ~(1 << 63))
                    continue
                if n & (1 << 62):
                    msg = read_exact(self.p.stdout, n & ~(1 << 62)).decode(errors="replace")
                    return {"rc": "error_frame", "err": msg[-1500:], "time": time.time() - t0}
                break
            return {"rc": 0, "reply": decode(read_exact(self.p.stdout, n)), "time": time.time() - t0}
        except Exception as e:
            return {"rc": "dead", "err": repr(e), "time": time.time() - t0, "poll": self.p.poll()}
        finally:
            timer.cancel()

    def rss_mb(self):
        try:
            for line in open(f"/proc/{self.p.pid}/status"):
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
        except OSError:
            pass
        return None

    def alive(self):
        return self.p.poll() is None

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=20)
        except Exception:
            self.p.kill()
            self.p.wait()
        self.lock.__exit__()
        self.errf.close()


def run_gpu_fresh(payloads):
    """Run payloads in order in a new worker; return the last reply's result."""
    w = Worker("fresh")
    try:
        r = None
        for b in payloads:
            r = w.run(b)
            if r["rc"] == "dead":
                break
        return r
    finally:
        w.close()


# ---------------------------------------------------------------- fingerprint
def h(x):
    return hashlib.blake2b(repr(x).encode(), digest_size=8).hexdigest()


def fingerprint(reply, unjoin=False):
    fp = {k: reply.get(k) for k in NUMS}
    states = copy.deepcopy(reply.get("States", {}) or {})
    for s in states.values():
        if not s.get("Edges"):
            s["CanonicalHash"] = 1  # F6: the empty state's hash differs by mode
    shash = {sid: s.get("CanonicalHash") for sid, s in states.items()}
    fp["states"] = h(sorted((s.get("CanonicalHash"), len(s.get("Edges", []))) for s in states.values()))
    fp["n_states_listed"] = len(states)
    fp["state_steps"] = h(sorted((s.get("CanonicalHash"), s.get("Step")) for s in states.values()))
    events = reply.get("Events", {}) or {}
    lab, members, missing = {}, {}, 0
    for eid, e in events.items():
        ih, oh = shash.get(e.get("CanonicalInputState")), shash.get(e.get("CanonicalOutputState"))
        if ih is None or oh is None:
            missing += 1
        lab[eid] = repr((e.get("RuleIndex"), ih, oh))
        members.setdefault(e.get("CanonicalId"), []).append(lab[eid])
    canon_lab = {c: h(sorted(ls)) for c, ls in members.items()}
    fp["events"] = h(sorted(lab.values()))
    fp["n_events_listed"] = len(events)
    fp["event_state_missing"] = missing

    def wl(nodes_lab, pairs, directed):
        col = dict(nodes_lab)
        for a, b in pairs:
            col.setdefault(a, "?"); col.setdefault(b, "?")
        outn, inn = {}, {}
        for a, b in pairs:
            outn.setdefault(a, []).append(b); inn.setdefault(b, []).append(a)
        for _ in range(3):
            if directed:
                col = {v: h((c, sorted(col[x] for x in outn.get(v, [])), sorted(col[x] for x in inn.get(v, []))))
                       for v, c in col.items()}
            else:
                col = {v: h((c, sorted([col[x] for x in outn.get(v, [])] + [col[x] for x in inn.get(v, [])])))
                       for v, c in col.items()}
        ep = sorted((col[a], col[b]) if directed else tuple(sorted((col[a], col[b]))) for a, b in pairs)
        return h((sorted(col.values()), ep))

    ce = reply.get("CausalEdges", []) or []
    pairs = [(c.get("RawFrom", c.get("From")), c.get("RawTo", c.get("To"))) for c in ce]
    unknown = sum(1 for a, b in pairs if a not in lab or b not in lab)
    fp["causal"] = wl({e: lab[e] for p in pairs for e in p}, pairs, True) if not unknown and not unjoin else None
    fp["causal_struct"] = wl({}, pairs, True)
    fp["n_causal_listed"] = len(ce)
    fp["causal_unknown_endpoint"] = unknown
    be = reply.get("BranchialEdges", []) or []
    bpairs = [(c.get("From"), c.get("To")) for c in be]
    bunk = sum(1 for a, b in bpairs if a not in canon_lab or b not in canon_lab)
    fp["branchial"] = wl({e: canon_lab[e] for p in bpairs for e in p}, bpairs, False) if not bunk and not unjoin else None
    fp["branchial_struct"] = wl({}, bpairs, False)
    fp["n_branchial_listed"] = len(be)
    fp["branchial_unknown_endpoint"] = bunk
    fp["warnings"] = sorted((w.get("Kind"), w.get("Count")) for w in reply.get("Warnings", []) or [])
    return fp


MODE_DEP = {"causal_unknown_endpoint", "branchial_unknown_endpoint", "event_state_missing"}


def unjoinable(opts):
    return (opts.get("CanonicalizeStates") == "Full" and opts.get("CanonicalizeEvents") == "Automatic"
            and opts.get("ExploreFromCanonicalStatesOnly") != "True")


def gpu_ignore(opts):
    """Fields a GPU reply is known to differ on (F2: Automatic state listing; F3: branchial under
    an event identity); counts are still compared."""
    ig = ("warnings",)
    if opts.get("CanonicalizeStates") == "Automatic":
        ig += ("states", "n_states_listed", "state_steps")
    if opts.get("CanonicalizeEvents", "None") != "None":
        ig += ("branchial", "branchial_struct")
    return ig


def diff_fp(a, b, ignore=()):
    out = []
    for k in set(a) | set(b):
        if k in ignore or k in MODE_DEP or a.get(k) == b.get(k):
            continue
        if k in ("causal", "branchial") and (a.get(k) is None or b.get(k) is None):
            continue
        out.append(k)
    return sorted(out)


def _has_cap(opts):
    return any(c in opts for c in CAPS) or ("UniformRandom" in opts and "MatchesPerStep" in opts)


def _recon(opts):
    return opts.get("CanonicalizeStates") == "Full" and (
        opts.get("ExploreFromCanonicalStatesOnly") == "True" or opts.get("CanonicalizeEvents") == "Automatic")


# Open defects on the RC3 board, by name, with the case shape each one produces. A finding that
# matches one is recorded under that name and does not fail the run; a crash, a dead worker and a
# worker leak never match. Remove a row when its defect closes.
KNOWN = [
    ("A2/B1: a cap under the quotient reconstruction depends on the schedule (F5)",
     lambda opts, kind, fields: _recon(opts) and _has_cap(opts)),
]


def known_class(opts, kind, fields=None):
    """The KNOWN name for a finding, or None."""
    if kind.startswith("crash") or kind in ("error", "worker_died", "worker_leak"):
        return None
    for name, pred in KNOWN:
        if pred(opts, kind, fields):
            return name
    return None


# ---------------------------------------------------------------- checks
def stats_of(r):
    return repr(r["reply"].get("StepStatistics")) if r.get("rc") == 0 else None


def check_case(case, worker, budget_s):
    """Run the three classes on one case. Returns (log, findings); each finding names the check it
    failed so the shrinker can re-run that check alone."""
    log, out = {"seed": case["seed"]}, []
    while True:
        c4 = run_cpu(job_bytes(case), 4)
        if c4["rc"] == "timeout" or (c4["rc"] == 0 and c4["time"] > budget_s) or "bad_alloc" in c4.get("err", ""):
            if case["steps"] > 1:
                case["steps"] -= 1
                continue
            log["skipped"] = "slow at 1 step"
            return log, out
        break
    log["steps"] = case["steps"]
    opts = case["opts"]
    if c4["rc"] not in (0,) and isinstance(c4["rc"], int) and c4["rc"] < 0:
        out.append({"kind": "crash_cpu", "check": "cpu4", "rc": c4["rc"], "err": c4.get("err", "")[-500:]})
    g = worker.run(job_bytes(case))
    log["counts"] = [c4["reply"].get(k) for k in NUMS] if c4["rc"] == 0 else None
    if g["rc"] == "dead":
        out.append({"kind": "worker_died", "check": "worker", "res": {k: g.get(k) for k in ("err", "poll")}})
        return log, out
    if c4["rc"] == 0 and g["rc"] == "error_frame":
        f = run_gpu_fresh([job_bytes(case)])
        if f and f["rc"] == 0:
            out.append({"kind": "worker_leak", "check": "leak", "what": "GPU worker error, fresh GPU and CPU ok",
                        "gpu": g["err"][-400:], "history_len": len(worker.history)})
        else:
            out.append({"kind": "error", "check": "worker", "what": "GPU error, CPU ok", "gpu": g["err"][-400:]})
    elif c4["rc"] != 0 and g["rc"] == 0:
        out.append({"kind": "error", "check": "worker", "what": "CPU error, GPU ok", "cpu": str(c4.get("err"))[-400:]})
    if c4["rc"] != 0 or g["rc"] != 0:
        return log, out
    cr, gr = c4["reply"], g["reply"]
    uj = unjoinable(opts)
    fc = fingerprint(cr, uj)
    # (a)
    d = diff_fp(fc, fingerprint(gr, uj), ignore=gpu_ignore(opts))
    if d:
        f = run_gpu_fresh([job_bytes(case)])
        fresh_d = diff_fp(fc, fingerprint(f["reply"], uj), ignore=gpu_ignore(opts)) if f and f["rc"] == 0 else ["error"]
        if fresh_d:
            out.append({"kind": "worker", "check": "gpu", "fields": d,
                        "gpu_warnings": [x.get("Kind") for x in gr.get("Warnings", []) or []]})
        else:
            out.append({"kind": "worker_leak", "check": "leak", "fields": d, "history_len": len(worker.history)})
    # (b)
    has_ss = "StepStatistics" in opts["RequestedData"]
    if has_ss and stats_of(c4) != stats_of(g):
        out.append({"kind": "stats_gpu", "check": "stats_gpu", "cpu": stats_of(c4)[:800], "gpu": stats_of(g)[:800]})
    for th in (1, 16):
        r = run_cpu(job_bytes(case), th)
        if r["rc"] != 0:
            out.append({"kind": "threads", "check": f"cpu{th}", "what": f"error at {th} threads", "err": str(r.get("err"))[-400:]})
            continue
        d = diff_fp(fc, fingerprint(r["reply"], uj), ignore=("warnings",))
        if d:
            out.append({"kind": "threads", "check": f"cpu{th}", "fields": d})
        if has_ss and stats_of(r) != stats_of(c4):
            out.append({"kind": "stats_threads", "check": f"ss{th}"})
    # (c)
    co = job_bytes(case, counts_only(opts))
    cc = run_cpu(co, 4)
    gc = worker.run(co)
    if gc["rc"] == "dead":
        out.append({"kind": "worker_died", "check": "worker", "res": {k: gc.get(k) for k in ("err", "poll")}})
    for name, full, part in (("cpu", cr, cc), ("gpu", gr, gc)):
        if part.get("rc") != 0:
            if part.get("rc") != "dead":
                out.append({"kind": "counts", "check": f"counts_{name}", "what": "counts-only error", "err": str(part.get("err"))[-400:]})
            continue
        a, b = [full.get(k) for k in NUMS], [part["reply"].get(k) for k in NUMS]
        if a != b:
            out.append({"kind": "counts", "check": f"counts_{name}", "full": a, "counts_only": b})
    return log, out


def still_fails(case, check):
    """Re-run one check on a candidate case with fresh processes."""
    try:
        opts = case["opts"]
        c4 = run_cpu(job_bytes(case), 4)
        if check == "cpu4":
            return isinstance(c4["rc"], int) and c4["rc"] < 0
        if check == "worker":
            g = run_gpu_fresh([job_bytes(case)])
            return bool(g) and g["rc"] in (0, "error_frame") and (c4["rc"] == 0) != (g["rc"] == 0)
        if c4["rc"] != 0:
            return False
        uj = unjoinable(opts)
        fc = fingerprint(c4["reply"], uj)
        if check in ("gpu", "stats_gpu", "counts_gpu"):
            payloads = [job_bytes(case)] + ([job_bytes(case, counts_only(opts))] if check == "counts_gpu" else [])
            g = run_gpu_fresh(payloads)
            if check == "counts_gpu":
                g0 = run_gpu_fresh([job_bytes(case)])
                return (g and g["rc"] == 0 and g0 and g0["rc"] == 0 and
                        [g0["reply"].get(k) for k in NUMS] != [g["reply"].get(k) for k in NUMS])
            if not g or g["rc"] != 0:
                return False
            if check == "stats_gpu":
                return stats_of(c4) != stats_of(g)
            return bool(diff_fp(fc, fingerprint(g["reply"], uj), ignore=gpu_ignore(opts)))
        if check.startswith("cpu") or check.startswith("ss"):
            r = run_cpu(job_bytes(case), int(check[3:] if check.startswith("cpu") else check[2:]))
            if r["rc"] != 0:
                return check.startswith("cpu")
            if check.startswith("ss"):
                return stats_of(r) != stats_of(c4)
            return bool(diff_fp(fc, fingerprint(r["reply"], uj), ignore=("warnings",)))
        if check == "counts_cpu":
            cc = run_cpu(job_bytes(case, counts_only(opts)), 4)
            return cc["rc"] != 0 or [c4["reply"].get(k) for k in NUMS] != [cc["reply"].get(k) for k in NUMS]
    except Exception:
        return False
    return False


def candidates(c):
    if c["steps"] > 1:
        d = copy.deepcopy(c); d["steps"] -= 1; yield d
    for i in range(len(c["inits"])) if len(c["inits"]) > 1 else ():
        d = copy.deepcopy(c); del d["inits"][i]; yield d
    for si, st in enumerate(c["inits"]):
        for ei in range(len(st)) if len(st) > 1 else ():
            d = copy.deepcopy(c); del d["inits"][si][ei]; yield d
    for i in range(len(c["rules"])) if len(c["rules"]) > 1 else ():
        d = copy.deepcopy(c); del d["rules"][i]; d["opts"].pop("RuleWeights", None); yield d
    for ri, (lhs, rhs) in enumerate(c["rules"]):
        for ei in range(len(rhs)):
            d = copy.deepcopy(c); del d["rules"][ri][1][ei]; yield d
        for ei in range(len(lhs)) if len(lhs) > 1 else ():
            d = copy.deepcopy(c); del d["rules"][ri][0][ei]
            lv = {v for e in d["rules"][ri][0] for v in e}
            if all(v in lv or v > max(lv) for e in d["rules"][ri][1] for v in e):
                yield d
    for k in list(c["opts"]):
        if k not in ("IncludeCanonicalHashes", "RequestedData"):
            d = copy.deepcopy(c); del d["opts"][k]; yield d
    for si, st in enumerate(c["inits"]):
        for ei, e in enumerate(st):
            if len(e) > 1:
                d = copy.deepcopy(c); d["inits"][si][ei] = e[:-1]; yield d
    vs = sorted({v for st in c["inits"] for e in st for v in e})
    for v in vs[1:]:
        d = copy.deepcopy(c)
        d["inits"] = [[[vs[0] if x == v else x for x in e] for e in st] for st in d["inits"]]
        yield d


def shrink(case, check, seconds):
    t_end = time.time() + seconds
    if not still_fails(case, check):
        return case, False
    changed = True
    while changed and time.time() < t_end:
        changed = False
        for d in candidates(case):
            if time.time() > t_end:
                break
            if still_fails(d, check):
                case, changed = d, True
                break
    return case, True


def shrink_history(history, seconds):
    """Smallest job subsequence, ending with the last job, whose last reply in one fresh worker
    differs from the last job's reply in another fresh worker."""
    t_end = time.time() + seconds

    def fails(seq):
        a, b = run_gpu_fresh(seq), run_gpu_fresh(seq[-1:])
        if not a or a["rc"] != 0 or not b or b["rc"] != 0:
            return a is None or a.get("rc") != (b or {}).get("rc")
        return repr(a["reply"]) != repr(b["reply"])
    seq = list(history)
    if not fails(seq):
        return seq, False
    chunk = max(1, (len(seq) - 1) // 2)
    while chunk >= 1 and time.time() < t_end:
        i, progress = 0, False
        while i < len(seq) - 1 and time.time() < t_end:
            trial = seq[:i] + seq[min(i + chunk, len(seq) - 1):]
            if len(trial) < len(seq) and fails(trial):
                seq, progress = trial, True
            else:
                i += chunk
        if not progress:
            chunk //= 2
    return seq, True


def write_min(seed, kind, case):
    base = os.path.join(Cfg.out, f"min_{seed}_{kind}")
    with open(base + ".json", "w") as f:
        json.dump(to_json(case), f)
    with open(base + ".wxf", "wb") as f:
        f.write(job_bytes(case))
    return base


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument("--minutes", type=float, default=0, help="stop starting cases after this long (0: no limit)")
    ap.add_argument("--cpu", default=os.path.join(BIN_DIR, "hg_evolve"))
    ap.add_argument("--gpu", default=os.path.join(BIN_DIR, "hg_evolve_gpu"))
    ap.add_argument("--out", default=os.path.join(ROOT, ".scratch", "worker_fuzz"))
    ap.add_argument("--cpu-va-mb", type=int, default=8000)
    ap.add_argument("--gpu-va-mb", type=int, default=16000)
    ap.add_argument("--case-seconds", type=float, default=4.0, help="steps are cut until the CPU run fits")
    ap.add_argument("--shrink-seconds", type=float, default=300)
    ap.add_argument("--max-shrinks", type=int, default=3, help="minimised findings per (kind, check)")
    ap.add_argument("--replay", help="re-run the checks on one case JSON and print the findings")
    a = ap.parse_args()
    Cfg.cpu, Cfg.gpu, Cfg.out = os.path.abspath(a.cpu), os.path.abspath(a.gpu), os.path.abspath(a.out)
    Cfg.cpu_va, Cfg.gpu_va = a.cpu_va_mb << 20, a.gpu_va_mb << 20
    os.makedirs(Cfg.out, exist_ok=True)
    for b in (Cfg.cpu, Cfg.gpu):
        if not os.access(b, os.X_OK):
            sys.exit(f"worker_fuzz: no executable {b}")
    Cfg.shim = build_shim(Cfg.out)

    if a.replay:
        case = from_json(json.load(open(a.replay)))
        w = Worker("replay")
        try:
            log, fs = check_case(case, w, 1e9)
        finally:
            w.close()
        print(json.dumps({"log": log, "findings": fs}, default=str, indent=1))
        return 1 if fs else 0

    deadline = time.time() + a.minutes * 60 if a.minutes else 0
    res_f = open(os.path.join(Cfg.out, "results.jsonl"), "a")
    fnd_f = open(os.path.join(Cfg.out, "findings.jsonl"), "a")
    tag = f"s{a.seed}"
    w = Worker(tag)
    shrunk = {}
    tally = {"cases": 0, "skipped": 0, "a_worker": 0, "b_stats": 0, "c_counts": 0, "known": 0, "unknown": 0,
             "worker_restarts": 0}
    try:
        for seed in range(a.seed, a.seed + a.cases):
            if deadline and time.time() > deadline:
                break
            case = gen_case(seed)
            try:
                log, fs = check_case(case, w, a.case_seconds)
            except Exception:
                log, fs = {"seed": seed, "harness_error": traceback.format_exc()[-800:]}, []
            tally["cases"] += 1
            if "skipped" in log:
                tally["skipped"] += 1
            elif "counts" in log and log["counts"] is not None:
                tally["a_worker"] += 1
                tally["c_counts"] += 1
                tally["b_stats"] += "StepStatistics" in case["opts"]["RequestedData"]
            for fd in fs:
                fd["seed"] = seed
                fd["known"] = known_class(case["opts"], fd["kind"], fd.get("fields"))
                fd["case"] = to_json(case)
                if fd["known"]:
                    tally["known"] += 1
                else:
                    tally["unknown"] += 1
                    sig = (fd["kind"], fd.get("check"))
                    shrunk[sig] = shrunk.get(sig, 0) + 1
                    if shrunk[sig] > a.max_shrinks:
                        fd["minimal"] = None
                    elif fd["kind"] == "worker_leak":
                        seq, ok = shrink_history(w.history, a.shrink_seconds)
                        fd["minimal_sequence_jobs"] = len(seq)
                        base = os.path.join(Cfg.out, f"min_{seed}_worker_leak")
                        os.makedirs(base, exist_ok=True)
                        for i, b in enumerate(seq):
                            open(os.path.join(base, f"{i:04d}.wxf"), "wb").write(b)
                        fd["minimal"] = base
                    elif fd["kind"] != "worker_died":
                        small, ok = shrink(copy.deepcopy(case), fd["check"], a.shrink_seconds)
                        fd["reproduced"] = ok
                        fd["minimal"] = write_min(seed, fd["check"], small)
                        fd["minimal_case"] = to_json(small)
                fnd_f.write(json.dumps(fd, default=str) + "\n")
                fnd_f.flush()
            log["case"] = to_json(case)
            log["findings"] = [(f["kind"], f.get("check"), f.get("known")) for f in fs]
            res_f.write(json.dumps(log, default=str) + "\n")
            res_f.flush()
            print(f"seed {seed} steps {log.get('steps')} counts {log.get('counts')} "
                  f"findings {log['findings']} worker_jobs {len(w.history)} worker_rss_MB {w.rss_mb()}", flush=True)
            if not w.alive():
                w.close()
                w = Worker(tag)
                tally["worker_restarts"] += 1
    finally:
        w.close()
    print("DONE", json.dumps(tally), flush=True)
    return 1 if tally["unknown"] else 0


if __name__ == "__main__":
    sys.exit(main())
