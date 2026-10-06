#!/usr/bin/env python3
"""Per-function device stack frame sizes and the call graph, read from a built object.

WHAT IT READS. `cuobjdump -res-usage` reports STACK:0 for every function in a
relocatable object: under `-rdc=true` the ABI frame is laid out by nvlink at
device-link time, and nvlink declines to size an entry whose call graph has a
recursive cycle ("STACK:UNKNOWN" on the linked binary). The PTX carries the
per-function number anyway -- each body opens with

    .local .align N .b8 __local_depot<k>[BYTES];

which is that function's own frame: its explicit locals and its spills, before
the ABI's fixed per-call save area. So a depot sum over a chain is a LOWER BOUND
on the chain's true stack.

Usage:
    ptx_frame_sizes.py <file.cu.o | lib.a | file.ptx> [--calls | name-substring ...]

`--calls` reports every recursion cycle in the call graph (the functions in it
and its depot sum per level) and, per entry, the deepest depot sum over a call
chain that does not repeat a function. EngineState::kDeviceStackBytes has to
cover the deepest entry plus its cycles' levels.
"""

import os
import re
import subprocess
import sys

CUOBJDUMP = os.environ.get('CUOBJDUMP', '/usr/local/cuda/bin/cuobjdump')

FUNC = re.compile(r'^\s*(?:\.visible\s+|\.weak\s+)?\.(?:func|entry)\b')
DEPOT = re.compile(r'\.local\s+\.align\s+(\d+)\s+\.b8\s+__local_depot\d+\[(\d+)\]')

HEADER = re.compile(r'^\s*(?:\.visible\s+|\.weak\s+|\.extern\s+)*\.(?:func|entry)\s+'
                    r'(?:\([^)]*\)\s*)?([_A-Za-z$][_A-Za-z0-9$]*)')


def header_name(line):
    """The function's name on a PTX `.func`/`.entry` header: the first identifier after the
    directive and the optional return parameter `(.param .b32 func_retval0)`. The argument
    list may follow on the same line or the next."""
    m = HEADER.match(line)
    return m.group(1) if m else None


def ptx_lines(path):
    """PTX text for a .ptx file, or extracted from an object via cuobjdump."""
    if path.endswith('.ptx'):
        with open(path, 'r', errors='replace') as fh:
            yield from fh
        return
    proc = subprocess.run([CUOBJDUMP, '-ptx', path], capture_output=True, text=True,
                          errors='replace')
    if proc.returncode != 0:
        sys.exit(f'{CUOBJDUMP} -ptx {path} failed:\n{proc.stderr}')
    yield from proc.stdout.splitlines(keepends=True)


def demangle(names):
    """Map mangled -> demangled in one c++filt call; identity if unavailable."""
    try:
        out = subprocess.run(['c++filt'], input='\n'.join(names), capture_output=True,
                             text=True, check=True).stdout.splitlines()
        return dict(zip(names, out))
    except (OSError, subprocess.CalledProcessError):
        return {n: n for n in names}


def parse(lines):
    """Yield (mangled_name, frame_bytes) for every function that declares a depot."""
    pending = None          # name from the most recent function header
    for line in lines:
        if FUNC.match(line):
            pending = header_name(line)
            continue
        if pending is None:
            continue
        d = DEPOT.search(line)
        if d:
            yield pending, int(d.group(2))
            pending = None
        elif line.startswith('}'):
            pending = None


def collect(path):
    frames = {}
    for name, size in parse(ptx_lines(path)):
        # A name can appear in several PTX sections (one per sm_ target); they
        # agree, and if they ever did not the larger is what has to be covered.
        frames[name] = max(frames.get(name, 0), size)
    pretty = demangle(sorted(frames))
    return {pretty[n]: frames[n] for n in sorted(frames)}


# A call spans lines: `call.uni`, an optional `(retval0),`, then the target, then the
# argument list. The target is a function name, or a register for an indirect call.
CALL_START = re.compile(r'^\s*call(?:\.uni)?\b(.*)$')
TARGET = re.compile(r'^\s*(?:\([^)]*\)\s*,\s*)?([%_A-Za-z$][_A-Za-z0-9$]*)')


def call_graph(lines):
    """(frames, callees, entries) over every function body in the PTX."""
    frames, callees, entries = {}, {}, set()
    cur, pending_call, depth, opened = None, None, 0, False
    for line in lines:
        # A body ends where its braces balance; each call sequence inside it is a `{ ... }`
        # block of its own, written at column 0.
        code = line.split('//', 1)[0]
        if cur is not None:
            depth += code.count('{') - code.count('}')
            opened = opened or '{' in code
            if opened and depth <= 0:
                cur, pending_call, depth, opened = None, None, 0, False
                continue
        if FUNC.match(line):
            depth, opened = 0, False
            cur = header_name(line)
            if cur:
                frames.setdefault(cur, 0)
                callees.setdefault(cur, set())
                if '.entry' in line:
                    entries.add(cur)
            continue
        if cur is None:
            continue
        d = DEPOT.search(line)
        if d:
            frames[cur] = max(frames[cur], int(d.group(2)))
            continue
        if pending_call is not None:
            pending_call += ' ' + line.strip()
            t = TARGET.match(pending_call)
            if t:
                callees[cur].add(t.group(1) if not t.group(1).startswith('%') else '<indirect>')
                pending_call = None
            continue
        c = CALL_START.match(line)
        if c:
            pending_call = c.group(1).strip()
            t = TARGET.match(pending_call)
            if t:
                callees[cur].add(t.group(1) if not t.group(1).startswith('%') else '<indirect>')
                pending_call = None
    return frames, callees, entries


def sccs(nodes, edges):
    """Strongly connected components (Tarjan, iterative)."""
    index, low, on, stack, out, n = {}, {}, set(), [], [], [0]
    for root in nodes:
        if root in index:
            continue
        work = [(root, iter(sorted(edges.get(root, ()))))]
        index[root] = low[root] = n[0]; n[0] += 1
        stack.append(root); on.add(root)
        while work:
            v, it = work[-1]
            w = next(it, None)
            if w is not None:
                if w not in index:
                    index[w] = low[w] = n[0]; n[0] += 1
                    stack.append(w); on.add(w)
                    work.append((w, iter(sorted(edges.get(w, ())))))
                elif w in on:
                    low[v] = min(low[v], index[w])
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[v])
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop(); on.discard(w); comp.append(w)
                    if w == v:
                        break
                out.append(comp)
    return out


def report_calls(path):
    frames, callees, entries = call_graph(ptx_lines(path))
    pretty = demangle(sorted(frames))
    comps = sccs(sorted(frames), callees)
    comp_of = {f: i for i, c in enumerate(comps) for f in c}
    cyclic = [c for c in comps if len(c) > 1 or c[0] in callees.get(c[0], ())]
    indirect = sorted(f for f, cs in callees.items() if '<indirect>' in cs)
    print(f'{len(indirect)} function(s) with an indirect call:')
    for f in indirect:
        print(f'          {pretty.get(f, f)[:150]}')
    print(f'{len(cyclic)} recursion cycle(s):')
    for c in cyclic:
        print(f'  {sum(frames.get(f, 0) for f in c):6d} bytes of depot per level over {len(c)} function(s):')
        for f in sorted(c, key=lambda f: -frames.get(f, 0)):
            print(f'  {frames.get(f, 0):6d}    {pretty.get(f, f)[:150]}')
    # Deepest chain over the condensation: a component costs its depot sum once.
    memo = {}

    def deepest(ci):
        if ci in memo:
            return memo[ci]
        memo[ci] = 0
        own = sum(frames.get(f, 0) for f in comps[ci])
        best = 0
        for f in comps[ci]:
            for g in callees.get(f, ()):
                if g in comp_of and comp_of[g] != ci:
                    best = max(best, deepest(comp_of[g]))
        memo[ci] = own + best
        return memo[ci]

    print()
    print('Deepest depot chain per entry (a cycle counted once), largest first:')
    rows = sorted(((deepest(comp_of[e]), e) for e in entries if e in comp_of), reverse=True)
    for size, e in rows:
        rec = any(comp_of[e] == comps.index(c) for c in cyclic) or \
            _reaches_cycle(comp_of[e], comps, comp_of, callees, cyclic)
        print(f'{size:6d}  {"recursive " if rec else ""}{pretty.get(e, e)[:140]}')


def _reaches_cycle(ci, comps, comp_of, callees, cyclic):
    cyc = {comps.index(c) for c in cyclic}
    seen, todo = set(), [ci]
    while todo:
        x = todo.pop()
        if x in seen:
            continue
        seen.add(x)
        if x in cyc:
            return True
        for f in comps[x]:
            for g in callees.get(f, ()):
                if g in comp_of:
                    todo.append(comp_of[g])
    return False


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path, rest = sys.argv[1], sys.argv[2:]
    if rest == ['--calls']:
        report_calls(path)
        return
    frames = collect(path)

    rows = sorted(((sz, lbl) for lbl, sz in frames.items()
                   if not rest or any(w in lbl for w in rest)), reverse=True)
    for size, label in rows:
        print(f'{size:6d}  {label}')
    print('-' * 6)
    print(f'{sum(s for s, _ in rows):6d}  TOTAL over {len(rows)} function(s) shown')


if __name__ == '__main__':
    main()
