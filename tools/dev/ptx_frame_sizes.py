#!/usr/bin/env python3
"""Per-function device stack frame sizes, read from a built object.

WHY THIS EXISTS. `EngineState::kDeviceStackBytesPerDepth` is the bytes one level
of the reconstruction replay's recursion cycle costs, and the replay bounds its
own depth from it so a deep run returns a partial result instead of faulting. It
is a MEASURED constant, and the method recorded beside it -- run until a 32 KB
stack faults, run until a 64 KB stack faults, divide -- needs two faulting GPU
runs and tells you nothing about WHICH frame grew. This reads the same quantity
out of an object that is already built, with no GPU and no run.

WHAT IT READS. `cuobjdump -res-usage` reports STACK:0 for every function in a
relocatable object: under `-rdc=true` the ABI frame is laid out by nvlink at
device-link time, and nvlink declines to size a recursive cycle at all ("stack
size for entry function ... cannot be statically determined"). The PTX carries
the per-function number anyway -- each body opens with

    .local .align N .b8 __local_depot<k>[BYTES];

which is that function's own frame: its explicit locals and its spills, before
the ABI's fixed per-call save area. So the depot sum over a cycle is a LOWER
BOUND on the cycle's true per-level cost, and the difference between that bound
and the measured constant is the ABI overhead of the frames in it. Both terms
matter: a change that adds bytes moves the first, and a change that adds a CALL
moves the second.

Usage:
    ptx_frame_sizes.py <file.cu.o | file.ptx> [--cycle | name-substring ...]

`--cycle` reports the reconstruction replay's recursion cycle and its depot sum,
which is the number `kDeviceStackBytesPerDepth` has to cover.
"""

import os
import re
import subprocess
import sys

CUOBJDUMP = os.environ.get('CUOBJDUMP', '/usr/local/cuda/bin/cuobjdump')

FUNC = re.compile(r'^\s*(?:\.visible\s+|\.weak\s+)?\.(?:func|entry)\b')
NAME = re.compile(r'([_A-Za-z$][_A-Za-z0-9$]*)\s*\(')
DEPOT = re.compile(r'\.local\s+\.align\s+(\d+)\s+\.b8\s+__local_depot\d+\[(\d+)\]')

# The replay's recursion cycle, in call order, as (label, match-mode, pattern).
# A demangled lambda carries its enclosing function's whole signature, so
# `qe_drive_instance(` appears inside three different frames' names -- hence the
# explicit mode per entry, and one frame binding to at most one slot below.
# qe_add_instance is deliberately absent: nvcc inlines it completely, so it is in
# no PTX and costs no frame.
CYCLE = [
    ('qe_drive_instance',      'func',   'hg_gpu::qe_drive_instance('),
    ('qe_for_each_match_from', 'in',     'qe_for_each_match_from<'),
    ('its match lambda',       'suffix', '::operator()(hg_gpu::DeviceSlotMatch const&) const'),
    ('qe_apply',               'func',   'hg_gpu::qe_apply('),
    ('qr_apply',               'in',     'hgcommon::qr_apply<'),
    ('DeviceQrCtx::descend',   'in',     'DeviceQrCtx::descend('),
]


def matches(label, mode, pattern):
    # 'func' is the function ITSELF and not a lambda inside it: a demangled lambda
    # opens with its enclosing function's entire signature, so a plain prefix test
    # binds `qe_drive_instance` to its own match lambda.
    if mode == 'func':
        return label.startswith(pattern) and '{lambda' not in label
    if mode == 'suffix':
        return label.endswith(pattern)
    return pattern in label


def header_name(line):
    """The function's name on a PTX `.func`/`.entry` header.

    A header reads `.weak .func (.param .b32 func_retval0) _ZN...(`, so the FIRST
    identifier-before-paren is the return parameter's `func`, not the function.
    The name is the LAST such identifier on the line.
    """
    found = NAME.findall(line)
    return found[-1] if found else None


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
        # A header whose name spilled onto its own line: take it over the `func`
        # the return-parameter contributed.
        if line.lstrip().startswith('_Z') and '(' in line:
            name = header_name(line)
            if name:
                pending = name
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


def report_cycle(frames):
    """The replay cycle in call order, with the depot sum that bounds a level."""
    print('Reconstruction replay recursion cycle, in call order:')
    print()
    total, present, claimed = 0, 0, set()
    for name, mode, pattern in CYCLE:
        hit = [(lbl, sz) for lbl, sz in frames.items()
               if lbl not in claimed and matches(lbl, mode, pattern)]
        if not hit:
            print(f'{"--":>6}  {name}   NOT PRESENT (inlined away, or not in this object)')
            continue
        # A frame binds to at most one slot, so a cycle whose shape changed shows
        # up as a missing slot rather than as a silently doubled sum.
        lbl, sz = max(hit, key=lambda kv: kv[1])
        claimed.add(lbl)
        total += sz
        present += 1
        print(f'{sz:6d}  {name}')
        print(f'{"":6}    {lbl}')
    print('-' * 6)
    print(f'{total:6d}  depot sum over the cycle -- a LOWER BOUND on the per-level cost,')
    print(f'{"":6}  excluding the ABI save area of the {present} frames in it.')
    print()
    print('Compare against EngineState::kDeviceStackBytesPerDepth. A change that raises')
    print('this sum, or that adds a frame to the cycle, invalidates that constant.')
    return total


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path, rest = sys.argv[1], sys.argv[2:]
    frames = collect(path)

    if rest == ['--cycle']:
        report_cycle(frames)
        return

    rows = sorted(((sz, lbl) for lbl, sz in frames.items()
                   if not rest or any(w in lbl for w in rest)), reverse=True)
    for size, label in rows:
        print(f'{size:6d}  {label}')
    print('-' * 6)
    print(f'{sum(s for s, _ in rows):6d}  TOTAL over {len(rows)} function(s) shown')


if __name__ == '__main__':
    main()
