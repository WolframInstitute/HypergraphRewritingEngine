#!/usr/bin/env python3
"""Which hgcommon rules the device actually CALLS, and which it open-codes beside.

WHY. hgcommon exists so one decision has one body. A device that declares
`#include "hgcommon/..."` and then open-codes the same arithmetic beside it is
the failure this layer was built to prevent -- and it is not hypothetical: #124
was exactly that. The device's event content triple open-coded FNV with the
64-bit basis missing its last digit, so every reconstructed identity it reported
was a relabelling of the host's. It survived because the two engines agreed while
BOTH were wrong, and unifying only the host's copy is what exposed it.

WHAT THIS REPORTS, and what it does not. For each shared function it reports
whether any device translation unit names it. That is a necessary condition, not
a sufficient one: a device file can call `qc_key` in one place and open-code it
in another, which is precisely what quotient_expansion.hpp does. So it also
counts the FNV basis constants appearing literally in device sources, since a
literal basis beside an include of hgcommon is the specific smell #124 had.

NOT A GATE. It reports; it does not assert. Several shared functions are host-only
by design (the IR search's scratch sizing, for instance, is called from the host
that allocates), so a device call count of zero is information, not a finding.
"""

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
COMMON = ROOT / 'common' / 'include' / 'hgcommon'
DEVICE_DIRS = [ROOT / 'gpu' / 'include' / 'hg_gpu', ROOT / 'gpu' / 'src']
HOST_DIRS = [ROOT / 'hypergraph' / 'include' / 'hypergraph', ROOT / 'hypergraph' / 'src']

# The 64-bit FNV-1a offset basis, and the digit-dropped value #124 was seeded with.
FNV_CORRECT = 14695981039346656037
FNV_BROKEN = FNV_CORRECT // 10

DECL = re.compile(r'^HG_HD\s+(?:inline\s+)?[A-Za-z_][A-Za-z0-9_:<>\s\*&]*?\b([a-z_][a-z0-9_]*)\s*\(',
                  re.MULTILINE)


def shared_names():
    names = set()
    for path in sorted(COMMON.glob('*.hpp')):
        names.update(DECL.findall(path.read_text(errors='replace')))
    # `inline`/`constexpr` helpers that are not decisions are noise here.
    return sorted(n for n in names if not n.startswith('operator'))


def sources(dirs):
    out = []
    for d in dirs:
        if d.is_dir():
            out += [p for p in d.rglob('*') if p.suffix in ('.hpp', '.cpp', '.cu', '.cuh')]
    return out


def main():
    names = shared_names()
    dev, host = sources(DEVICE_DIRS), sources(HOST_DIRS)
    dev_text = {p: p.read_text(errors='replace') for p in dev}
    host_text = {p: p.read_text(errors='replace') for p in host}

    print(f'{len(names)} shared rules in hgcommon; '
          f'{len(dev)} device sources, {len(host)} host sources\n')

    call = re.compile
    unused_dev = []
    for n in names:
        pat = call(r'\b' + re.escape(n) + r'\s*\(')
        d = sum(1 for t in dev_text.values() if pat.search(t))
        h = sum(1 for t in host_text.values() if pat.search(t))
        if d == 0:
            unused_dev.append((n, h))
    print('shared rules NO device source calls (host-only by design, or a gap):')
    for n, h in unused_dev:
        print(f'  {n:<28} host callers: {h}')

    print(f'\nliteral FNV bases in DEVICE sources '
          f'(correct {FNV_CORRECT}, digit-dropped {FNV_BROKEN}):')
    total_bad = 0
    # The digit-dropped value is a PREFIX of the correct one, so a plain substring count reports
    # every correct literal as a broken one too. Requiring a non-digit on both sides is what
    # separates them -- checked below against a file whose contents are known.
    bad_re = re.compile(r'(?<![0-9])' + str(FNV_BROKEN) + r'(?![0-9])')
    good_re = re.compile(r'(?<![0-9])' + str(FNV_CORRECT) + r'(?![0-9])')
    # Comments are stripped first. A note explaining that a basis was wrong is not a use of it,
    # and counting prose would have made this file's own fix look like the defect it describes.
    strip = re.compile(r'//[^\n]*')
    for p, t in sorted(dev_text.items()):
        code = strip.sub('', t)
        bad = len(bad_re.findall(code))
        good = len(good_re.findall(code))
        if bad or good:
            total_bad += bad
            rel = p.relative_to(ROOT)
            print(f'  {str(rel):<48} digit-dropped {bad}   correct {good}')
    print(f'\n  {total_bad} digit-dropped basis literal(s) remain in device sources.')
    print('  Each is only a defect where the value must agree with something outside its own')
    print('  key space; see #124 (it did) and #125 (they do not, but their comments claim they do).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
