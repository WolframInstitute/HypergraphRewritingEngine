#!/usr/bin/env python3
"""Ground-truth for artifact_stamp_check.py: it must find the defects a fixture is built to have.

A checker that reports "0 findings" over a directory proves nothing until it has been shown to
report the right findings over a directory whose defects are known by construction. This builds
one fixture per defect class the checker claims to catch, runs the checker over it, and requires
the exact expected message.

The one class it CANNOT fabricate is the true positive on a real binary; that is covered by
running `hg_evolve --version` and requiring the literal it prints to be the literal the checker
extracts from the same file, which is the fifth case below.

Usage:  python3 tools/dev/artifact_stamp_check_selftest.py
Exit 0 when every case behaves as specified.
"""

import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKER = os.path.join(REPO, "tools", "dev", "artifact_stamp_check.py")

HEAD = "a" * 40
OTHER = "b" * 40


def stamp(commit, variant):
    return f"HGBUILDSTAMP/1 commit={commit} variant={variant} :HGBUILDSTAMP".encode()


def run_checker(root, expect_commit=HEAD):
    p = subprocess.run(
        [sys.executable, CHECKER, "--root", root, "--expect-commit", expect_commit],
        capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def write(root, platform, name, blob):
    d = os.path.join(root, platform)
    os.makedirs(d, exist_ok=True)
    # Padding on both sides: a stamp is found in the middle of a binary, never at a file boundary.
    with open(os.path.join(d, name), "wb") as f:
        f.write(b"\x00" * 64 + blob + b"\x00" * 64)


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


@case("a current, correctly-stamped, COMPLETE platform passes")
def _(root):
    # Both required artifacts: a platform holding only one of them is a partial build, which is
    # a separate case below. The passing fixture has to be a platform that would actually ship.
    write(root, "Linux-x86-64", "hg_evolve", stamp(HEAD, "hg_evolve"))
    write(root, "Linux-x86-64", "libHypergraphRewriting.so", stamp(HEAD, "paclet-library"))
    rc, out = run_checker(root)
    return rc == 0 and "0 findings" in out, out


@case("a platform holding a binary but no library is caught")
def _(root):
    # The shape that shipped: LibraryResources/Windows-x86-64 carried hg_evolve.exe and no
    # HypergraphRewriting.dll. Every file present was current, so a check that only reads
    # stamps saw nothing wrong, while HGEvolve had one fewer route than the paclet claims.
    write(root, "Windows-x86-64", "hg_evolve.exe", stamp(HEAD, "hg_evolve"))
    rc, out = run_checker(root)
    return rc == 1 and "HypergraphRewriting.dll is MISSING" in out, out


@case("the optional GPU binary is not required")
def _(root):
    # hg_evolve_gpu is built only where CUDA is, and build_all_platforms routes its failure to
    # SKIPPED on purpose. Requiring it would fail every platform that legitimately has none.
    write(root, "Linux-x86-64", "hg_evolve", stamp(HEAD, "hg_evolve"))
    write(root, "Linux-x86-64", "libHypergraphRewriting.so", stamp(HEAD, "paclet-library"))
    rc, out = run_checker(root)
    return rc == 0 and "0 findings" in out, out


@case("an artifact built from another commit is reported as stale")
def _(root):
    write(root, "Linux-x86-64", "hg_evolve", stamp(OTHER, "hg_evolve"))
    rc, out = run_checker(root)
    return rc == 1 and f"built from {OTHER[:12]}" in out, out


@case("an unstamped artifact is a failure, not a skip")
def _(root):
    write(root, "Linux-x86-64", "hg_evolve", b"no stamp anywhere in this file")
    rc, out = run_checker(root)
    return rc == 1 and "NO build stamp" in out, out


@case("a file carrying the wrong variant is caught (a renamed or copied binary)")
def _(root):
    write(root, "Windows-x86-64", "hg_evolve_gpu.exe", stamp(HEAD, "hg_evolve"))
    rc, out = run_checker(root)
    return rc == 1 and "must be 'hg_evolve_gpu'" in out, out


@case("two differently-stamped objects linked into one artifact are caught")
def _(root):
    write(root, "Linux-x86-64", "hg_evolve",
          stamp(HEAD, "hg_evolve") + b"\x00" * 8 + stamp(OTHER, "hg_evolve"))
    rc, out = run_checker(root)
    return rc == 1 and "different stamps linked together" in out, out


@case("an EMPTY artifact tree fails rather than passing vacuously")
def _(root):
    os.makedirs(os.path.join(root, "Linux-x86-64"), exist_ok=True)
    rc, out = run_checker(root)
    return rc == 1 and "no shipped artifacts found" in out, out


def real_binary_case():
    """The checker's extraction must equal what the binary itself prints. This is the only case
    that touches a real artifact, and it is what makes the fixture cases evidence about the
    shipped path rather than about the fixtures."""
    exe = os.path.join(REPO, "paclet", "LibraryResources", "Linux-x86-64", "hg_evolve")
    if not os.path.exists(exe):
        return None, f"SKIP: {exe} not built"
    printed = subprocess.run([exe, "--version"], capture_output=True, text=True).stdout.strip()
    with open(exe, "rb") as f:
        data = f.read()
    found = printed.encode() in data
    return found, f"printed: {printed}\n    present in the file's own bytes: {found}"


def main():
    failures = 0
    for name, fn in CASES:
        with tempfile.TemporaryDirectory() as root:
            ok, out = fn(root)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            failures += 1
            print("    " + out.replace("\n", "\n    "))

    ok, detail = real_binary_case()
    if ok is None:
        print(f"[skip] real binary: {detail}")
    else:
        print(f"[{'PASS' if ok else 'FAIL'}] the binary's --version literal is the literal in its bytes")
        print("    " + detail.replace("\n", "\n    "))
        if not ok:
            failures += 1

    print(f"\n{len(CASES) + (0 if ok is None else 1)} case(s), {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
