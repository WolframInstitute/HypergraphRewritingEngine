#!/usr/bin/env python3
"""Every shipped artifact names the commit it was built from, and it must be HEAD.

WHY THIS EXISTS. A release ships binaries for six platforms plus two CUDA executables, and the
machine that assembles them can execute at most one platform's worth: an ARM64 macOS dylib on a
Linux box is a file, not a program. So "is this artifact built from current source?" cannot be
asked by running it. It is asked by reading a literal the build wrote into the file.

WHAT WENT WRONG WITHOUT IT. build_all_platforms.sh routes a Windows-GPU build failure to SKIPPED
rather than FAILED on purpose -- the six platform libraries are the required artifacts and an
optional GPU build must not block them. The consequence is that a broken CUDA config ships a
Windows directory whose hg_evolve_gpu.exe is whatever was there before, silently. The only thing
that distinguished current from stale was a human reading a file date.

WHAT IT CHECKS
  1. Every artifact under paclet/LibraryResources/<platform>/ carries a stamp at all. A missing
     stamp is a failure, not a skip: it means the artifact predates this instrument or the linker
     dropped the object, and either way the file's provenance is unknown.
  2. Its commit equals `git rev-parse HEAD`.
  3. Its variant matches the file it was found in, so a copy of hg_evolve renamed to
     hg_evolve_gpu.exe (or a platform directory populated from another) is caught.
  4. With --require-clean, the working tree has no modified tracked file. This is SEPARATE from
     the stamp on purpose: the stamp is written at CONFIGURE time and cannot know what the tree
     looked like at BUILD time, so a "dirty" flag baked into it would be a claim the build cannot
     support. Each question is answered where it is answerable.

The stamp format is fixed by paclet_source/build_stamp.hpp; the regex below is the other half of
that contract.

Usage:
    python3 tools/dev/artifact_stamp_check.py [--require-clean] [--platform PAT]

Exit 0 when every artifact found is stamped with HEAD; 1 otherwise. Finding NO artifacts is also
a failure -- an empty tree passing a check would be the same false green as the SKIPPED path.
"""

import argparse
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIBRESOURCES = os.path.join(REPO, "paclet", "LibraryResources")

STAMP_RE = re.compile(
    rb"HGBUILDSTAMP/1 commit=([0-9a-f]{40}|unknown) variant=([A-Za-z0-9_.-]+) :HGBUILDSTAMP"
)

# filename -> the variant its build must have stamped. A file not listed here is not a shipped
# artifact and is not scanned; every file that IS shipped must appear, so adding a fourth
# artifact without teaching this table fails loudly at the next sign-off rather than silently
# shipping something unstamped.
EXPECTED_VARIANT = {
    "libHypergraphRewriting.so": "paclet-library",
    "libHypergraphRewriting.dylib": "paclet-library",
    "HypergraphRewriting.dll": "paclet-library",
    "hg_evolve": "hg_evolve",
    "hg_evolve.exe": "hg_evolve",
    "hg_evolve_gpu": "hg_evolve_gpu",
    "hg_evolve_gpu.exe": "hg_evolve_gpu",
}


def head_commit():
    return subprocess.run(
        ["git", "-C", REPO, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def find_stamps(path):
    """Every stamp in the file. More than one means two differently-stamped objects were linked
    together, which is a defect worth naming rather than resolving by picking the first."""
    with open(path, "rb") as f:
        data = f.read()
    return [(m.group(1).decode(), m.group(2).decode()) for m in STAMP_RE.finditer(data)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-clean", action="store_true",
                    help="also fail when a tracked file is modified in the working tree")
    ap.add_argument("--platform", default="",
                    help="only scan platform directories whose name contains this substring")
    ap.add_argument("--root", default=LIBRESOURCES,
                    help="directory of platform subdirectories to scan (default: the paclet's)")
    ap.add_argument("--expect-commit", default="",
                    help="the commit artifacts must carry (default: git rev-parse HEAD). This is "
                         "how the checker is ground-truthed: pointed at a fixture with a known "
                         "stamp, it must report exactly the mismatch that fixture contains.")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    head = args.expect_commit or head_commit()
    findings = []
    scanned = 0

    if not os.path.isdir(root):
        print(f"FAIL: {root} does not exist")
        return 1

    for platform in sorted(os.listdir(root)):
        pdir = os.path.join(root, platform)
        if not os.path.isdir(pdir):
            continue
        if args.platform and args.platform not in platform:
            continue
        for name in sorted(os.listdir(pdir)):
            expected = EXPECTED_VARIANT.get(name)
            if expected is None:
                continue
            path = os.path.join(pdir, name)
            scanned += 1
            stamps = find_stamps(path)
            rel = os.path.relpath(path, REPO)
            if not stamps:
                findings.append(f"{rel}: NO build stamp -- provenance unknown")
                continue
            if len(set(stamps)) > 1:
                findings.append(f"{rel}: {len(set(stamps))} different stamps linked together: {sorted(set(stamps))}")
                continue
            commit, variant = stamps[0]
            if commit != head:
                findings.append(f"{rel}: built from {commit[:12]}, HEAD is {head[:12]}")
            if variant != expected:
                findings.append(f"{rel}: stamped variant '{variant}', this file must be '{expected}'")
            if not findings or not findings[-1].startswith(rel):
                print(f"  ok  {rel}  {commit[:12]}  {variant}")

    if scanned == 0:
        print("FAIL: no shipped artifacts found to check "
              "(an empty tree must not pass -- that is the silent-skip failure this catches)")
        return 1

    if args.require_clean:
        dirty = subprocess.run(
            ["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if dirty:
            n = len(dirty.splitlines())
            findings.append(
                f"working tree has {n} modified tracked file(s); the stamp names HEAD but the "
                f"artifacts were built from HEAD plus those edits:\n    "
                + "\n    ".join(dirty.splitlines()))

    print(f"\n{scanned} artifact(s) scanned against HEAD {head[:12]}")
    if findings:
        print(f"{len(findings)} finding(s):")
        for f in findings:
            print(f"  FAIL  {f}")
        return 1
    print("0 findings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
