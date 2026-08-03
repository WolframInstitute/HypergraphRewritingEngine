#!/usr/bin/env python3
"""Check the Windows DLL export list against the sources that must define it.

WHY THIS EXISTS, AND WHY NO BUILD CAUGHT IT. `paclet_source/HypergraphRewriting.def`
lists the symbols the Windows paclet DLL exports. A Windows linker resolves every name in
that list and fails the link if one is missing; an ELF shared object does not, so a Linux
build of the same sources links and says nothing. Every gate in this repository runs on
Linux, and the two on-push Windows and macOS legs configure with
-DBUILD_WOLFRAM_LANGUAGE_PACLET=OFF, so nothing consumes this file at all. It went stale
when the visualisation split deleted three exported functions and stayed stale until
someone ran the release matrix by hand.

WHAT THIS CHECKS. Each name under EXPORTS is defined somewhere in the tracked C/C++
sources, as a DEFINITION and not merely a call: the name must appear followed by a
parameter list and then an opening brace. That is what the linker needs and it is
decidable without a compiler, a Wolfram install or a Windows runner.

WHAT IT CANNOT CONCLUDE. It does not check that the definition is reachable from the
paclet target's own translation units, nor that it carries C linkage, nor that a symbol
survives the librarian. A name found here can still fail to link. The converse is the
useful direction and it is sound: a name found NOWHERE cannot possibly link, which is
exactly the break that shipped.

Exit code is the number of findings, so CI can gate on it.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEF = os.path.join(ROOT, "paclet_source", "HypergraphRewriting.def")
SOURCE_EXT = (".c", ".cc", ".cpp", ".cu", ".h", ".hpp", ".cuh")

# Names the Wolfram runtime resolves rather than the library defining them itself. Each is
# supplied by LibraryLink's own header as an inline or macro, so it is a real export with
# no definition in this tree.
RUNTIME_PROVIDED = {"WolframLibrary_getVersion"}


def exported_names(path):
    """The names under EXPORTS. `;` opens a comment and a name may carry link attributes."""
    names, in_exports = [], False
    with open(path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.split(";", 1)[0].strip()
            if not line:
                continue
            if line.upper().startswith("EXPORTS"):
                in_exports = True
                rest = line[len("EXPORTS"):].strip()
                if rest:
                    names.append((rest.split()[0], lineno))
                continue
            if not in_exports:
                continue          # LIBRARY / DESCRIPTION / VERSION precede EXPORTS
            names.append((line.split()[0], lineno))
    return names


def main():
    if not os.path.exists(DEF):
        sys.exit(f"{DEF} does not exist. If the Windows export list was deleted, delete "
                 f"this check and its CI leg in the same commit.")

    out = subprocess.run(["git", "-C", ROOT, "ls-files"], capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit("git ls-files failed; run this inside the repository")

    blobs = []
    for p in out.stdout.splitlines():
        if p.endswith(SOURCE_EXT):
            with open(os.path.join(ROOT, p), errors="replace") as f:
                blobs.append((p, f.read()))

    names = exported_names(DEF)
    findings = []
    for name, lineno in names:
        if name in RUNTIME_PROVIDED:
            continue
        # A DEFINITION: the name, a parenthesised parameter list, then a brace. This
        # deliberately does not match a call or a prototype, both of which a deleted
        # function leaves behind and neither of which the linker can use.
        pat = re.compile(r"\b" + re.escape(name) + r"\s*\([^;{]*\)\s*(?:const\s*)?\{", re.S)
        if not any(pat.search(text) for _, text in blobs):
            findings.append(
                f"UNDEFINED {os.path.basename(DEF)}:{lineno} exports `{name}`, which no "
                f"tracked source defines. A Windows link of the paclet DLL fails on this; "
                f"a Linux .so links anyway, which is why no gate here sees it.")

    for f_ in findings:
        print(f_)
    print(f"\n{len(findings)} findings over {len(names)} exported names, "
          f"{len(blobs)} tracked sources")
    return len(findings)


if __name__ == "__main__":
    sys.exit(main())
