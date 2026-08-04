#!/usr/bin/env python3
"""Check the shipped documentation against the symbols the paclet exports.

WHY THIS EXISTS. `paclet/Kernel/HypergraphRewriting.wl` declares its public surface with
PackageExport. The documentation under `paclet/Documentation/English/` is a set of BUILT
notebooks, tracked in git, and generated from three markdown sources -- so a page can
outlive the symbol it documents and nothing regenerates or removes it. That is what
happened: the visualisation split deleted 21 functions and their reference pages stayed,
inside the shipped archive, describing calls a user cannot make.

TWO CHECKS, both mechanical:

  ORPHAN PAGE   ReferencePages/Symbols/<Name>.nb where <Name> is not exported
  DEAD LINK     a guide or tutorial links to `.../ref/<Name>` for a <Name> not exported

The second matters on its own: removing a page while leaving the guide entry turns a
wrong page into a broken link, which is not an improvement.

WHAT THIS CANNOT CONCLUDE. It does not check that a page's CONTENT is accurate, only that
its subject exists. A page documenting an exported symbol incorrectly reads as fine here.

Exit code is the number of findings, so CI can gate on it.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KERNEL = os.path.join(ROOT, "paclet", "Kernel", "HypergraphRewriting.wl")
DOCS = os.path.join(ROOT, "paclet", "Documentation", "English")
PAGES = os.path.join(DOCS, "ReferencePages", "Symbols")

EXPORT_RE = re.compile(r'PackageExport\[\s*"([A-Za-z$][A-Za-z0-9$]*)"\s*\]')
# A documentation link is "paclet:<publisher>/<paclet>/ref/<Symbol>".
REF_RE = re.compile(r'/ref/([A-Za-z$][A-Za-z0-9$]*)')
# A notebook wraps long strings with a backslash-newline, and it does so mid-name: a link
# stored as "ref/\<newline>HGMinkowskiSprinkling" reads as a link to nothing unless the
# wrap is undone first. Missing those made this report 3 dead links where there are 4.
CONTINUATION_RE = re.compile(r'\\\r?\n')


def links_in(path):
    with open(path, errors="replace") as f:
        return set(REF_RE.findall(CONTINUATION_RE.sub("", f.read())))


def main():
    if not os.path.exists(KERNEL):
        sys.exit(f"{KERNEL} does not exist; run this inside the repository")

    with open(KERNEL, errors="replace") as f:
        exported = set(EXPORT_RE.findall(f.read()))
    if not exported:
        sys.exit("found no PackageExport in the Kernel source; refusing to report every "
                 "page as an orphan on what is more likely a parse failure here")

    findings = []

    if os.path.isdir(PAGES):
        for name in sorted(os.listdir(PAGES)):
            if not name.endswith(".nb"):
                continue
            symbol = name[:-3]
            if symbol not in exported:
                findings.append(
                    f"ORPHAN   ReferencePages/Symbols/{name} documents `{symbol}`, which "
                    f"the Kernel does not export. It ships, so a user reads a page for a "
                    f"call they cannot make.")

    for dirpath, _dirs, files in os.walk(DOCS):
        for name in sorted(files):
            if not name.endswith(".nb"):
                continue
            path = os.path.join(dirpath, name)
            linked = links_in(path)
            rel = os.path.relpath(path, ROOT)
            for symbol in sorted(linked - exported):
                findings.append(
                    f"DEADLINK {rel} links to `{symbol}`, which the Kernel does not "
                    f"export. The link resolves to no page.")

    for f_ in findings:
        print(f_)
    print(f"\n{len(findings)} findings over {len(exported)} exported symbol(s): "
          f"{', '.join(sorted(exported))}")
    return len(findings)


if __name__ == "__main__":
    sys.exit(main())
