#!/usr/bin/env python3
"""Check docs/CODEMAP.md against the tree it describes.

CODEMAP is prose: it says what a directory is FOR, which is not derivable from the
sources and so cannot be generated. What CAN be checked is the part that rots -- the
inventory. Two questions, both mechanical:

  MISSING   a source file exists in a documented directory and CODEMAP does not name it
  STALE     CODEMAP names a file, or a `backticked` C++ identifier, that is not in the tree

The identifier check is the one that caught a live defect before it existed: CODEMAP
described `EdgeCausalInfo` as a live type for as long as it took someone to read the
header and notice nothing used it. A name is looked up as a whole word across the
tracked sources, so a type that is deleted, renamed, or moved out of the codebase reads
as STALE the moment it goes.

WHAT THIS CANNOT CONCLUDE. A found identifier is a name that appears somewhere in the
sources, not one that appears where CODEMAP says it does. This checks that the map names
real things, not that it names them in the right place -- the second needs the semantic
index (tools/dev/source_map.py) and a claim about location that CODEMAP does not make.

Exit code is the number of findings, so CI can gate on it.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODEMAP = os.path.join(ROOT, "docs", "CODEMAP.md")

SOURCE_EXT = (".hpp", ".h", ".cpp", ".cu", ".cuh")
# A name in CODEMAP can resolve against more than C++: a build target lives in a
# CMakeLists, a WL symbol in a .wl, and a probe's name is its file's, never a token
# inside it. Searching only the C++ corpus reports all three as missing.
CORPUS_EXT = SOURCE_EXT + (".wl", ".wls", ".cmake", ".txt")

# A section header names the directory it documents: "## `gpu/src/` -- CUDA kernels".
SECTION_RE = re.compile(r"^##+ +`([^`]+)`")
# A bulleted file entry opens with a bold span, which may name SEVERAL files:
# "- **`event_core.hpp` / `match_core.hpp`** -- the shared semantic cores".
FILE_RE = re.compile(r"^\s*-\s+\*\*(.+?)\*\*")
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Directories whose contents CODEMAP deliberately summarises rather than enumerates:
# one line for a family of near-identical files is the useful description, and demanding
# an entry per file would make the map longer and worse.
SUMMARISED = ("tools/", "benchmarks/", "benchmarking/", "paclet/", "visualisation/")

# Words that are backticked in CODEMAP as prose or as non-C++ names. Each is spelled the
# way an identifier is, so the pattern cannot tell them apart and the list is the only
# way to keep them out of the findings.
NOT_IDENTIFIERS = {
    "main", "true", "false", "null", "nullptr", "void", "int", "bool", "char",
    "size_t", "uint32_t", "uint64_t", "int32_t", "int64_t", "uint8_t", "double",
    "float", "auto", "const", "static", "inline", "template", "typename",
    "std", "gtest", "cmake", "nvcc", "clang", "gcc", "msvc", "wasm", "emscripten",
    "python", "bash", "make", "git", "CMakeLists", "README", "LICENSE",
    "venv",
}


def tracked_files():
    out = subprocess.run(["git", "-C", ROOT, "ls-files"], capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit("git ls-files failed; run this inside the repository")
    return out.stdout.splitlines()


def main():
    if not os.path.exists(CODEMAP):
        sys.exit(f"{CODEMAP} does not exist. If CODEMAP was deleted, delete this check "
                 f"and its CI leg in the same commit.")

    tracked = tracked_files()
    sources = [p for p in tracked if p.endswith(SOURCE_EXT)]
    by_dir = {}
    for p in sources:
        by_dir.setdefault(os.path.dirname(p) + "/", set()).add(os.path.basename(p))

    # One pass over the corpus rather than one search per identifier: the identifier list
    # is in the hundreds and a subprocess each would dominate the runtime.
    corpus_words = set()
    for p in tracked:
        if not p.endswith(CORPUS_EXT):
            continue
        with open(os.path.join(ROOT, p), errors="replace") as f:
            corpus_words.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", f.read()))
    # A file's own name is a name CODEMAP may use -- a probe, a test binary, a benchmark
    # group -- and it need not appear as a token in any file's text.
    basenames = {os.path.splitext(os.path.basename(p))[0] for p in tracked}
    corpus_words |= basenames

    documented = {}          # directory -> documented at all (the section exists)
    named_files = set()      # every source file name CODEMAP mentions, in any section
    idents = {}              # identifier -> line number of its first mention
    section = None
    findings = []

    with open(CODEMAP) as f:
        for lineno, line in enumerate(f, 1):
            m = SECTION_RE.match(line)
            if m:
                section = m.group(1)
                if not section.endswith("/"):
                    section = None          # a section about a file, not a directory
                else:
                    documented.setdefault(section, set())
                continue
            fm = FILE_RE.match(line)
            if fm and section:
                # Names come from the bullet's HEAD -- every bold span before the description
                # separator, since a header and its implementation are written as two spans --
                # and never from the description, which cites files it does not document.
                head = re.split(r" -- ", line, maxsplit=1)[0]
                for part in re.findall(r"`([^`]+)`", head):
                    # Not every backticked thing ending in an extension is a file name: a
                    # bullet writes a paired implementation as a bare suffix
                    # ("`hg_gpu_backend.hpp`/`.cpp`") and a directory as a glob ("`*.cu`").
                    # Both need a real stem to be a name this can look up.
                    base = os.path.basename(part)
                    stem, ext = os.path.splitext(base)
                    if ext in SOURCE_EXT and stem and "*" not in stem:
                        named_files.add(base)
            for tok in re.findall(r"`([^`]+)`", line):
                if IDENT_RE.match(tok) and tok not in NOT_IDENTIFIERS:
                    idents.setdefault(tok, lineno)

    # Both checks are over the whole map rather than per section. A bullet legitimately
    # names a file from another directory -- a .cu section naming the header it implements
    # -- and scoping the comparison to the section reports those as missing on one side and
    # stale on the other, which is two findings for a map that is correct.
    all_basenames = {os.path.basename(p) for p in sources}
    for directory in sorted(documented):
        if directory.startswith(SUMMARISED):
            continue
        actual = by_dir.get(directory, set())
        if not actual:
            findings.append(f"STALE   directory `{directory}` has no tracked sources")
            continue
        for name in sorted(actual - named_files):
            findings.append(f"MISSING {directory}{name} exists and CODEMAP does not name it")
    for name in sorted(named_files - all_basenames):
        findings.append(f"STALE   CODEMAP names `{name}`, which is not in the tree "
                        f"(a NEW file reads this way until it is `git add`ed -- the tree here is "
                        f"`git ls-files`, because CODEMAP documents the repository)")

    for name, lineno in sorted(idents.items()):
        if name in corpus_words:
            continue
        # A trailing underscore marks a file-name PREFIX standing for a group, as the
        # benchmark groups are written; it resolves if anything is named after it.
        if name.endswith("_") and any(b.startswith(name) for b in basenames):
            continue
        findings.append(f"STALE   CODEMAP:{lineno} names `{name}`, which is in no tracked file")

    for f_ in findings:
        print(f_)
    print(f"\n{len(findings)} findings over {len(documented)} documented directories, "
          f"{len(idents)} identifiers, {len(sources)} tracked sources")
    return len(findings)


if __name__ == "__main__":
    sys.exit(main())
