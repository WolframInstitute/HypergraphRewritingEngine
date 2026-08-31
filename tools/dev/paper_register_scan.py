#!/usr/bin/env python3
"""Flag sentences that fall outside formal expository register.

Reads the paper's section sources, strips LaTeX commands and math, splits into sentences and
reports, per file, sentences that are (a) very short declaratives used as prose (seven words or
fewer, outside itemize/caption/algorithm environments), (b) copula aphorisms ("The X is a Y."
of eight words or fewer), (c) sentence-initial coordinating conjunctions, (d) figurative verbs
or phrases from the list below, (e) narrative of the work ("we found", "it turned out"), or
(f) contractions. Each hit is printed with its file and a short context; the exit code is the
number of hits, so the scan can gate a build.

Usage: python3 tools/dev/paper_register_scan.py [paper/sections/*.tex ...]
"""
import re
import sys
import glob
import os

FIGURATIVE = [
    r"\bcuts? against\b", r"\bpays? for\b", r"\blives? in\b", r"\btally\b", r"\bthe tell\b",
    r"\bload-bearing\b", r"\bknife-edge\b", r"\bpoor fit\b", r"\bshape of (?:the )?work\b",
    r"\bon the same footing\b", r"\bhang(?:s|ing)? off\b", r"\bmints?\b", r"\bminted\b",
    r"\bspends? (?:its|their) cycles\b", r"\bcheap(?:er|est)? (?:trick|win)\b",
    r"\bhand(?:s|ed)? (?:it|them) (?:off|back)\b", r"\bstands? in front of\b",
    r"\bfalls? out of\b", r"\bdrops? out\b", r"\bbuys?\b", r"\bat its core\b",
    r"\bin the wild\b", r"\bunder the hood\b", r"\bfront and cent(?:er|re)\b",
    r"\bfoot(?:ing|print) of\b", r"\bfire(?:s|d)?\b(?! (?:on|when|if|only))",
]
NARRATIVE = [r"\bwe found\b", r"\bit turned out\b", r"\bwe discovered\b", r"\bwe noticed\b",
             r"\bwe realised\b", r"\bwe realized\b", r"\bwe tried\b", r"\bwe had\b"]
CONTRACTION = re.compile(r"\b\w+n't\b|\b(?:it|that|there|what|who)'s\b|\b\w+'(?:re|ve|ll|d)\b")
INITIAL_CONJ = re.compile(r"^(?:And|But|So|Or|Yet)\b")
SKIP_ENV = ("itemize", "enumerate", "description", "algorithmic", "algorithm", "tabular",
            "table", "table*", "figure", "figure*", "align", "align*", "equation", "tikzpicture",
            "axis", "strip", "abstract")


def strip_tex(text):
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = re.sub(r"\$[^$]*\$", " M ", text)
    text = re.sub(r"\\(?:cite|ref|label|eqref|url|texttt|emph|textbf|textsc|textit|mathrm|caption)\*?\{[^}]*\}",
                  " R ", text)
    text = re.sub(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?", " ", text)
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"~", " ", text)
    return text


def prose_blocks(src):
    """Yield text outside skipped environments."""
    depth = 0
    out = []
    for line in src.splitlines():
        m = re.match(r"\s*\\begin\{([^}]*)\}", line)
        if m and m.group(1) in SKIP_ENV:
            depth += 1
            continue
        m = re.match(r"\s*\\end\{([^}]*)\}", line)
        if m and m.group(1) in SKIP_ENV:
            depth = max(0, depth - 1)
            continue
        if depth == 0 and not line.strip().startswith(("\\section", "\\subsection", "\\paragraph",
                                                        "\\subsubsection", "\\label", "\\input")):
            out.append(line)
    return "\n".join(out)


def sentences(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\b(?:e\.g|i\.e|cf|vs|et al|Fig|Sec|Eq|Prop|Tab|No)\.", lambda m: m.group(0).replace(".", "§"), text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text)
    return [p.replace("§", ".").strip() for p in parts if p.strip()]


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "..", "paper", "sections", "*.tex")))
    total = 0
    for f in files:
        src = open(f, encoding="utf-8").read()
        text = strip_tex(prose_blocks(src))
        hits = []
        for s in sentences(text):
            words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]*", s) if w not in ("M", "R")]
            n = len(words)
            if n == 0:
                continue
            if n <= 7 and s.endswith("."):
                hits.append(("short", s))
            elif n <= 8 and re.match(r"^(?:The|A|An|This|That|Each|Every) [A-Za-z' -]+ (?:is|are|was|were) (?:a|an|the|not|only|what|how)\b", s):
                hits.append(("copula", s))
            if INITIAL_CONJ.match(s):
                hits.append(("initial-conjunction", s))
            for pat in FIGURATIVE:
                if re.search(pat, s):
                    hits.append(("figurative:" + pat, s))
            for pat in NARRATIVE:
                if re.search(pat, s):
                    hits.append(("narrative", s))
            if CONTRACTION.search(s):
                hits.append(("contraction", s))
        rel = os.path.relpath(f)
        print(f"== {rel}: {len(hits)} hit(s)")
        for kind, s in hits:
            print(f"  [{kind}] {s[:150]}")
        total += len(hits)
    print(f"{total} hit(s)")
    return min(total, 255)


if __name__ == "__main__":
    sys.exit(main())
