#!/usr/bin/env python3
"""Check the paper for LLM writing tells, war stories, and advertised optimisations.

WHY THIS EXISTS. A reader who spots machine phrasing discounts the content, and the phrasing is
recognisable: there are public catalogues of it (claudisms.ai is the one this list is drawn from).
The tells are not a matter of taste that a careful re-read catches -- they recur, they survive
editing, and they are invisible to the person who wrote them. A grep is.

Three separate checks, because they fail for three different reasons:

  TELLS       phrasing that reads as machine-written. Elevated diction where a plain word exists,
              announcement fillers, metaphor standing in for a mechanism, value-claim assertions,
              negative parallelism, totalizing superlatives, self-qualifiers.

  WAR STORY   the paper describes the system AS IT IS, not the history of arriving at it. A
              sentence about a defect that was fixed, a before/after framing, or an account of
              what was tried belongs in the commit log. Released software does not advertise the
              bugs it no longer has.

  OPPORTUNITY anything named as remaining performance work. Under the v1.0.0 standard that is a
              work item to close or refute, not a sentence to keep -- so its presence in the paper
              is a claim that the implementation was not finished.

Exit status is 1 if anything is found and 0 if not; the count is NOT the exit code, because a
finding count that lands on 256 exits 0 and this project has been bitten by exactly that.

Usage:  tools/dev/paper_style_check.py [paper/main.tex ...]
"""

import re
import sys
from pathlib import Path

# A pattern is (regex, why). Case-insensitive unless the pattern needs otherwise. Word boundaries
# are deliberate: "harness" the verb is a tell, but "harness" the noun is what verification/genmc
# holds, and the paper says so legitimately -- so entries that collide with this project's own
# vocabulary carry a narrower pattern rather than being dropped.
TELLS = [
    # Elevated diction where a plain word exists.
    (r"\bdelve[sd]?\b", "elevated diction: delve"),
    (r"\bleverag(e|es|ed|ing)\b", "elevated diction: leverage -- say 'use'"),
    (r"\bharness(es|ed|ing)\b(?! )", "elevated diction: harness as a verb"),
    (r"\bseamless(ly)?\b", "elevated diction: seamless"),
    (r"\bintricate(ly)?\b", "elevated diction: intricate"),
    (r"\bholistic(ally)?\b", "elevated diction: holistic"),
    (r"\bpivotal\b", "elevated diction: pivotal"),
    (r"\btransformative\b", "elevated diction: transformative"),
    (r"\bgroundbreaking\b", "elevated diction: groundbreaking"),
    (r"\bgame[- ]changing\b", "elevated diction: game-changing"),
    (r"\bcutting[- ]edge\b", "elevated diction: cutting-edge"),
    (r"\btestament\b", "elevated diction: testament"),
    (r"\brealm\b", "elevated diction: realm"),
    (r"\blandscape\b", "figurative landscape"),
    (r"\bnavigat(e|es|ed|ing)\s+(the|this|these|a)\b", "figurative navigate"),
    (r"\bunderscor(e|es|ed|ing)\b", "elevated diction: underscore"),
    (r"\bfoster(s|ed|ing)?\b", "elevated diction: foster"),
    (r"\bshed(s|ding)? light on\b", "elevated diction: shed light on"),
    (r"\bpave[sd]? the way\b", "elevated diction: pave the way"),
    (r"\bdive[sd]? into\b", "elevated diction: dive into"),
    # Announcement fillers and signposting.
    (r"\bit(?:'s| is) worth noting\b", "announcement filler"),
    (r"\bit(?:'s| is) important to note\b", "announcement filler"),
    (r"\bwhen it comes to\b", "announcement filler"),
    (r"\bat its core\b", "announcement filler"),
    (r"\blet(?:'s| us) break (?:it|this) down\b", "announcement filler"),
    (r"\bhere(?:'s| is) where it gets interesting\b", "announcement filler"),
    (r"\bthe point is\b", "announcement filler"),
    (r"\bcannot be overstated\b", "announcement filler"),
    (r"\bthis is where .{1,30} comes in\b", "announcement filler"),
    # Value-claim assertions.
    (r"\bthis matters\b", "value-claim assertion"),
    (r"\bworth (noting|asking|considering|examining|exploring|drawing|making)\b",
     "value-claim assertion"),
    (r"\bthe (right|useful) (way|answer|question|tool|time|part|thing)\b",
     "value-claim assertion"),
    # Metaphor standing in for a mechanism.
    (r"\bload[- ]bearing\b", "metaphor for a mechanism"),
    (r"\bthe tell\b", "metaphor for a mechanism"),
    (r"\bdoing the (heavy lifting|work)\b", "metaphor for a mechanism"),
    (r"\bheavy lifting\b", "metaphor for a mechanism"),
    (r"\bthe physics of\b", "pseudo-scientific metaphor"),
    (r"\blives in the\b", "abstract placement metaphor"),
    (r"\bhits? hardest\b", "over-dramatisation"),
    # Negative parallelism and cleft contrast.
    (r"\bnot just .{1,40}, (but|it(?:'s| is))\b", "negative parallelism"),
    (r"\bnot only .{1,40}, but\b", "negative parallelism"),
    (r"\bisn(?:'t| not) about .{1,40}\.\s+It(?:'s| is) about\b", "negative parallelism"),
    # Totalizing superlatives.
    (r"\bthe whole (game|ballgame|point)\b", "totalizing superlative"),
    (r"\bthe only thing that matters\b", "totalizing superlative"),
    (r"\bthe entire point\b", "totalizing superlative"),
    # Self-qualifiers about one's own claims.
    (r"\bhonest(ly)?\b", "self-qualifier"),
    (r"\bto be clear\b", "self-qualifier"),
    (r"\bwe want to be (careful|precise)\b", "self-qualifier"),
]

# The paper states the system as it is. These describe getting there.
WAR_STORY = [
    (r"\bused to\b", "history, not the current system"),
    (r"\bpreviously\b", "history, not the current system"),
    (r"\bearlier version(s)?\b", "history, not the current system"),
    (r"\b(was|were|had been) (a )?(bug|defect|regression|slower|broken)\b",
     "defect narrative"),
    (r"\bwe (found|discovered|noticed|observed) that .{0,40}(bug|defect|slow|regress)",
     "defect narrative"),
    (r"\b(has|have|had) since been (fixed|corrected|resolved|addressed)\b", "defect narrative"),
    # "no longer" describing RUNTIME state -- "a completion that can no longer occur" -- is the
    # system as it is, and the paper is entitled to say it. Only the code-history sense is a war
    # story, so the pattern requires a subject that makes it one.
    (r"\b(we|the (implementation|engine|code|design)) (do(es)? )?no longer\b",
     "defined by contrast with absent code"),
    (r"\bbefore (this|the) (change|fix|optimi[sz]ation)\b", "before/after framing"),
    (r"\bafter (this|the) (change|fix|optimi[sz]ation)\b", "before/after framing"),
    (r"\binitially,? (we|the)\b", "history, not the current system"),
    (r"\bat first,? (we|the)\b", "history, not the current system"),
    (r"\bturned out to be\b", "discovery narrative"),
]

# Remaining performance work. Scope items (distributed execution, rule learning) are NOT this --
# they are directions, not optimisations of what ships -- so the patterns name performance.
OPPORTUNITY = [
    (r"\bremaining incremental step\b", "advertised optimisation"),
    (r"\bnot (yet )?optimi[sz]ed\b", "advertised optimisation"),
    (r"\bcould be (further )?optimi[sz]ed\b", "advertised optimisation"),
    (r"\broom for improvement\b", "advertised optimisation"),
    (r"\bfurther (speedup|optimi[sz]ation|performance)\b", "advertised optimisation"),
    (r"\bwould (further )?(improve|reduce|speed up)\b", "advertised optimisation"),
    (r"\byet to be (exploited|realised|realized)\b", "advertised optimisation"),
    (r"\bopportunit(y|ies) (for|to)\b", "advertised optimisation"),
    (r"\bin its reach\b", "advertised optimisation"),
    (r"\bis the remaining\b", "advertised optimisation"),
]

CATEGORIES = [("TELL", TELLS), ("WAR STORY", WAR_STORY), ("OPPORTUNITY", OPPORTUNITY)]

# A comment line is not prose the reader sees. LaTeX comments start with an unescaped %.
COMMENT = re.compile(r"(?<!\\)%")


def strip_comment(line: str) -> str:
    m = COMMENT.search(line)
    return line[: m.start()] if m else line


def check(path: Path):
    findings = []
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for n, raw in enumerate(text, 1):
        line = strip_comment(raw)
        if not line.strip():
            continue
        for label, patterns in CATEGORIES:
            for pat, why in patterns:
                for m in re.finditer(pat, line, re.IGNORECASE):
                    findings.append((n, label, why, m.group(0), line.strip()[:110]))
    return findings


def em_dash_density(path: Path):
    """LaTeX writes an em dash as ---. Overuse is a structural tell; a count is not a verdict."""
    text = path.read_text(encoding="utf-8", errors="replace")
    body = "\n".join(strip_comment(l) for l in text.splitlines())
    dashes = len(re.findall(r"(?<!-)---(?!-)", body))
    words = len(re.findall(r"\b\w+\b", body))
    return dashes, words


def main(argv):
    paths = [Path(p) for p in argv[1:]] or [Path(__file__).resolve().parents[2] / "paper" / "main.tex"]
    total = 0
    for path in paths:
        if not path.exists():
            print(f"paper_style_check: no such file: {path}", file=sys.stderr)
            return 2
        findings = check(path)
        total += len(findings)
        for n, label, why, hit, ctx in findings:
            print(f"{path}:{n}: {label}: {why} -- “{hit}”\n    {ctx}")
        d, w = em_dash_density(path)
        per_1k = (1000.0 * d / w) if w else 0.0
        print(f"{path}: {len(findings)} finding(s); {d} em dashes over {w} words "
              f"({per_1k:.1f} per 1000 words)")
    # Status, not count: a count that lands on 256 exits 0.
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
