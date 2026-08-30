#!/usr/bin/env bash
# The sub-second repository-completeness gates CI runs, run BEFORE a commit exists, because
# every commit auto-pushes and a push with one of these broken is a red CI nobody gets to
# amend. Engine gates (suites, sanitizers) are not here -- they are minutes, they run at
# commit batches -- this is only what a7960950 went red on: checks over the tree's own
# completeness that cost nothing and were skipped.
#
# Install on a clone:  ln -sf ../../tools/dev/precommit_gates.sh .git/hooks/pre-commit
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
fail=0

# docs/CODEMAP.md names what exists, and names everything that exists.
python3 tools/dev/codemap_check.py || fail=1

# Every fragment the paper inputs is in the INDEX (about to be committed), and every indexed
# fragment is input. The staged tree is what CI will see: a generated-but-untracked fragment
# passes every local paper build and fails every clean clone.
inputs=$(grep -oE '\\input\{tables/[a-z0-9_]+\}' paper/main.tex | sed 's/.*tables\///; s/}//' | sort -u)
staged=$(git ls-files --cached paper/tables/ | xargs -n1 basename 2>/dev/null | sed -n 's/\.tex$//p' | sort -u)
if [ "$inputs" != "$staged" ]; then
  echo "pre-commit: the fragments main.tex inputs and the fragments git tracks differ:" >&2
  diff <(echo "$inputs") <(echo "$staged") | sed 's/^/  /' >&2
  fail=1
fi

# The paper's voice.
python3 tools/dev/paper_style_check.py >/dev/null || fail=1

# The measured content percolates: provenance present, no verdict tokens, no stale fragments.
python3 tools/dev/paper_integrity_check.py || fail=1

exit $fail
