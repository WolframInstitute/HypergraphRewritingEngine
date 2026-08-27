#!/usr/bin/env bash
# Run the TLC model-checking cells in this directory and check each one's declared verdict.
#
# WHY THIS EXISTS. Every cell here was run by hand and its numbers pasted into README.md. That
# makes the results a transcript rather than a gate: nothing re-runs them, so a spec edit that
# turns a PASS into a VIOLATION -- or, worse, turns the CALIBRATION cell's violation into a pass,
# which means the calibration has stopped calibrating -- is invisible until someone re-reads the
# README and repeats the commands. GenMC's harnesses have run.sh and a ctest; these did not.
#
# THE CELL DECLARES ITS OWN VERDICT. Line 1 of each .cfg ends in `Expected: PASS` or
# `Expected: VIOLATION`, and that string is what this script checks against. Keeping the
# expectation in the .cfg rather than in a table here means a new cell cannot be added without
# saying what it should do, and a cell whose verdict legitimately changes is edited in one place.
#
# WHAT IS CHECKED IS THE VERDICT, NOT THE STATE COUNT. Distinct-state counts belong in README.md,
# where they are compared by a reader who knows why they moved; TLC reports different generated
# counts for a VIOLATING cell depending on -workers, because it stops at the first counterexample
# and how far the other workers had gone varies. Asserting those numbers would fail on a machine
# with a different core count. The verdicts do not vary.
#
# SKIPS RATHER THAN FAILS when the toolchain is absent (exit 2, which CMakeLists registers as
# SKIP_RETURN_CODE): TLC needs a JVM and tla2tools.jar, neither of which is vendored here.
#
# Usage:
#   verification/tla/run.sh              every cell
#   verification/tla/run.sh <cell>       one cell, named without .cfg
#   verification/tla/run.sh --quick      every cell except the deep ones (see kDeep below)
#
# Environment:
#   TLA2TOOLS_JAR   path to tla2tools.jar          (default ~/tla/tla2tools.jar)
#   TLC_WORKERS     -workers passed to TLC         (default: all cores)

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

JAR="${TLA2TOOLS_JAR:-$HOME/tla/tla2tools.jar}"
WORKERS="${TLC_WORKERS:-$(nproc 2>/dev/null || echo 4)}"

if ! command -v java >/dev/null 2>&1; then
    echo "SKIP: no java on PATH; TLC cannot run." >&2
    exit 2
fi
if [ ! -f "$JAR" ]; then
    echo "SKIP: tla2tools.jar not at '$JAR'. Set TLA2TOOLS_JAR to override." >&2
    exit 2
fi

# The deep cell is a minute on its own, against seconds for every other cell. --quick drops it so
# the routine gate stays cheap; the full run includes it.
is_deep() { case "$1" in *Deep) return 0 ;; *) return 1 ;; esac; }

# A cell's module is the spec it instantiates: the MCMatchForwarding cells go through the
# MCMatchForwarding wrapper, the SegmentedArray cells straight at SegmentedArray.
module_for() {
    case "$1" in
        MCMatchForwarding*) echo "MCMatchForwarding.tla" ;;
        MCSegmentedArray*)  echo "SegmentedArray.tla" ;;
        MCDepthRelaxation*) echo "DepthRelaxation.tla" ;;
        *) return 1 ;;
    esac
}

QUICK=0
CELLS=()
case "${1:-}" in
    --quick) QUICK=1 ;;
    "")      ;;
    *)       CELLS=("$1") ;;
esac

if [ ${#CELLS[@]} -eq 0 ]; then
    for cfg in *.cfg; do CELLS+=("${cfg%.cfg}"); done
fi

fails=0
skipped=0
for cell in "${CELLS[@]}"; do
    cfg="$cell.cfg"
    if [ ! -f "$cfg" ]; then
        echo "--- $cell: NO SUCH CELL ($cfg)" >&2
        fails=$((fails + 1))
        continue
    fi
    if [ "$QUICK" = 1 ] && is_deep "$cell"; then
        echo "--- $cell: skipped (--quick)"
        skipped=$((skipped + 1))
        continue
    fi

    module="$(module_for "$cell")" || {
        echo "--- $cell: no module mapping; add one to module_for()" >&2
        fails=$((fails + 1))
        continue
    }

    expected="$(head -1 "$cfg" | grep -oE 'Expected: *(PASS|VIOLATION)' | grep -oE '(PASS|VIOLATION)')"
    if [ -z "$expected" ]; then
        echo "--- $cell: line 1 declares no 'Expected: PASS' or 'Expected: VIOLATION'" >&2
        fails=$((fails + 1))
        continue
    fi

    out="$(mktemp)"
    start=$(date +%s)
    java -cp "$JAR" tlc2.TLC -workers "$WORKERS" -deadlock -config "$cfg" "$module" >"$out" 2>&1
    rc=$?
    elapsed=$(($(date +%s) - start))

    # TLC exits 0 having said so when the cell is clean, and 12 on a violated invariant. Both the
    # code and the sentence are checked: a run killed early can leave one without the other.
    if grep -q "Model checking completed. No error has been found" "$out" && [ "$rc" -eq 0 ]; then
        actual=PASS
    elif grep -qE "Error: Invariant .* is violated|Error: Action property .* is violated" "$out"; then
        actual=VIOLATION
    else
        actual="INDETERMINATE(exit $rc)"
    fi

    # The LAST such line, not the first: TLC prints periodic progress before the final summary,
    # and progress numbers carry thousands separators, so `[0-9]+` matches only their last group
    # ("1,234,111 distinct states found" -> "111"). The final summary line is unseparated.
    counts="$(grep -oE '[0-9]+ distinct states found' "$out" | tail -1)"
    if [ "$actual" = "$expected" ]; then
        echo "--- $cell: $actual as declared, ${elapsed}s, $counts"
    else
        echo "--- $cell: EXPECTED $expected, GOT $actual, ${elapsed}s" >&2
        tail -25 "$out" >&2
        fails=$((fails + 1))
    fi
    rm -f "$out"
done

if [ "$fails" -gt 0 ]; then
    echo "$fails cell(s) did not match their declared verdict." >&2
    exit 1
fi
echo "all cells matched their declared verdict ($skipped skipped)."
exit 0
