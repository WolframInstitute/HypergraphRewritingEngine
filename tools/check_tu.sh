#!/usr/bin/env bash
# Semantic-check one translation unit with clangd, using the EXACT flags from
# build_linux/compile_commands.json -- no build, no link, ~7 s for the largest TU.
#
# WHY THIS EXISTS. This box is shared and sometimes has no memory for a real build (safe_build
# refuses below its floor). The fallback used then was `g++ -fsyntax-only` with hand-guessed
# include paths, which checks against guessed flags rather than the build's. clangd --check reads
# the compile database, so it checks the same -std, -I, -D and -O the build would use.
#
# NOISE FILTER. `clangd --check` also dry-runs every refactoring tweak and counts each
# inapplicable one as an "error" (ExtractFunction on a break/continue, and so on), so its final
# "N errors" line is useless as a verdict. Real compiler diagnostics carry "error:"/"warning:";
# tweak lines carry "tweak:". The filter below is that distinction, and the exit code is computed
# from the filtered count, not clangd's.
#
# Usage: tools/check_tu.sh <file.cpp> [more files...]
#        exit 0 = no compiler errors in any file; 1 = at least one; 2 = usage/tooling failure

set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

[ $# -ge 1 ] || { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
command -v clangd >/dev/null || { echo "check_tu: clangd not on PATH" >&2; exit 2; }
[ -f build_linux/compile_commands.json ] || {
    echo "check_tu: build_linux/compile_commands.json missing; configure cmake once first" >&2
    exit 2
}

fail=0
for f in "$@"; do
    out="$(timeout 300 clangd --check="$f" --compile-commands-dir=build_linux 2>&1)"
    rc=$?
    # clangd's own exit code counts the tweak noise, so it cannot be the verdict; only a timeout
    # or crash is a tooling failure. The verdict is the filtered diagnostics and nothing else.
    #
    # A --check diagnostic is "E[time] [diag_name] Line N: message" -- no "error:" substring, so
    # matching compiler-style spelling catches nothing (measured: an injected undeclared
    # identifier passed a filter written that way). "] Line N:" is the discriminator; the E/W
    # prefix is the severity; tweak lines carry neither.
    diags="$(printf '%s\n' "$out" | grep -E "\] Line [0-9]+:")"
    errs="$(printf '%s\n' "$diags" | grep -cE "^E\[")"
    if [ "$rc" -ge 124 ]; then
        echo "check_tu: $f -- clangd timed out or crashed (rc=$rc)"; fail=1
    elif [ "$errs" -gt 0 ]; then
        echo "check_tu: $f -- $errs error(s)"; printf '%s\n' "$diags" | head -20; fail=1
    else
        w="$(printf '%s\n' "$diags" | grep -cE "^W\[")"
        echo "check_tu: $f -- clean ($w warning(s))"
        [ "$w" -gt 0 ] && printf '%s\n' "$diags" | head -5
    fi
done
exit $fail
