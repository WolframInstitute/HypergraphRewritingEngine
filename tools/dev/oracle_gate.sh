#!/bin/bash
# The reference corpus gate: reference/oracle_fuzz.wls on a fixed seed range (every case class,
# CPU and GPU, against reference/MultiwayReference.wl), then tools/dev/worker_fuzz.py at its
# default budget (CPU against the GPU worker, 1 to 32 CPU workers). A local gate, not run in CI.
#
#   tools/dev/oracle_gate.sh                  # seeds 0..41 (three per class), then worker_fuzz
#   ORACLE_SEEDS="100 199" tools/dev/oracle_gate.sh
#   ORACLE_GPU=0 tools/dev/oracle_gate.sh     # no GPU
#   ORACLE_SKIP_WORKER_FUZZ=1 tools/dev/oracle_gate.sh
#
# The paclet is loaded from a copy of paclet/ named by the hash of its binaries and Kernel, so a
# rebuilt binary is never served from a cache of the old one. On WSL the Windows Wolfram Engine
# runs the fuzz and loads the Windows binaries (./build_windows_msvc.sh); elsewhere wolframscript
# on PATH loads this platform's binaries. WOLFRAMSCRIPT overrides the program. Runs of
# wolframscript take the lock /tmp/hg_wolfram.lock, one at a time.
#
# Exit status: 0 when both pass; 1 when either reports a difference (the fuzz writes disc.txt
# under the output directory, worker_fuzz its minimised findings).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${ORACLE_OUT:-$ROOT/.scratch/oracle_gate}"
read -r SEED_FROM SEED_TO <<< "${ORACLE_SEEDS:-0 41}"
GPU="${ORACLE_GPU:-1}"
mkdir -p "$OUT"

WIN=0
if grep -qi microsoft /proc/version 2>/dev/null; then WIN=1; fi
if [ -n "${WOLFRAMSCRIPT:-}" ]; then
  W="$WOLFRAMSCRIPT"
elif [ "$WIN" = 1 ]; then
  W=$(ls -d "/mnt/c/Program Files/Wolfram Research/Wolfram/"*/wolframscript.exe 2>/dev/null | sort -V | tail -1)
else
  W=wolframscript
fi
[ -n "$W" ] || { echo "oracle_gate: no wolframscript"; exit 2; }

if [ "$WIN" = 1 ]; then SYS=Windows-x86-64; else SYS=Linux-x86-64; fi
tag=$(cat "$ROOT"/paclet/LibraryResources/$SYS/* "$ROOT"/paclet/Kernel/*.wl "$ROOT"/paclet/PacletInfo.wl | md5sum | cut -c1-8)
PACLET="${TMPDIR:-/tmp}/hg_oracle_paclet_$tag"
if [ ! -d "$PACLET" ]; then
  mkdir -p "$PACLET/LibraryResources"
  cp -r "$ROOT/paclet/Kernel" "$ROOT/paclet/PacletInfo.wl" "$PACLET/"
  cp -r "$ROOT/paclet/LibraryResources/$SYS" "$PACLET/LibraryResources/"
fi
path() { if [ "$WIN" = 1 ]; then wslpath -w "$1"; else echo "$1"; fi; }

run="$OUT/fuzz_${SEED_FROM}_${SEED_TO}"
rm -rf "$run"; mkdir -p "$run"
echo "oracle_gate: oracle_fuzz seeds $SEED_FROM..$SEED_TO, gpu $GPU, paclet $PACLET"
for attempt in 1 2 3 4 5; do
  flock /tmp/hg_wolfram.lock "$W" -file "$(path "$ROOT/reference/oracle_fuzz.wls")" \
    "$(path "$PACLET")" "$(path "$run")" "$SEED_FROM" "$SEED_TO" "$GPU" 0 > "$run/log.txt" 2>&1
  rc=$?
  # exit 5 with no output is the transient licence failure
  if [ "$rc" = 5 ] && ! grep -q "^seed" "$run/log.txt"; then sleep 5; continue; fi
  break
done
grep -E "^seed|^DONE" "$run/log.txt"
fuzz_rc=$rc
[ "$fuzz_rc" = 0 ] || echo "oracle_gate: oracle_fuzz exit $fuzz_rc; differences in $run/disc.txt"

wf_rc=0
if [ "${ORACLE_SKIP_WORKER_FUZZ:-0}" != 1 ]; then
  echo "oracle_gate: worker_fuzz"
  python3 -u "$ROOT/tools/dev/worker_fuzz.py" --out "$OUT/worker_fuzz"
  wf_rc=$?
  [ "$wf_rc" = 0 ] || echo "oracle_gate: worker_fuzz exit $wf_rc; findings in $OUT/worker_fuzz"
fi
[ "$fuzz_rc" = 0 ] && [ "$wf_rc" = 0 ]
