#!/bin/bash
# The first-call gate: N fresh wolframscript kernels, each making its first HGEvolve call through
# reference/first_call_probe.wls, and the number of kernels that ended during that call. Measured
# on Windows with Wolfram 15.0 (CPU unless named): the socket worker of 93d295c2 ended 3 of 40
# kernels; the pipe worker of 7bef595b ended 0 of 40 on the CPU and 0 of 40 on the GPU.
#
#   tools/dev/first_call_gate.sh                     # 40 kernels, CPU, this tree's paclet
#   FIRST_CALL_N=40 FIRST_CALL_DEVICE=GPU tools/dev/first_call_gate.sh
#   FIRST_CALL_PACLET=/path/to/paclet tools/dev/first_call_gate.sh   # another paclet directory
#
# Each kernel runs under the lock /tmp/hg_wolfram.lock, one at a time, with HG_TUNE_MEMORY_LIMIT
# (default 3 GB) passed to Windows through WSLENV. Check that the GPU is idle (nvidia-smi) before
# a GPU run. A kernel whose marker has no "start" did not run the script (the transient licence
# failure) and is run again, up to five times.
#
# Exit status: 0 when no kernel ended during its first call, 1 otherwise, 2 on a setup error.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
N="${FIRST_CALL_N:-40}"
DEVICE="${FIRST_CALL_DEVICE:-CPU}"
export HG_TUNE_MEMORY_LIMIT="${HG_TUNE_MEMORY_LIMIT:-3221225472}"

WIN=0
if grep -qi microsoft /proc/version 2>/dev/null; then WIN=1; fi
if [ -n "${WOLFRAMSCRIPT:-}" ]; then
  W="$WOLFRAMSCRIPT"
elif [ "$WIN" = 1 ]; then
  W=$(ls -d "/mnt/c/Program Files/Wolfram Research/Wolfram/"*/wolframscript.exe 2>/dev/null | sort -V | tail -1)
  export WSLENV="HG_TUNE_MEMORY_LIMIT${WSLENV:+:$WSLENV}"
else
  W=wolframscript
fi
[ -n "$W" ] || { echo "first_call_gate: no wolframscript"; exit 2; }
path() { if [ "$WIN" = 1 ]; then wslpath -w "$1"; else echo "$1"; fi; }

PACLET="${FIRST_CALL_PACLET:-$ROOT/paclet}"
if [ "$WIN" = 1 ]; then
  MARK_WSL=/mnt/c/Temp/hg_first_call_marker.txt
  mkdir -p /mnt/c/Temp
else
  MARK_WSL="${TMPDIR:-/tmp}/hg_first_call_marker.txt"
fi
MARK="$(path "$MARK_WSL")"
case "$PACLET" in /*) PACLET="$(path "$PACLET")";; esac

ended=0
returned=0
for i in $(seq 1 "$N"); do
  for attempt in 1 2 3 4 5; do
    rm -f "$MARK_WSL"
    flock /tmp/hg_wolfram.lock timeout 300 "$W" -file "$(path "$ROOT/reference/first_call_probe.wls")" \
      "$PACLET" "$MARK" "$DEVICE" > /dev/null 2>&1
    rc=$?
    if grep -q start "$MARK_WSL" 2>/dev/null; then break; fi
  done
  if grep -q returned "$MARK_WSL" 2>/dev/null; then
    returned=$((returned + 1))
    echo "kernel $i rc=$rc $(grep returned "$MARK_WSL" | tr -d '\r')"
  else
    ended=$((ended + 1))
    echo "kernel $i rc=$rc ENDED during its first call"
  fi
done
echo "first_call_gate: device $DEVICE, $N kernels: $returned returned, $ended ended during the first call"
[ "$ended" = 0 ]
