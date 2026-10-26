#!/bin/bash
# The first-call gate: N fresh wolframscript kernels, each making its first HGEvolve call through
# reference/first_call_probe.wls, and the number of kernels that ended during that call.
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
# A first call that returns no state count, or reports HGEvolve::workerlost, is a failure. A kernel
# that never starts after five attempts is reported apart and is not a failure. A kernel that
# reaches the 300 s timeout is ended by the process id it wrote to its own marker.
#
# Exit status: 0 when no kernel ended during or failed its first call, 1 otherwise, 2 on a setup
# error.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
N="${FIRST_CALL_N:-40}"
DEVICE="${FIRST_CALL_DEVICE:-CPU}"
# The flag that runs a script file: -file for wolframscript, -script for a WolframKernel given as
# WOLFRAMSCRIPT.
FLAG="${WOLFRAMSCRIPT_FILE_FLAG:--file}"
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
  MARK_DIR=/mnt/c/Temp
else
  MARK_DIR="${TMPDIR:-/tmp}"
fi
mkdir -p "$MARK_DIR"
case "$PACLET" in /*) PACLET="$(path "$PACLET")";; esac

# Ends the kernel whose process id the probe wrote in its marker, with its children.
kill_kernel() {
  local pid
  pid=$(sed -n 's/^start \([0-9]*\).*/\1/p' "$1" | tr -d '\r' | head -1)
  [ -n "$pid" ] || return
  if [ "$WIN" = 1 ]; then
    taskkill.exe /F /T /PID "$pid" > /dev/null 2>&1
  else
    kill -9 "$pid" 2> /dev/null
  fi
}

ended=0
failed=0
returned=0
not_started=0
for i in $(seq 1 "$N"); do
  MARK_WSL="$MARK_DIR/hg_first_call_$$_$i.txt"
  MARK="$(path "$MARK_WSL")"
  started=0
  for attempt in 1 2 3 4 5; do
    rm -f "$MARK_WSL"
    flock /tmp/hg_wolfram.lock timeout 300 "$W" "$FLAG" "$(path "$ROOT/reference/first_call_probe.wls")" \
      "$PACLET" "$MARK" "$DEVICE" > /dev/null 2>&1
    rc=$?
    if grep -q '^start' "$MARK_WSL" 2>/dev/null; then started=1; break; fi
  done
  if [ "$rc" = 124 ]; then kill_kernel "$MARK_WSL"; fi
  result=$(sed -n 's/^returned //p' "$MARK_WSL" 2>/dev/null | tr -d '\r')
  messages=$(sed -n 's/^messages //p' "$MARK_WSL" 2>/dev/null | tr -d '\r')
  if [ "$started" = 0 ]; then
    not_started=$((not_started + 1))
    echo "kernel $i rc=$rc NOT STARTED after 5 attempts"
  elif [ -z "$result" ]; then
    ended=$((ended + 1))
    echo "kernel $i rc=$rc ENDED during its first call"
  elif ! [[ "$result" =~ ^[0-9]+$ ]] || [[ "$messages" == *workerlost* ]]; then
    failed=$((failed + 1))
    echo "kernel $i rc=$rc FAILED: returned $result, messages $messages"
  else
    returned=$((returned + 1))
    echo "kernel $i rc=$rc returned $result, messages $messages"
  fi
  rm -f "$MARK_WSL"
done
echo "first_call_gate: device $DEVICE, $N kernels: $returned returned, $failed failed," \
  "$ended ended during the first call, $not_started not started"
[ "$ended" = 0 ] && [ "$failed" = 0 ]
