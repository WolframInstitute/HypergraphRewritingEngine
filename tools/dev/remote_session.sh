#!/usr/bin/env bash
# The P7.6 remote measurement session, as one script: rent the box, run this, copy one
# bundle back, tear down. Modeled on a runbook whose design goal this keeps: everything a
# box must produce is collected in a single sitting, and the script prints SAFE TO TEARDOWN
# only when the bundle is complete, so the box never needs a second rental to answer a
# question the first one could have answered.
#
#   ssh <box> 'bash -s' < tools/dev/remote_session.sh            # runs everything
#   scp <box>:hg_bundle_*.tar.gz .                               # then tear down
#
# NO CREDENTIALS LAND ON THE BOX: the repository is public and is cloned read-only over
# HTTPS. Never copy a token or an SSH private key onto rented hardware.
#
# Assumes Ubuntu 24.04 with the NVIDIA driver installed (the provider's image). Everything
# else is installed here. Wolfram authority rows (T2) are NOT measured remotely -- they need
# a licensed kernel -- and the generator carries their macros forward from the local run.
# P3.9 (Windows-vs-Linux) also stays local: this box has no Windows leg.
set -uo pipefail

REPO="https://github.com/WolframInstitute/HypergraphRewritingEngine.git"
COMMIT="${1:-master}"
ROOT="$HOME/hg_session"
LOG="$ROOT/session.log"
mkdir -p "$ROOT"
cd "$ROOT"

say()  { echo "==> $*" | tee -a "$LOG"; }
fail() { echo "XX  $*" | tee -a "$LOG"; echo "BUNDLE INCOMPLETE — do not tear down without reading $LOG"; exit 1; }

# --------------------------------------------------------------------------- 0 preflight
say "preflight: $(date -u +%FT%TZ)"
{
  echo "--- os";      grep -E "^(NAME|VERSION)=" /etc/os-release
  echo "--- cpu";     lscpu | grep -E "Model name|Socket|Core|Thread|NUMA node\(s\)|MHz"
  echo "--- mem";     free -g
  echo "--- load";    uptime
  echo "--- gpu";     nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
  echo "--- tenants"; ps -eo comm,pcpu --sort=-pcpu --no-headers | head -5
} > "$ROOT/preflight.txt" 2>&1
cat "$ROOT/preflight.txt" | tee -a "$LOG"
LOAD1=$(awk '{print $1}' /proc/loadavg)
awk -v l="$LOAD1" 'BEGIN { exit (l < 0.7) ? 0 : 1 }' \
  || fail "load ${LOAD1} >= 0.7 — this box is not quiet; a table stamped CONTENDED is not worth the rental"

# The pinned set: one hardware thread per physical core, and the sweep in powers of two up
# to the physical-core count. A homogeneous box is the whole point of the rental; the
# generator still stamps the set so the table carries its own provenance.
mapfile -t FIRST_THREADS < <(lscpu -p=CPU,CORE | grep -v '^#' | awk -F, '!seen[$2]++ {print $1}')
NPHYS=${#FIRST_THREADS[@]}
CPUSET=$(IFS=,; echo "${FIRST_THREADS[*]}")
SWEEP="1"; n=2; while [ "$n" -lt "$NPHYS" ]; do SWEEP="$SWEEP,$n"; n=$((n*2)); done; SWEEP="$SWEEP,$NPHYS"
say "physical cores: $NPHYS   pin set: ${CPUSET:0:60}...   sweep: $SWEEP"

say "deps"
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq >> "$LOG" 2>&1
sudo apt-get install -y -qq build-essential cmake ninja-build git python3 valgrind \
  linux-tools-common "linux-tools-$(uname -r)" || true >> "$LOG" 2>&1
command -v nvcc >/dev/null || sudo apt-get install -y -qq nvidia-cuda-toolkit >> "$LOG" 2>&1
command -v nvcc >/dev/null || fail "no nvcc — install a CUDA toolkit matching the driver first"

say "governor + clocks"
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1 || true
sudo nvidia-smi -pm 1 >> "$LOG" 2>&1 || true

# GPU counter permission: ncu needs it for the P7.5 device captures. Recorded, not fatal —
# the wall-clock tables do not depend on it.
NCU_OK=no
if command -v ncu >/dev/null; then
  ncu --query-metrics >/dev/null 2>&1 && NCU_OK=yes
  [ "$NCU_OK" = no ] && sudo sh -c 'echo options nvidia NVreg_RestrictProfilingToAdminUsers=0 > /etc/modprobe.d/ncu.conf' 2>/dev/null || true
fi
say "ncu counters: $NCU_OK"

# --------------------------------------------------------------------------- 1 source
say "clone @ $COMMIT"
git clone --recurse-submodules "$REPO" src >> "$LOG" 2>&1 || fail "clone failed"
cd src && git checkout "$COMMIT" >> "$LOG" 2>&1 || fail "checkout $COMMIT failed"
say "HEAD: $(git rev-parse --short HEAD)  $(git log -1 --format=%s | head -c 60)"

# --------------------------------------------------------------------------- 2 build
say "build host (build_linux, -j$(nproc))"
cmake -S . -B build_linux -DCMAKE_BUILD_TYPE=Release >> "$LOG" 2>&1 || fail "host configure"
cmake --build build_linux --target all_tests bench_cpu_evolve cost_matrix -j"$(nproc)" >> "$LOG" 2>&1 || fail "host build"

say "build gpu (build_gpu, -j8 — dedicated box, RAM permitting)"
cmake -S . -B build_gpu -DCMAKE_BUILD_TYPE=Release -DBUILD_GPU=ON >> "$LOG" 2>&1 || fail "gpu configure"
cmake --build build_gpu --target hg_gpu_tests gpu_differential_tests bench_gpu_evolve -j8 >> "$LOG" 2>&1 || fail "gpu build"

# --------------------------------------------------------------------------- 3 gates
# A number from an unverified box is not a measurement. Full suites, once, before anything
# is recorded; red aborts the session with the log.
say "gates"
./build_linux/all_tests > "$ROOT/gate_all_tests.log" 2>&1 || fail "all_tests red — see gate_all_tests.log"
./build_gpu/hg_gpu_tests > "$ROOT/gate_gpu_tests.log" 2>&1 || fail "hg_gpu_tests red"
./build_gpu/gpu_differential_tests > "$ROOT/gate_differential.log" 2>&1 || fail "differential red"
say "gates green: $(grep -h "PASSED" "$ROOT"/gate_*.log | tr '\n' ' ')"

# --------------------------------------------------------------------------- 4 measure
say "paper tables (pinned, no --wolfram: authority macros carry forward from the local run)"
python3 -u tools/dev/paper_tables.py --gpu --authority-depth 7 --steps 7 \
  --cpus "$CPUSET" --thread-sweep "$SWEEP" \
  --build-dir build_linux --gpu-build-dir build_gpu > "$ROOT/paper_tables.log" 2>&1 \
  || fail "paper_tables failed — see paper_tables.log"

say "scaling sweep"
python3 -u tools/dev/scaling_sweep.py --sections cpu,shapes,memory,gpu \
  --build-dir build_linux --gpu-build-dir build_gpu > "$ROOT/scaling_sweep.log" 2>&1 \
  || fail "scaling_sweep failed — see scaling_sweep.log"

say "device floor (P4.11 adjudication)"
./build_gpu/bench_gpu_evolve 4 20 2 triangle > "$ROOT/floor_triangle.log" 2>&1 || true
./build_gpu/bench_gpu_evolve 7 5 2 wpp     > "$ROOT/floor_wpp.log"      2>&1 || true

if [ "$NCU_OK" = yes ]; then
  say "ncu captures (P7.5 device side)"
  ncu --set full -o "$ROOT/ncu_wpp" ./build_gpu/bench_gpu_evolve 6 1 2 wpp >> "$LOG" 2>&1 || true
fi

# --------------------------------------------------------------------------- 5 bundle
cd "$ROOT"
BUNDLE="$HOME/hg_bundle_$(hostname)_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
tar czf "$BUNDLE" preflight.txt session.log gate_*.log paper_tables.log scaling_sweep.log \
  floor_*.log ncu_* src/paper/tables 2>/dev/null
say "bundle: $BUNDLE ($(du -h "$BUNDLE" | cut -f1))"
echo ""
echo "SAFE TO TEARDOWN — copy the bundle first:  scp <box>:$BUNDLE ."
