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

# Root already (a rented container), sudo available (a bare-metal login), or neither. The
# privileged steps are all optional -- governor, persistence mode, counter permission -- so
# "neither" degrades rather than fails, and the log says which box this was.
if   [ "$(id -u)" = 0 ];        then SUDO=""
elif command -v sudo >/dev/null; then SUDO="sudo"
else                                 SUDO="";  say "no root and no sudo: skipping governor/clock/counter setup"
fi

# nvidia-smi is on PATH on a rented box and is NOT under WSL (/usr/lib/wsl/lib), and nvcc
# is at /usr/local/cuda/bin, which no distro puts on PATH. CMake's CUDA probe reads PATH and
# DISABLES GPU SUPPORT WITH EXIT 0 when it finds nothing, so both are resolved before any
# configure runs -- see the assertion after the GPU configure for why exit 0 is not enough.
export PATH="/usr/local/cuda/bin:$PATH"
NVSMI="$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)"

# --------------------------------------------------------------------------- 0 preflight
say "preflight: $(date -u +%FT%TZ)"
{
  echo "--- os";      grep -E "^(NAME|VERSION)=" /etc/os-release
  echo "--- cpu";     lscpu | grep -E "Model name|Socket|Core|Thread|NUMA node\(s\)|MHz"
  echo "--- mem";     free -g
  echo "--- load";    uptime
  echo "--- gpu";     "$NVSMI" --query-gpu=name,driver_version,memory.total,compute_cap --format=csv || true
  echo "--- tenants"; ps -eo comm,pcpu --sort=-pcpu --no-headers | head -5
} > "$ROOT/preflight.txt" 2>&1
cat "$ROOT/preflight.txt" | tee -a "$LOG"
# Disk: the clone plus a host build plus a CUDA build with its object files. A box that
# cannot hold them fails at link time, an hour in.
FREE_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
[ "${FREE_GB:-0}" -ge 25 ] || fail "only ${FREE_GB}G free under $HOME; the two builds need ~25G"

LOAD1=$(awk '{print $1}' /proc/loadavg)
# A rented container reports the HOST's load average, so this also catches a rental that is
# not the whole machine. HG_ACCEPT_CONTENDED=1 proceeds anyway -- the generator stamps every
# table CONTENDED, so the override cannot quietly produce a number that reads as clean.
if ! awk -v l="$LOAD1" 'BEGIN { exit (l < 0.7) ? 0 : 1 }'; then
  if [ "${HG_ACCEPT_CONTENDED:-0}" = 1 ]; then
    say "load ${LOAD1} >= 0.7 and HG_ACCEPT_CONTENDED=1: proceeding, every table will be stamped CONTENDED"
  else
    fail "load ${LOAD1} >= 0.7 — not a quiet box. Re-run with HG_ACCEPT_CONTENDED=1 to measure anyway."
  fi
fi

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
$SUDO apt-get update -qq >> "$LOG" 2>&1 || true
$SUDO apt-get install -y -qq build-essential cmake ninja-build git python3 valgrind \
  linux-tools-common "linux-tools-$(uname -r)" numactl >> "$LOG" 2>&1 || true
command -v nvcc >/dev/null || $SUDO apt-get install -y -qq nvidia-cuda-toolkit >> "$LOG" 2>&1 || true
command -v nvcc >/dev/null || fail "no nvcc — install a CUDA toolkit matching the driver first"

say "governor + clocks"
echo performance | $SUDO tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1 || true
$SUDO "$NVSMI" -pm 1 >> "$LOG" 2>&1 || true

# GPU counter permission: ncu needs it for the P7.5 device captures. Recorded, not fatal —
# the wall-clock tables do not depend on it.
NCU_OK=no
if command -v ncu >/dev/null; then
  ncu --query-metrics >/dev/null 2>&1 && NCU_OK=yes
  [ "$NCU_OK" = no ] && $SUDO sh -c 'echo options nvidia NVreg_RestrictProfilingToAdminUsers=0 > /etc/modprobe.d/ncu.conf' 2>/dev/null || true
fi
say "ncu counters: $NCU_OK"

# --------------------------------------------------------------------------- 1 source
say "clone @ $COMMIT"
# No --recurse-submodules: the only submodule is the markdown-to-notebook converter,
# which nothing in this session builds, and it drags several nested repositories.
git clone "$REPO" src >> "$LOG" 2>&1 || fail "clone failed"
cd src && git checkout "$COMMIT" >> "$LOG" 2>&1 || fail "checkout $COMMIT failed"
say "HEAD: $(git rev-parse --short HEAD)  $(git log -1 --format=%s | head -c 60)"

# --------------------------------------------------------------------------- 2 build
# BUILD_WOLFRAM_LANGUAGE_PACLET DEFAULTS ON AND IS FATAL WITHOUT A WOLFRAM INSTALL:
# paclet_source does find_package(WolframLanguage REQUIRED), so on a box with no Wolfram the
# configure aborts before a single file compiles. Off here, as CI has it. The paclet targets
# are the serialization tests, which this session does not measure.
COMMON_FLAGS=(-DCMAKE_BUILD_TYPE=Release -DBUILD_WOLFRAM_LANGUAGE_PACLET=OFF -DBUILD_VISUALIZATION=OFF)

# The two probes are EXCLUDE_FROM_ALL (every tools/*.cpp is), and paper_tables.py runs BOTH:
# quotient_reconstruction_cost_probe for the C/R ratio table and mode_matrix_probe for the
# identity-mode matrix. Named here because a default build does not produce them and the
# generator would exit an hour into the session saying so.
say "build host (build_linux, -j$(nproc))"
cmake -S . -B build_linux "${COMMON_FLAGS[@]}" -DBUILD_GPU=OFF >> "$LOG" 2>&1 || fail "host configure"
cmake --build build_linux -j"$(nproc)" --target all_tests bench_cpu_evolve cost_matrix \
  mode_matrix_probe quotient_reconstruction_cost_probe >> "$LOG" 2>&1 || fail "host build"

# nvcc forks a multi-GB cicc per translation unit, so the CUDA build's width is bounded by
# MEMORY, not by cores: a 64-core box with modest RAM would swap and take longer than -j1.
MEM_GB=$(free -g | awk '/^Mem:/ {print $2}')
GPU_J=$(( MEM_GB / 6 )); [ "$GPU_J" -lt 1 ] && GPU_J=1; [ "$GPU_J" -gt 8 ] && GPU_J=8

ARCH="$("$NVSMI" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')"
[ -n "$ARCH" ] || ARCH=89          # 4090 = sm_89; the query is the authority when it answers
say "build gpu (build_gpu, sm_$ARCH, -j$GPU_J from ${MEM_GB}G RAM)"
cmake -S . -B build_gpu "${COMMON_FLAGS[@]}" -DBUILD_GPU=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" >> "$LOG" 2>&1 || fail "gpu configure"
# CMake DISABLES GPU SUPPORT AND EXITS 0 when it cannot find a CUDA compiler, so a successful
# configure proves nothing. Reproduced deliberately: without /usr/local/cuda/bin on PATH the
# configure above returns 0, logs "CUDA not found - GPU support disabled", and emits none of
# these targets -- which would surface an hour later as a missing binary.
(cd build_gpu && make help 2>/dev/null | grep -qx "... bench_gpu_evolve") \
  || fail "the GPU configure produced no GPU targets: CUDA was not found. Install a toolkit and ensure nvcc is on PATH."
cmake --build build_gpu -j"$GPU_J" --target hg_gpu_tests gpu_differential_tests bench_gpu_evolve >> "$LOG" 2>&1 || fail "gpu build"

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
