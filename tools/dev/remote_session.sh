#!/usr/bin/env bash
# The P7.6 measurement session, run on a RENTED, EPHEMERAL box.
#
#   bash remote_session.sh [commit] [phase]
#      phase: prep | prep-host | prep-gpu | sync | tables | sweep | floor | attrib | all
#
# prep-host and prep-gpu exist because a box can arrive with the GPU present but no
# driver installed, and the CPU half of the session needs no GPU at all. Splitting them
# lets the host tables be measured while the driver question is still open, instead of
# the whole session waiting on it.
#
# NOTHING HERE MAY BE THE ONLY COPY OF ANYTHING. The box can vanish between one command and
# the next -- a reassignment, a reboot that wipes the local disk, the day simply ending -- so
# this script is built to be driven phase by phase by tools/dev/remote_drive.sh, which pulls
# every artifact to the local machine as soon as the phase that produced it returns. Two
# properties follow and are the point of the phase split:
#
#   RE-RUNNABLE ON A FRESH BOX. No phase depends on state from a PREVIOUS box. `prep` rebuilds
#   everything it needs from the public repository, so losing the machine costs the build time
#   and nothing else -- never a measurement, because measurements are already local.
#
#   RE-ENTRANT ON THE SAME BOX. `prep` skips a clone that is already there and lets CMake do
#   its own incremental work, so a phase re-run after a dropped connection is cheap. That is an
#   optimisation only: correctness never depends on the box having kept anything.
#
# NO CREDENTIALS LAND ON THE BOX: the repository is public and cloned read-only over HTTPS.
# Nothing is ever pushed from here, and no key or token is copied here.
#
# Assumes Ubuntu 22.04/24.04 with an NVIDIA driver. Wolfram authority rows (T2) are not
# measured remotely -- they need a licensed kernel -- and the generator carries their macros
# forward. P3.9 (Windows-vs-Linux) also stays local: this box has no Windows leg.
set -uo pipefail

REPO="https://github.com/WolframInstitute/HypergraphRewritingEngine.git"
COMMIT="${1:-master}"
PHASE="${2:-all}"
ROOT="$HOME/hg_session"
SRC="$ROOT/src"
LOG="$ROOT/session.log"
mkdir -p "$ROOT"

# WHO HOLDS THE BOX. This process already runs under the shared flock (the driver wraps it),
# so writing the holder here is what lets the other tenant -- the plr project shares this
# machine -- see who is measuring and since when, rather than only that something is. Removed
# on exit, including on a crash or a dropped connection, because the trap fires either way and
# the lock itself is released by the kernel when the file descriptor closes.
# ---------------------------------------------------------------------------
# THE BOX IS SHARED with the other project on this machine, and both do
# timing-sensitive work: a build or a benchmark from one destroys the other's
# numbers. The lock is taken HERE rather than by whoever invokes this script,
# because a caller that forgets to wrap protects nothing, and this script is
# often piped in over ssh where it has no path to re-exec itself.
#
# Held for the lifetime of this process on file descriptor 9. The kernel
# releases it when the process dies, so a dropped ssh or a killed phase leaves
# no stale lock to clear by hand.
exec 9>/tmp/hgbox.lock 2>/dev/null || true
if command -v flock >/dev/null 2>&1; then
  if ! flock -w "${LOCK_WAIT:-7200}" 9; then
    echo "the box is busy: $(cat /tmp/hgbox.holder 2>/dev/null || echo 'holder unknown')" >&2
    echo "waited ${LOCK_WAIT:-7200}s for /tmp/hgbox.lock; not starting." >&2
    exit 75
  fi
fi

BOX_HOLDER=/tmp/hgbox.holder
printf 'hypergraph-engine | phase=%s | commit=%s | pid=%s | since=%s\n' \
  "${2:-all}" "${1:-master}" "$$" "$(date -u +%FT%TZ)" > "$BOX_HOLDER" 2>/dev/null || true
trap 'rm -f "$BOX_HOLDER" 2>/dev/null' EXIT

say()  { echo "==> $*" | tee -a "$LOG"; }
fail() { echo "XX  $*" | tee -a "$LOG"; echo "PHASE FAILED — artifacts so far are already local; see the log"; exit 1; }
# `all` is the measurement pass: prep, then the three measuring phases. `sync` and `attrib`
# are opt-in only -- sync because prep already covers a first run, and attrib because it
# builds a second, instrumented tree and runs callgrind, which is orders of magnitude slower
# than the thing it attributes and has no place in a timing pass.
want() {
  [ "$PHASE" = "$1" ] && return 0
  [ "$PHASE" = all ] || return 1
  case " prep tables sweep floor " in *" $1 "*) return 0 ;; esac
  return 1
}

# Root already (a rented container), sudo available (a bare-metal login), or neither. The
# privileged steps are all optional -- governor, persistence mode, counter permission -- so
# "neither" degrades rather than fails.
if   [ "$(id -u)" = 0 ];        then SUDO=""
elif command -v sudo >/dev/null; then SUDO="sudo"
else                                 SUDO=""
fi

# nvidia-smi is on PATH on a rented box and is NOT under WSL (/usr/lib/wsl/lib), and nvcc is at
# /usr/local/cuda/bin, which no distro puts on PATH. CMake's CUDA probe reads PATH and DISABLES
# GPU SUPPORT WITH EXIT 0 when it finds nothing -- see the assertion after the GPU configure.
export PATH="/usr/local/cuda/bin:$PATH"
NVSMI="$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)"

# The pinned set and the sweep, recomputed every phase because they are cheap and because a
# phase must not depend on a file a previous phase left behind.
mapfile -t FIRST_THREADS < <(lscpu -p=CPU,CORE | grep -v '^#' | awk -F, '!seen[$2]++ {print $1}')
NPHYS=${#FIRST_THREADS[@]}
CPUSET=$(IFS=,; echo "${FIRST_THREADS[*]}")
SWEEP="1"; n=2; while [ "$n" -lt "$NPHYS" ]; do SWEEP="$SWEEP,$n"; n=$((n*2)); done; SWEEP="$SWEEP,$NPHYS"
# A caller that wants every count, or a different CPU set (the SMT siblings for counts past
# the physical cores), names them: HG_SWEEP is the comma list of thread counts, HG_CPUSET the
# comma list of logical CPUs the workers are pinned to.
SWEEP="${HG_SWEEP:-$SWEEP}"
CPUSET="${HG_CPUSET:-$CPUSET}"

# A measurement phase waits for the box to settle rather than failing on the decaying load
# average of the phase before it. HG_ACCEPT_CONTENDED=1 measures anyway; it cannot produce a
# number that reads as clean, because the generator stamps every table CONTENDED.
wait_quiet() {
  # QUIET IS RELATIVE TO THE MACHINE, NOT AN ABSOLUTE NUMBER. The threshold was 0.7 whatever
  # the box was, which on 32 hardware threads is two per cent utilisation -- and the kernel's
  # one-minute average decays with a time constant near sixty seconds, so after a -j32 build
  # and two threaded gate suites it CANNOT reach 0.7 inside this function's own 120s budget.
  # It duly refused to measure on an otherwise idle rented box at load 1.18, having first
  # spent the full 120 seconds of it. The bar is now a tenth of a core per hardware thread,
  # floored at 0.7 so a small box is not held to a looser standard than before.
  local waited=0 load limit
  limit=$(awk -v n="$(nproc)" 'BEGIN { l = n * 0.1; print (l < 0.7) ? 0.7 : l }')
  while :; do
    load=$(awk '{print $1}' /proc/loadavg)
    awk -v l="$load" -v lim="$limit" 'BEGIN { exit (l < lim) ? 0 : 1 }' && return 0
    # The override short-circuits the wait rather than serving it out: a caller who has
    # already accepted contention would otherwise pay 120s of rented time per phase to be
    # told what they said.
    if [ "${HG_ACCEPT_CONTENDED:-0}" = 1 ]; then
      say "load $load over limit $limit, HG_ACCEPT_CONTENDED=1: measuring anyway, tables stamped CONTENDED"
      return 0
    fi
    [ "$waited" -ge 120 ] && fail "load $load exceeds $limit after ${waited}s — not a quiet box. Re-run with HG_ACCEPT_CONTENDED=1 to measure anyway."
    sleep 10; waited=$((waited + 10))
  done
}

# BUILD_WOLFRAM_LANGUAGE_PACLET DEFAULTS ON AND IS FATAL WITHOUT A WOLFRAM INSTALL:
# paclet_source does find_package(WolframLanguage REQUIRED), so on a box with no Wolfram the
# configure aborts before a single file compiles. Off here, as CI has it.
#
# Release is the TIMING build and is what the tables are measured with: it carries the
# project's LTO (CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE), so cross-TU inlining is the same
# as a shipped build, and it defines no instrumentation macro. Nothing is added here that
# could move a number -- the instrumented build is a separate directory (`attrib`).
COMMON_FLAGS=(-DCMAKE_BUILD_TYPE=Release -DBUILD_WOLFRAM_LANGUAGE_PACLET=OFF -DBUILD_VISUALIZATION=OFF)

# Both builds, incremental. Called by `prep` on a new box and by `sync` for every commit
# after, so the iteration loop and the first setup cannot drift apart.
build_host() {
  # The two probes are EXCLUDE_FROM_ALL (every tools/*.cpp is), and paper_tables.py runs BOTH:
  # quotient_reconstruction_cost_probe for the C/R ratio table and mode_matrix_probe for the
  # identity-mode matrix. A default build does not produce them.
  say "build host (-j$(nproc))"
  cmake -S . -B build_linux "${COMMON_FLAGS[@]}" -DBUILD_GPU=OFF >> "$LOG" 2>&1 || fail "host configure"
  # THE COMPLETE SET THE GENERATORS INVOKE, derived from them rather than remembered:
  #   grep -ohE 'binary\((build|a\.build_dir)[^)]*"[a-z_0-9]+"' tools/dev/paper_tables.py \
  #     tools/dev/scaling_sweep.py | grep -oE '"[a-z_0-9]+"'
  # Every tools/*.cpp is EXCLUDE_FROM_ALL, so a default build produces none of them. An earlier
  # list here was assembled by a pattern matching bench|probe|matrix|tests|suite, which cannot
  # match sampling_cost_smoke -- so the tables phase ran five tables and died on the sixth,
  # after the measurement time for those five had already been spent on a rented box.
  cmake --build build_linux -j"$(nproc)" --target all_tests bench_cpu_evolve cost_matrix \
    mode_matrix_probe quotient_reconstruction_cost_probe sampling_cost_smoke \
    >> "$LOG" 2>&1 || fail "host build"

  # THE MEASURING BINARIES ARE A RELEASE BUILD. build_linux keeps the diagnostic counters the
  # gate suites read (HG_ENGINE_STATS defaults ON); with them the arena's per-worker fast path
  # is compiled out and every allocation goes through shared per-site counters, which on this
  # box was 18-39% of the cross-CCX fills by itself. Every number a table or a sweep records
  # comes from build_release, and each binary's own stamp says so (release_only below).
  say "build release measuring binaries (-j$(nproc))"
  cmake -S . -B build_release "${COMMON_FLAGS[@]}" -DBUILD_GPU=OFF -DHG_ENGINE_STATS=OFF \
    -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_EXAMPLES=OFF >> "$LOG" 2>&1 || fail "release configure"
  cmake --build build_release -j"$(nproc)" --target bench_cpu_evolve \
    mode_matrix_probe quotient_reconstruction_cost_probe sampling_cost_smoke \
    >> "$LOG" 2>&1 || fail "release build"
  release_only ./build_release/bench_cpu_evolve
  release_only ./build_release/sampling_cost_smoke
}

# The binary says what it was built with (hgcommon/build_stamp.hpp); a measuring phase runs
# nothing whose stamp is not the release configuration.
release_only() {
  local stamp
  stamp="$("$1" --build-info 2>/dev/null | grep '^HGBUILDSTAMP/2;' || true)"
  case "$stamp" in
    *";stats=0;phase_timing=0;ndebug=1;asan=0;tsan=0;ubsan=0;"*) say "release build: $1" ;;
    *) fail "$1 is not a release build: ${stamp:-no stamp}" ;;
  esac
}

build_gpu_targets() {
  # BOUNDED BY MEMORY AND BY CORES, AND BY NOTHING ELSE. nvcc forks a multi-GB cicc per
  # translation unit, so RAM is the real ceiling -- but the ceiling is not a constant, and it
  # is not this developer's. A cap sized for the 19 GB shared desktop this script is written
  # on leaves three quarters of a 503 GB, 32-thread rented box idle through every CUDA build,
  # so the width is whichever of memory and cores is smaller, computed here.
  local MEM_GB GPU_J ARCH NPROC
  MEM_GB=$(free -g | awk '/^Mem:/ {print $2}')
  NPROC=$(nproc)
  GPU_J=$(( MEM_GB / 6 ))
  [ "$GPU_J" -gt "$NPROC" ] && GPU_J="$NPROC"
  [ "$GPU_J" -lt 1 ] && GPU_J=1
  ARCH="$("$NVSMI" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')"
  [ -n "$ARCH" ] || ARCH=89          # 4090 = sm_89; the query is the authority when it answers
  # A build directory whose cache pins a DIFFERENT nvcc than the PATH's poisons the build
  # with mixed toolchains (measured 30/08: prep's PATH put cuda-13 first over a dir cached on
  # /usr/bin/nvcc; the mix died in glibc's _Float128 declarations and an nvlink load error).
  for gd in build_gpu build_gpu_release; do
    if [ -f "$gd/CMakeCache.txt" ]; then
      cached=$(grep -oP "CMAKE_CUDA_COMPILER:\w+=\K.*" "$gd/CMakeCache.txt" | head -1)
      [ -n "$cached" ] && [ "$cached" != "$(command -v nvcc)" ] && rm -rf "$gd"
    fi
  done
  say "build gpu (sm_$ARCH, -j$GPU_J from ${MEM_GB}G RAM and ${NPROC} threads)"
  # HG_GPU_ARCHS IS THE KNOB, not CMAKE_CUDA_ARCHITECTURES. gpu/CMakeLists.txt caches
  # "75;80;86;89;90" and sets each target's CUDA_ARCHITECTURES property from it, so passing
  # CMAKE_CUDA_ARCHITECTURES is silently ignored and every build compiles SASS for five
  # architectures. Measured on the rented box: cicc/ptxas in flight carried compute_75, 80,
  # 86, 89 and 90 simultaneously -- five times the work, for one card whose arch is known.
  # A rented box builds for the device it has.
  cmake -S . -B build_gpu "${COMMON_FLAGS[@]}" -DBUILD_GPU=ON \
    -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" \
    -DHG_GPU_ARCHS="$ARCH" >> "$LOG" 2>&1 || fail "gpu configure"
  # CMake DISABLES GPU SUPPORT AND EXITS 0 when it cannot find a CUDA compiler, so a successful
  # configure proves nothing. Reproduced deliberately: without /usr/local/cuda/bin on PATH the
  # configure returns 0, logs "CUDA not found - GPU support disabled", and emits no targets.
  # NO PIPE HERE, DELIBERATELY. `make help | grep -q` returns NON-ZERO under `set -o
  # pipefail` even when the pattern matches: grep -q exits at the first match, make gets
  # SIGPIPE, and pipefail reports make's death as the pipeline's status. That cost two failed
  # phases on a rented box, each looking like "CUDA was not found" while CUDA was present and
  # the targets existed. The output is captured first and matched as a string.
  gpu_help="$( (cd build_gpu && make help 2>/dev/null) || true )"
  case "$gpu_help" in
    *"... bench_gpu_evolve"*) ;;
    *) fail "the GPU configure produced no GPU targets. CUDA was not found, or the build directory did not generate." ;;
  esac
  cmake --build build_gpu -j"$GPU_J" --target hg_gpu_tests gpu_differential_tests bench_gpu_evolve \
    >> "$LOG" 2>&1 || fail "gpu build"
  # The device bench that tables and the floor phase time is a release build too; build_gpu
  # keeps the counters for its two gate suites.
  say "build gpu release bench (sm_$ARCH, -j$GPU_J)"
  cmake -S . -B build_gpu_release "${COMMON_FLAGS[@]}" -DBUILD_GPU=ON -DHG_ENGINE_STATS=OFF \
    -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_EXAMPLES=OFF \
    -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" \
    -DHG_GPU_ARCHS="$ARCH" >> "$LOG" 2>&1 || fail "gpu release configure"
  cmake --build build_gpu_release -j"$GPU_J" --target bench_gpu_evolve >> "$LOG" 2>&1 || fail "gpu release build"
  release_only ./build_gpu_release/bench_gpu_evolve
}

# DISCARD THE BOX'S LOCAL MODIFICATIONS, AFTER WRITING THEM DOWN.
#
# A BRANCH NAME MEANS THE REMOTE'S BRANCH, NOT THIS CLONE'S STALE COPY OF IT. `git fetch`
# advances origin/master and leaves the local master exactly where it was, so `git checkout
# master` on a REUSED clone checks out whatever that box last built and then reports it as the
# commit under measurement. Observed: a box sitting on a local master 33 commits behind
# origin/master announced "sync -> 12191ff" and would have measured that tree.
#
# A raw SHA has no origin/<sha>, so it falls through unchanged and still works.
resolve_ref() {
    local want="$1"
    if git -C "$SRC" rev-parse -q --verify "origin/$want^{commit}" >/dev/null 2>&1; then
        echo "origin/$want"
    else
        echo "$want"
    fi
}

# A checkout onto a dirty tree fails, and this tree gets dirty in the ordinary course of a
# session: the generators write paper/tables/*.tex, and a device change under test is copied in
# by hand. Both are safe to drop HERE and only here -- the driver pulls every phase's artifacts
# before the next one starts, including a failed phase's, so anything present when sync runs was
# already brought home.
#
# "Safe to drop" is not the same as "dropped without a record": the diff goes to a timestamped
# patch beside the log first, because a box that is ephemeral is exactly where a lost edit cannot
# be recovered from.
clean_tree() {
  git -C "$SRC" diff --quiet && git -C "$SRC" diff --cached --quiet && return 0
  local keep="$ROOT/dirty-$(date -u +%Y%m%dT%H%M%SZ).patch"
  git -C "$SRC" diff HEAD > "$keep" 2>/dev/null || true
  say "working tree was dirty; its diff is saved at $keep before discarding:"
  git -C "$SRC" status --short >> "$LOG" 2>&1 || true
  git -C "$SRC" checkout -- . >> "$LOG" 2>&1 || fail "could not discard local modifications"
}

# --------------------------------------------------------------------------- prep
if want prep || [ "$PHASE" = prep-host ]; then
  say "prep on $(hostname) at $(date -u +%FT%TZ), commit=$COMMIT"
  {
    echo "--- os";      grep -E "^(NAME|VERSION)=" /etc/os-release
    echo "--- cpu";     lscpu | grep -E "Model name|Socket|Core|Thread|NUMA node\(s\)|MHz"
    echo "--- mem";     free -g
    echo "--- disk";    df -h "$HOME" | tail -1
    echo "--- load";    uptime
    echo "--- gpu";     "$NVSMI" --query-gpu=name,driver_version,memory.total,compute_cap --format=csv || true
    echo "--- tenants"; ps -eo comm,pcpu --sort=-pcpu --no-headers | head -5
  } > "$ROOT/preflight.txt" 2>&1
  cat "$ROOT/preflight.txt" | tee -a "$LOG"

  # The two builds plus object files. A box that cannot hold them fails at link time.
  FREE_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
  [ "${FREE_GB:-0}" -ge 25 ] || fail "only ${FREE_GB}G free under $HOME; the two builds need ~25G"
  wait_quiet
  say "physical cores: $NPHYS   sweep: $SWEEP"

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

  NCU_OK=no
  if command -v ncu >/dev/null; then
    ncu --query-metrics >/dev/null 2>&1 && NCU_OK=yes
    [ "$NCU_OK" = no ] && $SUDO sh -c 'echo options nvidia NVreg_RestrictProfilingToAdminUsers=0 > /etc/modprobe.d/ncu.conf' 2>/dev/null || true
  fi
  say "ncu counters: $NCU_OK"


  # No --recurse-submodules: the only submodule is the markdown-to-notebook converter, which
  # nothing here builds. A clone that is already present is reused and moved to the commit.
  if [ -d "$SRC/.git" ]; then
    say "clone present, fetching"
    git -C "$SRC" fetch --all -q >> "$LOG" 2>&1 || true
  else
    say "clone @ $COMMIT"
    git clone "$REPO" "$SRC" >> "$LOG" 2>&1 || fail "clone failed"
  fi
  clean_tree
  git -C "$SRC" checkout -q --detach "$(resolve_ref "$COMMIT")" >> "$LOG" 2>&1 \
    || fail "checkout $COMMIT failed"
  say "HEAD: $(git -C "$SRC" rev-parse --short HEAD)  $(git -C "$SRC" log -1 --format=%s | head -c 60)"

  cd "$SRC" || fail "no $SRC"
  build_host
  [ "$PHASE" = prep-host ] || build_gpu_targets

  # A number from an unverified box is not a measurement.
  say "gates"
  ./build_linux/all_tests > "$ROOT/gate_all_tests.log" 2>&1 || fail "all_tests red"
  if [ "$PHASE" != prep-host ]; then
    ./build_gpu/hg_gpu_tests > "$ROOT/gate_gpu_tests.log" 2>&1 || fail "hg_gpu_tests red"
    ./build_gpu/gpu_differential_tests > "$ROOT/gate_differential.log" 2>&1 || fail "differential red"
  else
    say "host-only prep: the GPU build and its two suites are deferred to prep-gpu"
  fi
  say "gates green: $(grep -h "PASSED" "$ROOT"/gate_*.log | tr '\n' ' ')"
fi

# --------------------------------------------------------------------------- prep-gpu
# The GPU half alone, for a box whose driver arrived after the host work started.
if want prep-gpu; then
  [ -d "$SRC/.git" ] || fail "no clone on this box — run prep-host or prep first"
  cd "$SRC" || fail "no $SRC"
  build_gpu_targets
  ./build_gpu/hg_gpu_tests > "$ROOT/gate_gpu_tests.log" 2>&1 || fail "hg_gpu_tests red"
  ./build_gpu/gpu_differential_tests > "$ROOT/gate_differential.log" 2>&1 || fail "differential red"
  say "gpu gates green: $(grep -h "PASSED" "$ROOT"/gate_gpu_tests.log "$ROOT"/gate_differential.log | tr '\n' ' ')"
fi

# --------------------------------------------------------------------------- sync
# THE ITERATION LOOP: pull a newly pushed commit and rebuild incrementally. No apt, no gates,
# no reconfigure -- seconds to a couple of minutes for a host-only change, because CMake
# rebuilds what the edit touched. Use `prep` once per box and `sync` for every commit after.
if want sync; then
  [ -d "$SRC/.git" ] || fail "no clone on this box — run the prep phase first"
  git -C "$SRC" fetch --all -q >> "$LOG" 2>&1 || fail "fetch failed"
  clean_tree
  git -C "$SRC" checkout -q --detach "$(resolve_ref "$COMMIT")" >> "$LOG" 2>&1 \
    || fail "checkout $COMMIT failed"
  say "sync -> $(git -C "$SRC" rev-parse --short HEAD)  $(git -C "$SRC" log -1 --format=%s | head -c 60)"
  cd "$SRC" || fail "no $SRC"
  build_host
  build_gpu_targets
fi

cd "$SRC" 2>/dev/null || { want prep || fail "no clone on this box — run the prep phase first"; }

# --------------------------------------------------------------------------- attrib
# THE INSTRUMENTED BUILD, IN ITS OWN DIRECTORY, AND NEVER THE SOURCE OF A WALL NUMBER.
# HG_PHASE_TIMING compiles in per-phase cycle counters; the option is off by default precisely
# so the timing build carries none of it, and mixing the two would report an instrumented
# workload's wall time as the engine's. Same optimisation as the timing build (Release, which
# turns on LTO) plus -g, which adds symbols WITHOUT changing codegen, so callgrind and perf
# attribute to file and line instead of to a mangled name.
if want attrib; then
  say "instrumented build (separate dir; wall numbers never come from here)"
  cmake -S . -B build_instr "${COMMON_FLAGS[@]}" -DBUILD_GPU=OFF -DHG_PHASE_TIMING=ON \
    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -g" >> "$LOG" 2>&1 || fail "instr configure"
  cmake --build build_instr -j"$(nproc)" --target bench_cpu_evolve >> "$LOG" 2>&1 || fail "instr build"
  wait_quiet
  say "phase attribution (wpp d7 and the quotient workload)"
  ./build_instr/bench_cpu_evolve 7 3 "$SWEEP" wpp "$CPUSET" > "$ROOT/attrib_wpp.log" 2>&1 || true
  ./build_instr/bench_cpu_evolve 3 3 "$SWEEP" disc-l3a2g2r2 "$CPUSET" \
    > "$ROOT/attrib_disc.log" 2>&1 || true
  if command -v valgrind >/dev/null; then
    say "callgrind (single thread, file:line attribution)"
    valgrind --tool=callgrind --callgrind-out-file="$ROOT/cg_wpp.out" \
      ./build_instr/bench_cpu_evolve 6 1 1 wpp >> "$LOG" 2>&1 || true
    command -v callgrind_annotate >/dev/null \
      && callgrind_annotate "$ROOT/cg_wpp.out" > "$ROOT/cg_wpp.txt" 2>/dev/null || true
  fi
fi

# --------------------------------------------------------------------------- tables
if want tables; then
  wait_quiet
  say "paper tables (pinned, no --wolfram: authority macros carry forward)"
  python3 -u tools/dev/paper_tables.py --gpu --authority-depth 7 --steps 7 \
    --cpus "$CPUSET" --thread-sweep "$SWEEP" \
    --build-dir build_release --gpu-build-dir build_gpu_release 2>&1 | tee "$ROOT/paper_tables.log"
  [ "${PIPESTATUS[0]}" = 0 ] || fail "paper_tables failed"
fi

# --------------------------------------------------------------------------- sweep
if want sweep; then
  wait_quiet
  say "scaling sweep (pinned to $CPUSET, sweep $SWEEP)"
  # --cpus AND --thread-sweep, because without them cpu_scaling measures UNPINNED and this
  # phase runs AFTER tables -- so its t8_scaling.tex replaces the pinned one that generator
  # wrote, and the figure the paper renders is the unpinned measurement. That is the exact
  # substitution cpu_scaling's docstring says was fixed: the fix added the parameters, and this
  # caller never passed them, so the defect survived its own repair.
  #
  # Measured on this box, wpp depth 7, the two side by side: 2 threads is 2.09x pinned and
  # 1.69x unpinned. All sixteen cores are the same speed here, so what the pin set buys is
  # PLACEMENT -- an EPYC 9174F gives 16 cores eight L3 instances, and the same pair of threads
  # differs by 21% depending on whether they share one.
  python3 -u tools/dev/scaling_sweep.py --sections cpu,shapes,memory,gpu \
    --cpus "$CPUSET" --thread-sweep "$SWEEP" \
    --build-dir build_release --gpu-build-dir build_gpu_release 2>&1 | tee "$ROOT/scaling_sweep.log"
  [ "${PIPESTATUS[0]}" = 0 ] || fail "scaling_sweep failed"
fi

# --------------------------------------------------------------------------- floor
if want floor; then
  wait_quiet
  say "device floor (P4.11 adjudication)"
  ./build_gpu_release/bench_gpu_evolve 4 20 2 triangle > "$ROOT/floor_triangle.log" 2>&1 || true
  ./build_gpu_release/bench_gpu_evolve 7 5 2 wpp      > "$ROOT/floor_wpp.log"      2>&1 || true
  if command -v ncu >/dev/null && ncu --query-metrics >/dev/null 2>&1; then
    say "ncu captures"
    ncu --set full -o "$ROOT/ncu_wpp" ./build_gpu_release/bench_gpu_evolve 6 1 2 wpp >> "$LOG" 2>&1 || true
  fi
fi

say "phase '$PHASE' complete — the driver pulls its artifacts now; nothing here is a last copy"
