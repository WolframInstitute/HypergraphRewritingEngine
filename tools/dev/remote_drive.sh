#!/usr/bin/env bash
# Drive the rented box from HERE, so the box's filesystem is never load-bearing.
#
#   tools/dev/remote_drive.sh <ssh-target> [commit] [phase...]
#   tools/dev/remote_drive.sh epyc                       # prep, tables, sweep, floor
#   tools/dev/remote_drive.sh epyc dc4643bf tables       # re-run one phase
#   tools/dev/remote_drive.sh epyc HEAD tuning           # phase 2 experiment
#
# THE RULE THIS ENFORCES: the box is ephemeral, so no artifact may exist only there. Each
# phase runs over ssh with its output teed locally as it is produced, and the moment the phase
# returns -- succeeded OR failed -- everything it wrote is pulled down. A box that dies costs
# the build time of the phase in flight and nothing else, because every completed phase is
# already on this machine.
#
# Losing the box entirely is a full re-run of `prep` on a new one: the source comes from the
# public repository and the builds are reproducible, so nothing needs recovering FROM the old
# box. That is the whole reason prep is separated from the measuring phases.
#
# The pull uses tar over ssh rather than rsync: ssh and tar are on every box, rsync is not.
set -uo pipefail

# A FRESH RENTED BOX HAS AN UNKNOWN HOST KEY, and the default prompt would hang a
# non-interactive run forever; accept-new records it without ever silently accepting a
# CHANGED one. The keepalives matter because a measuring phase can be silent for many minutes
# -- paper_tables between tables, a CUDA build between targets -- and a NAT or firewall that
# times an idle connection out would kill the phase in flight.
# SHARING THE BOX WITH THE OTHER PROJECT (../plr). Both sides do timing-sensitive work, so a
# build or a benchmark from one destroys the other's numbers. Every remote invocation here is
# wrapped in flock on a well-known path, which gives mutual exclusion AND a queue: a waiting
# side blocks until the holder finishes rather than racing it.
#
# flock was chosen over a hand-rolled lock because the kernel releases it when the holding
# process dies -- a dropped ssh, a killed phase, a rebooted box leave NO stale lock, which on
# an ephemeral machine is the failure that would otherwise need manual clearing.
#
# THE LOCK IS TAKEN BY THE REMOTE SCRIPTS THEMSELVES, not wrapped around them here: a
# caller that forgets to wrap protects nothing, and wrapping here as well would make parent
# and child contend for the same lock and deadlock until the timeout. The contract for the
# other project is therefore the same one this project follows -- take the lock inside your
# remote entry point, on a held file descriptor, and name yourself in /tmp/hgbox.holder.
# Whoever waits can then read that file to see who has the box and since when.
BOX_LOCK=/tmp/hgbox.lock
BOX_HOLDER_PATH=/tmp/hgbox.holder
LOCK_WAIT="${LOCK_WAIT:-7200}"          # 2h: longer than any single phase, shorter than a day

SSH_OPTS=(-o StrictHostKeyChecking=accept-new
          -o ConnectTimeout=15
          -o ServerAliveInterval=30
          -o ServerAliveCountMax=10)

TARGET="${1:?usage: remote_drive.sh <ssh-target> [commit] [phase...]}"
COMMIT="${2:-master}"
shift 2 2>/dev/null || shift $#
PHASES=("$@")
[ ${#PHASES[@]} -gt 0 ] || PHASES=(prep tables sweep floor)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

# THE BOX MEASURES WHAT GITHUB HAS, NOT WHAT IS IN THIS WORKING TREE. It clones over HTTPS, so
# an unpushed commit is one it cannot check out, and a table stamped with a commit no one else
# can obtain is not a reproducible measurement. Resolved and verified here rather than
# discovered as a checkout failure after the ssh.
if [ "$COMMIT" = HEAD ] || [ "$COMMIT" = . ]; then
  COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
fi
if git -C "$ROOT" cat-file -e "${COMMIT}^{commit}" 2>/dev/null; then
  # Captured rather than piped into grep -q: under pipefail a matching grep -q can still
  # report failure, because it exits at the first line and the writer dies on SIGPIPE.
  remote_branches="$(git -C "$ROOT" branch -r --contains "$COMMIT" 2>/dev/null || true)"
  if [ -z "$remote_branches" ]; then
    echo "refusing: commit $COMMIT is on no remote branch — push it first, or the box cannot check it out" >&2
    exit 2
  fi
fi
if [ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ]; then
  echo "NOTE: this working tree has uncommitted changes; the box measures $COMMIT, not them." >&2
fi
RUN="$ROOT/remote_runs/$(date -u +%Y%m%dT%H%M%SZ)_${TARGET//[^A-Za-z0-9]/_}"
mkdir -p "$RUN"
echo "local run directory: $RUN"
echo "target=$TARGET commit=$COMMIT phases=${PHASES[*]}" | tee "$RUN/driver.log"

# Everything the box wrote, as a stream. --ignore-failed-read so a phase that produced only
# some of these still yields the ones it did; the trailing `true` keeps a partial pull from
# failing the driver, because a partial pull is still strictly better than none.
pull() {
  local phase="$1" dest="$RUN/$phase"
  mkdir -p "$dest"
  # shellcheck disable=SC2029  # deliberate remote-side expansion of $HOME
  ssh "${SSH_OPTS[@]}" "$TARGET" 'cd "$HOME/hg_session" 2>/dev/null && tar cz --ignore-failed-read \
      preflight.txt session.log ./*.log ./*.tsv ./ncu_* src/paper/tables 2>/dev/null' \
    | tar xz -C "$dest" 2>/dev/null || true
  local n; n=$(find "$dest" -type f | wc -l)
  echo "    pulled $n file(s) into $dest" | tee -a "$RUN/driver.log"
  [ "$n" -gt 0 ] || echo "    WARNING: nothing came back for phase '$phase'" | tee -a "$RUN/driver.log"
}

status=0
for phase in "${PHASES[@]}"; do
  echo "" | tee -a "$RUN/driver.log"
  echo "=== phase: $phase ($(date -u +%H:%M:%SZ))" | tee -a "$RUN/driver.log"
  script="$HERE/remote_session.sh"
  [ "$phase" = tuning ] && script="$HERE/remote_tuning.sh"

  # The script is piped in rather than copied, so the box never holds a stale version of it,
  # and HG_ACCEPT_CONTENDED is forwarded if the caller set it.
  if [ "$phase" = tuning ]; then
    ssh "${SSH_OPTS[@]}" "$TARGET" "HG_ACCEPT_CONTENDED=${HG_ACCEPT_CONTENDED:-0} HG_SWEEP=${HG_SWEEP:-} HG_CPUSET=${HG_CPUSET:-} bash -s" < "$script" \
      2>&1 | tee -a "$RUN/driver.log"
  else
    ssh "${SSH_OPTS[@]}" "$TARGET" "HG_ACCEPT_CONTENDED=${HG_ACCEPT_CONTENDED:-0} HG_SWEEP=${HG_SWEEP:-} HG_CPUSET=${HG_CPUSET:-} bash -s -- '$COMMIT' '$phase'" < "$script" \
      2>&1 | tee -a "$RUN/driver.log"
  fi
  rc=${PIPESTATUS[0]}

  pull "$phase"                       # ALWAYS, including after a failure
  if [ "$rc" != 0 ]; then
    # A flock timeout is indistinguishable from any other non-zero exit unless we say so.
    holder=$(ssh "${SSH_OPTS[@]}" "$TARGET" "cat $BOX_HOLDER_PATH 2>/dev/null" 2>/dev/null)
    [ -n "$holder" ] && echo "    the box is held by: $holder" | tee -a "$RUN/driver.log"
    echo "phase '$phase' exited $rc — stopping; what it produced is in $RUN/$phase" | tee -a "$RUN/driver.log"
    status=$rc
    break
  fi
  echo "$phase $(date -u +%FT%TZ) ok" >> "$RUN/completed.txt"
done

echo "" | tee -a "$RUN/driver.log"
echo "completed phases:" | tee -a "$RUN/driver.log"
cat "$RUN/completed.txt" 2>/dev/null | sed 's/^/  /' | tee -a "$RUN/driver.log"
TBL=$(find "$RUN" -path '*/paper/tables/*' -name '*.tex' | wc -l)
echo "paper table fragments retrieved: $TBL" | tee -a "$RUN/driver.log"
echo "the box holds nothing this machine now lacks; it can be destroyed" | tee -a "$RUN/driver.log"
exit "$status"
