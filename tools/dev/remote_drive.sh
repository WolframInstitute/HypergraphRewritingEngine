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

TARGET="${1:?usage: remote_drive.sh <ssh-target> [commit] [phase...]}"
COMMIT="${2:-master}"
shift 2 2>/dev/null || shift $#
PHASES=("$@")
[ ${#PHASES[@]} -gt 0 ] || PHASES=(prep tables sweep floor)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
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
  ssh "$TARGET" 'cd "$HOME/hg_session" 2>/dev/null && tar cz --ignore-failed-read \
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
    ssh "$TARGET" "HG_ACCEPT_CONTENDED=${HG_ACCEPT_CONTENDED:-0} bash -s" < "$script" \
      2>&1 | tee -a "$RUN/driver.log"
  else
    ssh "$TARGET" "HG_ACCEPT_CONTENDED=${HG_ACCEPT_CONTENDED:-0} bash -s -- '$COMMIT' '$phase'" < "$script" \
      2>&1 | tee -a "$RUN/driver.log"
  fi
  rc=${PIPESTATUS[0]}

  pull "$phase"                       # ALWAYS, including after a failure
  if [ "$rc" != 0 ]; then
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
