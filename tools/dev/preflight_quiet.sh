#!/usr/bin/env bash
# A HARD GATE IN FRONT OF EVERY MEASUREMENT. Exits non-zero when the machine is not quiet, and
# chains the measurement behind itself so a dirty machine cannot produce a number at all.
#
#   tools/dev/preflight_quiet.sh -- ./tools/dev/corpus_scale_sweep.sh ...
#
# WHY IT IS A GATE AND NOT A WARNING. A timing run on a machine with other work on it produces a
# number that looks exactly like a clean one, and nothing downstream can tell them apart
# afterwards. The only place the distinction survives is here, before the run starts.
#
# AN ALLOWLIST, NOT A LIST OF THINGS TO LOOK FOR. Anything running that is not part of a logged-in
# session is a failure, whatever it is. A list of patterns to catch only catches what someone
# thought of: the wait-loop left by an earlier session -- `until ! pgrep -f ...; do sleep 10; done`
# -- ran across the whole process table every ten seconds on the benchmark box through an entire
# evening of measurements, and was found only when the box was handed to another project. It is
# not a benchmark, not a build, and has no binary of its own, so no denylist would have named it.
# It IS an unexpected process, which is all this needs to know.
#
# SELF-EXCLUSION IS BY PID, not by matching on names. This script's own lineage is the one thing
# legitimately running, and when a sweep chains this in front of itself the tree is
# `timeout -> sweep -> preflight`, so ancestors count as much as descendants.
set -u

WAIT_S=0
STRICT_USERS=${STRICT_USERS:-1}
EXTRA_ALLOW=${PREFLIGHT_ALLOW:-}

usage() {
    cat <<'USAGE'
usage: preflight_quiet.sh [--wait S] [--allow REGEX] [--allow-users] [-- CMD ...]

  --wait S       poll until quiet, up to S seconds, instead of failing at once. For a machine
                 whose last job is still exiting; the check itself is instantaneous.
  --allow REGEX  also treat processes whose command line matches REGEX as expected. For a
                 workstation that legitimately runs an editor or an agent; a rented benchmark
                 box needs none. Also settable as $PREFLIGHT_ALLOW.
  --allow-users  do not fail when another user is logged in
  -- CMD ...     run CMD only if every check passes; its exit status becomes ours

Exit: 0 quiet (and CMD's status, when given); 1 not quiet; 2 usage.
USAGE
}

CMD=()
while [ $# -gt 0 ]; do
    case "$1" in
        --wait)     WAIT_S="${2:?}"; shift 2 ;;
        --allow)    EXTRA_ALLOW="${EXTRA_ALLOW:+$EXTRA_ALLOW|}${2:?}"; shift 2 ;;
        --allow-users) STRICT_USERS=0; shift ;;
        -h|--help)  usage; exit 2 ;;
        --)         shift; CMD=("$@"); break ;;
        *)          echo "preflight: unknown argument '$1'" >&2; usage; exit 2 ;;
    esac
done

# This script's own lineage: itself, every ancestor to init, every descendant.
own_pids() {
    local out="$$" frontier="$$" next child p
    p=$$
    while [ -n "$p" ] && [ "$p" -gt 1 ] 2>/dev/null; do
        p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
        [ -n "$p" ] && out="$out $p"
    done
    for _ in 1 2 3 4 5; do
        next=""
        for p in $frontier; do
            for child in $(pgrep -P "$p" 2>/dev/null); do
                out="$out $child"; next="$next $child"
            done
        done
        [ -z "$next" ] && break
        frontier="$next"
    done
    echo "$out"
}

# What a logged-in session legitimately consists of, and nothing else: session plumbing and the
# small utilities a shell forks. No compilers, no interpreters, no shell running a loop. A rented
# box that has only been ssh'd into shows exactly these and nothing more.
ALLOW_COMM='^(sshd|systemd|\(sd-pam\)|dbus-daemon|bash|sh|zsh|dash|login|ps|awk|grep|sed|tr|head|tail|which|env)$'

# REAL USERS ONLY, by uid, not by "not root". A service account is not root either -- rabbitmq,
# syslog, polkitd, systemd-resolved and epmd all own long-lived daemons on an otherwise idle
# machine, and none of them is work someone put there. Login accounts start at 1000 on every
# distribution this runs on, and a benchmark is something a login account started.
offenders() {
    local mine; mine=" $(own_pids) "
    ps -eo pid,ppid,uid,user,etimes,comm,args --no-headers 2>/dev/null |
    while read -r pid ppid uid user etimes comm args; do
        case "$mine" in *" $pid "*) continue ;; esac
        # Kernel threads are the kernel, not work on the machine: pid 2 and its children.
        { [ "$pid" = "2" ] || [ "$ppid" = "2" ]; } && continue
        [ "$uid" -lt 1000 ] 2>/dev/null && continue
        printf '%s' "$comm" | grep -qE "$ALLOW_COMM" && continue
        [ -n "$EXTRA_ALLOW" ] && printf '%s' "$args" | grep -qE "$EXTRA_ALLOW" && continue
        printf '  pid=%-8s user=%-10s age=%-8ss %s\n' \
            "$pid" "$user" "$etimes" "$(printf '%.110s' "$args")"
    done
}

other_users() { who 2>/dev/null | awk '{print $1}' | sort -u | grep -vx "$(id -un)" | tr '\n' ' '; }

# NO LOAD-AVERAGE CHECK. It is a decaying mean over a minute, so it reports a machine busy long
# after its last job exited and reports it idle for the first seconds of a new one -- late in both
# directions. The process list answers the same question exactly and at the instant it is asked.
check_once() {
    local bad=0 found u
    found="$(offenders)"
    if [ -n "$found" ]; then
        echo "preflight: FAIL -- unexpected processes running:" >&2
        printf '%s\n' "$found" >&2
        bad=1
    fi
    if [ "$STRICT_USERS" = "1" ]; then
        u="$(other_users)"
        if [ -n "$u" ]; then
            echo "preflight: FAIL -- other users logged in: $u" >&2
            bad=1
        fi
    fi
    return $bad
}

deadline=$(( $(date +%s) + WAIT_S ))
while :; do
    if check_once; then break; fi
    [ "$(date +%s)" -ge "$deadline" ] && {
        echo "preflight: machine not quiet; refusing to measure" >&2
        exit 1
    }
    sleep 5
done

echo "preflight: quiet (nothing running but this session)"
[ ${#CMD[@]} -eq 0 ] && exit 0
exec "${CMD[@]}"
