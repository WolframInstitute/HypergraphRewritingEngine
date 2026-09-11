#!/usr/bin/env bash
# The engine-commit contract, checked on the message being committed: a commit that stages
# engine sources carries a "Measurement-inert:" line, or every paper fragment goes stale and the
# paper job fails after the automatic push. pre-commit cannot check this: it runs before the
# message exists. The rule and the engine directories are paper_integrity_check.py's.
#
# Install on a clone:  ln -sf ../../tools/dev/commit_msg_gate.sh .git/hooks/commit-msg
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
exec python3 tools/dev/paper_integrity_check.py --commit-msg "$1"
