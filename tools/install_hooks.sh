#!/usr/bin/env bash
# Install the repo's git hooks into this clone. Run once per machine:
#   ./tools/install_hooks.sh
# Currently: post-commit auto-push to every remote (the multi-machine workflow
# coordinates through a local bare remote + origin; a commit that is not pushed
# is invisible to the other machine's session).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

cat > .git/hooks/post-commit <<'HOOK'
#!/usr/bin/env bash
# Auto-push every commit to all remotes. Non-fatal on failure so a network
# hiccup never blocks committing; a failed push is printed loudly to re-run.
branch="$(git symbolic-ref --short HEAD 2>/dev/null)" || exit 0
for remote in $(git remote); do
    if git push "$remote" "$branch" >/dev/null 2>&1; then
        echo "  [post-commit] pushed $branch -> $remote"
    else
        echo "  [post-commit] !! FAILED to push $branch -> $remote — run: git push $remote $branch" >&2
    fi
done
HOOK
chmod +x .git/hooks/post-commit
echo "installed: post-commit (auto-push to all remotes)"
