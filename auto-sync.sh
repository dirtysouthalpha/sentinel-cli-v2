#!/bin/bash
# Auto-sync sentinel-cli-v2 to GitHub
# Runs via cron: */15 * * * * /workspace/sentinel-cli-v2/auto-sync.sh >> /tmp/sentinel-cli-sync.log 2>&1

cd /workspace/sentinel-cli-v2 || exit 1

# Check for changes
if git diff --quiet && git diff --staged --quiet && [ -z "$(git ls-files --others --exclude-standard)" ]; then
    exit 0  # No changes
fi

# Stage, commit, push
git add -A
git commit -m "auto-sync: $(date '+%Y-%m-%d %H:%M:%S')"
git push origin main
