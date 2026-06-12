#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

message="${1:-Update SFAR code}"

python -m compileall main.py sfar tools

git add -A -- \
  .gitignore \
  README.md \
  requirements.txt \
  configs \
  main.py \
  sfar \
  tools \
  push_code_only.sh

if git diff --cached --quiet; then
  echo "No code changes to commit."
  exit 0
fi

git status --short
git commit -m "$message"
git pull --rebase origin main
git push origin main
