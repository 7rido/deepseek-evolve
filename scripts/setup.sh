#!/usr/bin/env bash
# One-time setup on the Mac.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[mlx]"
git init -q 2>/dev/null || true
ln -sf ../../scripts/pre-commit .git/hooks/pre-commit
git add -A && git commit -qm "initial harness" || true
evolve pin-evals
echo "Pull the base model once (downloads ~4.5 GB for the 7B 4-bit):"
echo "  python -c \"from mlx_lm import load; load('$(grep 'base:' config.yaml | awk '{print $2}')')\""
