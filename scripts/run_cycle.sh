#!/usr/bin/env bash
# Nightly cycle; wire into launchd or cron. Logs to data/cycle.log.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
evolve cycle 2>&1 | tee -a data/cycle.log
