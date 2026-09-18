#!/usr/bin/env bash
# Run a project module inside the running nfl-line container:
#   ./run.sh backtest | model | features | live --mock | injuries --dry | poll_results | poll_odds --phase open
set -euo pipefail
mod="${1:?usage: ./run.sh <module> [args]}"
exec docker exec nfl-line python -m "src.${mod}" "${@:2}"
