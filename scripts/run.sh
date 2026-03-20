#!/usr/bin/env bash
# Usage: ./scripts/run.sh [paper|roostoo|live]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && { set -a; source .env; set +a; }
source .venv/bin/activate
MODE="${1:-paper}"
shift || true

EXTRA_ARGS=()
if [ "$MODE" = "roostoo" ] && [ $# -eq 0 ] && [ -n "${STRATEGY_PROFILE:-}" ]; then
    EXTRA_ARGS+=(--strategy-profile "$STRATEGY_PROFILE")
fi

exec .venv/bin/python3 main.py --mode "$MODE" "${EXTRA_ARGS[@]}" "$@"
