#!/usr/bin/env bash
# ============================================================================
# Run paper trading with synthetic data.
#
# Uses config/default.yaml by default. Override with CONFIG env var.
# Logs to both console and logs/trading_paper_<timestamp>.log
#
# Examples:
#   ./scripts/paper_trade.sh                              # default config
#   CONFIG=config/fast.yaml ./scripts/paper_trade.sh      # custom config
#   ./scripts/paper_trade.sh --mode live                  # override to live
# ============================================================================
set -euo pipefail

CONFIG="${CONFIG:-config/default.yaml}"

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== Paper Trading ==="
echo "Config: $CONFIG"
echo "Press Ctrl+C to stop"
echo ""

.venv/bin/python3 main.py --config "$CONFIG" --mode paper "$@"
