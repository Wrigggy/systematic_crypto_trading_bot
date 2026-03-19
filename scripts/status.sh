#!/usr/bin/env bash
# Show current bot status: NAV, cash, positions, risk metrics.
# Usage: ./scripts/status.sh [logs_dir]
set -euo pipefail

LOGS_DIR="${1:-logs}"

# Find the most recent roostoo log, fallback to any trading log
LOG=$(ls -t "$LOGS_DIR"/trading_roostoo_*.log 2>/dev/null | head -1)
[ -z "$LOG" ] && LOG=$(ls -t "$LOGS_DIR"/trading_*.log 2>/dev/null | head -1)

if [ -z "$LOG" ]; then
    echo "No log files found in $LOGS_DIR"
    exit 1
fi

echo "=== Bot Status (from $LOG) ==="
echo ""

# Latest NAV/Holdings line
echo "--- Portfolio ---"
grep -E "Iter [0-9]+ \| NAV=" "$LOG" | tail -1 || echo "(no status yet — still warming up?)"
echo ""

# Latest risk metrics
echo "--- Risk Metrics ---"
grep "Risk Metrics" "$LOG" | tail -1 || echo "(not enough data yet)"
echo ""

# Recent trades (last 5)
echo "--- Recent Trades (last 5) ---"
grep -E "(BUY filled|SELL filled|STOP triggered|FLAT →|HOLDING →)" "$LOG" | tail -5 || echo "(no trades yet)"
echo ""

# Errors/warnings in last 50 lines
ERRORS=$(tail -50 "$LOG" | grep -c -E "\[(ERROR|CRITICAL)\]" || true)
if [ "$ERRORS" -gt 0 ]; then
    echo "--- Recent Errors ($ERRORS in last 50 lines) ---"
    tail -50 "$LOG" | grep -E "\[(ERROR|CRITICAL)\]"
fi

# Warmup status
WARMUP=$(grep "warming up" "$LOG" | tail -1)
if [ -n "$WARMUP" ]; then
    echo ""
    echo "--- Warmup ---"
    echo "$WARMUP"
fi
