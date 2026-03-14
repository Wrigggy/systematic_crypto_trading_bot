#!/usr/bin/env bash
# ============================================================================
# Run the test suite with coverage report.
#
# Examples:
#   ./scripts/test.sh                                   # full suite + coverage
#   ./scripts/test.sh tests/test_strategy.py            # single file
#   ./scripts/test.sh -k "test_buy"                     # keyword filter
#   ./scripts/test.sh --no-cov                          # skip coverage
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

# Check if --no-cov was passed
COV_FLAGS="--cov=. --cov-report=term-missing"
for arg in "$@"; do
    if [ "$arg" = "--no-cov" ]; then
        COV_FLAGS=""
        # Remove --no-cov from args (pytest doesn't understand it)
        set -- $(echo "$@" | sed 's/--no-cov//')
        break
    fi
done

pytest $COV_FLAGS "$@"
