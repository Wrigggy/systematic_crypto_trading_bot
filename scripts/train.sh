#!/usr/bin/env bash
# ============================================================================
# Train the LSTM alpha model.
#
# Uses synthetic GBM data by default (no API keys needed).
# Outputs: artifacts/model.pt, artifacts/model.onnx
# Logs:    logs/train_<timestamp>.log
#
# Examples:
#   ./scripts/train.sh                          # defaults (synthetic, CPU)
#   DEVICE=cuda ./scripts/train.sh              # train on GPU
#   EPOCHS=100 USE_AMP=--amp ./scripts/train.sh # 100 epochs with mixed precision
# ============================================================================
set -euo pipefail

# ── Configurable ──────────────────────────────────────────────────────────
SYMBOLS="${SYMBOLS:-BTC/USDT}"          # space-separated for multiple
N_CANDLES="${N_CANDLES:-10000}"          # candles per symbol
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-0.001}"
DEVICE="${DEVICE:-auto}"                # "cpu", "cuda", or "auto"
SEED="${SEED:-42}"
USE_AMP="${USE_AMP:-}"                  # set to "--amp" for mixed precision
USE_COMPILE="${USE_COMPILE:-}"          # set to "--compile" for torch.compile
# ──────────────────────────────────────────────────────────────────────────

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== Training LSTM Alpha Model ==="
echo "Symbols:    $SYMBOLS"
echo "Candles:    $N_CANDLES per symbol"
echo "Epochs:     $EPOCHS"
echo "Device:     $DEVICE"
echo "AMP:        ${USE_AMP:-disabled}"
echo "Compile:    ${USE_COMPILE:-disabled}"
echo ""

python -m models.train \
    --symbols $SYMBOLS \
    --synthetic \
    --n-candles "$N_CANDLES" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --device "$DEVICE" \
    --seed "$SEED" \
    $USE_AMP $USE_COMPILE
