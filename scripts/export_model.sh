#!/usr/bin/env bash
# ============================================================================
# Export a trained PyTorch checkpoint (.pt) to ONNX format (.onnx).
#
# ONNX is the preferred inference format — 2-5x faster than PyTorch, no
# torch dependency in the hot loop.
#
# Usage:
#   ./scripts/export_model.sh                         # exports artifacts/model.pt
#   ./scripts/export_model.sh path/to/custom.pt       # exports custom checkpoint
# ============================================================================
set -euo pipefail

PT_PATH="${1:-artifacts/model.pt}"
ONNX_PATH="${PT_PATH%.pt}.onnx"
SEQ_LEN="${SEQ_LEN:-30}"
N_FEATURES="${N_FEATURES:-6}"

cd "$(dirname "$0")/.."
source .venv/bin/activate

if [ ! -f "$PT_PATH" ]; then
    echo "Error: $PT_PATH not found. Train a model first with ./scripts/train.sh"
    exit 1
fi

echo "=== Exporting Model ==="
echo "Input:    $PT_PATH"
echo "Output:   $ONNX_PATH"
echo "Seq len:  $SEQ_LEN"
echo "Features: $N_FEATURES"
echo ""

python -c "
import torch
from models.lstm_model import LSTMAlphaModel
from models.train import export_onnx

model = LSTMAlphaModel(n_features=$N_FEATURES)
model.load_state_dict(torch.load('$PT_PATH', map_location='cpu', weights_only=True))
model.eval()
export_onnx(model, $SEQ_LEN, $N_FEATURES, '$ONNX_PATH')
"

echo ""
echo "Done! ONNX model ready at $ONNX_PATH"
