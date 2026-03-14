#!/usr/bin/env bash
# ============================================================================
# Generate synthetic training data to CSV files.
#
# Creates GBM (geometric Brownian motion) candle data in data/historical/.
# Useful when exchange APIs are unavailable or for reproducible experiments.
#
# Examples:
#   ./scripts/generate_data.sh                          # defaults
#   N_CANDLES=50000 SEED=123 ./scripts/generate_data.sh # custom
# ============================================================================
set -euo pipefail

N_CANDLES="${N_CANDLES:-10000}"
SEED="${SEED:-42}"
SYMBOLS="${SYMBOLS:-BTC/USDT ETH/USDT}"

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== Generating Synthetic Data ==="
echo "Symbols:  $SYMBOLS"
echo "Candles:  $N_CANDLES per symbol"
echo "Seed:     $SEED"
echo ""

python -c "
import csv
from pathlib import Path
from models.train import generate_synthetic_ohlcv, SYNTHETIC_PRESETS

out_dir = Path('data/historical')
out_dir.mkdir(parents=True, exist_ok=True)

for sym in '$SYMBOLS'.split():
    preset = SYNTHETIC_PRESETS.get(sym, {})
    candles = generate_synthetic_ohlcv(
        symbol=sym, n_candles=$N_CANDLES, seed=$SEED,
        start_price=preset.get('start_price', 1000.0),
        drift=preset.get('drift', 0.0001),
        vol=preset.get('vol', 0.002),
        base_volume=preset.get('base_volume', 100.0),
    )
    csv_name = sym.replace('/', '_') + '_synthetic.csv'
    csv_path = out_dir / csv_name
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['symbol','timestamp','open','high','low','close','volume'])
        for c in candles:
            w.writerow([c.symbol, c.timestamp.isoformat(), c.open, c.high, c.low, c.close, c.volume])
    print(f'  Wrote {len(candles)} candles -> {csv_path}')
"

echo ""
echo "Done! Data saved to data/historical/"
