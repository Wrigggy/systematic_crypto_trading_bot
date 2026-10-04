"""Add exact September minute flow for eligible E38 replay; no model training."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import urllib.request
import numpy as np

from scripts.download_minute_history import parse_archive, month_bounds


def prepare(source, out):
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for symbol in ('BTCUSDT', 'XRPUSDT', 'BNBUSDT'):
        start, end = month_bounds('2026-08')
        raw = (source / f'{symbol}-1m-2026-08.zip').read_bytes()
        times, bars = parse_archive(raw, start, end)
        def get(day):
            stamp = int(datetime(2026, 9, day, tzinfo=timezone.utc).timestamp())
            name = f'{symbol}-1m-2026-09-{day:02d}.zip'
            url = f'https://data.binance.vision/data/spot/daily/klines/{symbol}/1m/{name}'
            with urllib.request.urlopen(url, timeout=45) as response:
                payload = response.read(1_000_001)
            if len(payload) > 1_000_000:
                raise ValueError('Unexpected daily minute archive size')
            t, b = parse_archive(payload, stamp, stamp + 86400)
            (out / name).write_bytes(payload)
            print(name, len(t), flush=True)
            return t, b, dict(url=url, rows=len(t), bytes=len(payload))
        with ThreadPoolExecutor(max_workers=3) as pool:
            values = list(pool.map(get, range(1, 17)))
        times = np.concatenate([times] + [v[0] for v in values])
        bars = np.concatenate([bars] + [v[1] for v in values])
        if np.any(np.diff(times) != 60):
            raise ValueError('Non-contiguous September flow extension')
        np.savez_compressed(out / f'{symbol}.npz', times=times, bars=bars)
        records.extend(v[2] for v in values)
    (out / 'manifest.json').write_text(json.dumps(dict(status='complete', records=records,
        reused='August 2026 archives from E39', full_file_hashes=False,
        validation='ZIP CRC, exact minute grid, finite OHLCV'), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.out)
