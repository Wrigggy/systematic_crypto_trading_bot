"""Download public spot minute archives; retain raw bars and audit time continuity.

No credentials, fills, forward filling, or model features are involved. Full-file
hashing is intentionally omitted; ZIP integrity, dimensions and bar invariants
are checked. Only complete monthly archives are accepted.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import time
import urllib.request
import zipfile

import numpy as np

BASE = 'https://data.binance.vision/data/spot/monthly/klines'


def month_bounds(month):
    start = datetime.strptime(month, '%Y-%m').replace(tzinfo=timezone.utc)
    end = start.replace(year=start.year + 1, month=1) if start.month == 12 else start.replace(month=start.month + 1)
    return int(start.timestamp()), int(end.timestamp())


def months_between(first, last):
    start, end = month_bounds(first)[0], month_bounds(last)[1]
    if start >= end:
        raise ValueError('Reversed month range')
    months = []
    while start < end:
        month = datetime.fromtimestamp(start, timezone.utc).strftime('%Y-%m')
        months.append(month)
        start = month_bounds(month)[1]
    return months


def parse_archive(raw, start, end):
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        if len(names) != 1 or not names[0].endswith('.csv'):
            raise ValueError('Expected exactly one CSV member')
        with archive.open(names[0]) as stream:
            bars = np.loadtxt(stream, delimiter=',', ndmin=2)
    if bars.shape != ((end - start) // 60, 12) or not np.isfinite(bars).all():
        raise ValueError('Invalid shape, missing minutes, or nonfinite archive')
    # Spot archives changed milliseconds to microseconds on January 1, 2025.
    scale = 1_000_000 if bars[0, 0] >= 1e14 else 1000
    times = (bars[:, 0] / scale).astype(np.int64)
    expected = np.arange(start, end, 60)
    if not np.array_equal(times, expected) or not np.array_equal(bars[:, 0], expected * scale):
        raise ValueError('Duplicate, missing, misaligned or wrong-month timestamps')
    if not np.array_equal(bars[:, 6], (expected + 60) * scale - 1):
        raise ValueError('Invalid close boundary')
    if (np.any(bars[:, 1:5] <= 0) or np.any(bars[:, 5] < 0)
            or np.any(bars[:, 2] < np.maximum(bars[:, 1], bars[:, 4]))
            or np.any(bars[:, 3] > np.minimum(bars[:, 1], bars[:, 4]))
            or np.any(bars[:, 7:11] < 0)):
        raise ValueError('Invalid OHLC or volume')
    return times, bars


def download_one(root, symbol, month):
    if not symbol.isalnum() or not symbol.endswith('USDT'):
        raise ValueError('Expected an alphanumeric USDT spot symbol')
    start, end = month_bounds(month)
    if end > time.time():
        raise ValueError('Refusing a future or incomplete month')
    path = root / f'{symbol}-1m-{month}.zip'
    url = f'{BASE}/{symbol}/1m/{path.name}'
    reused = path.exists()
    if reused:
        raw = path.read_bytes()
    else:
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=45) as response:
                    raw = response.read(15_000_001)
                if len(raw) > 15_000_000:
                    raise ValueError('Archive exceeds expected minute-data size')
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(1 + attempt)
    times, bars = parse_archive(raw, start, end)
    if not reused:
        temp = path.with_suffix('.zip.part')
        temp.write_bytes(raw)
        temp.replace(path)
    record = dict(symbol=symbol, month=month, url=url, bytes=len(raw), rows=len(times),
                  start=start, end=end, reused=reused, timestamp_unit='microseconds' if bars[0, 0] >= 1e14 else 'milliseconds')
    print(json.dumps(record), flush=True)
    return times, bars, record


def download(root, symbols, first, last):
    root.mkdir(parents=True, exist_ok=True)
    months = months_between(first, last)
    manifest = dict(source=BASE, interval_seconds=60, symbols=symbols, first=first, last=last,
                    full_file_hashes=False, validation='ZIP CRC, finite OHLCV, exact full-month grid', records=[])
    for symbol in symbols:
        with ThreadPoolExecutor(max_workers=3) as pool:
            values = list(pool.map(lambda m: download_one(root, symbol, m), months))
        times = np.concatenate([v[0] for v in values])
        bars = np.concatenate([v[1] for v in values])
        if np.any(np.diff(times) != 60):
            raise ValueError('Cross-month continuity failed')
        dest = root / f'{symbol}.npz'
        with dest.with_suffix('.npz.part').open('wb') as stream:
            np.savez_compressed(stream, times=times, bars=bars)
        dest.with_suffix('.npz.part').replace(dest)
        manifest['records'].extend(v[2] for v in values)
        (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    manifest.update(status='complete', total_rows=sum(r['rows'] for r in manifest['records']),
                    downloaded_bytes=sum(r['bytes'] for r in manifest['records'] if not r['reused']))
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'XRPUSDT', 'BNBUSDT'])
    parser.add_argument('--first', default='2025-09')
    parser.add_argument('--last', default='2026-08')
    args = parser.parse_args()
    download(args.out, args.symbols, args.first, args.last)
