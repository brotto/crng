"""
Freeze the canonical snapshot of real-market data for CRNG benchmarking.

This script downloads 7 assets × 5 years of daily close prices via yfinance
and saves them as a single CSV file at:

    benchmarks/snapshot_2026-04/prices.csv
    benchmarks/snapshot_2026-04/prices.sha256
    benchmarks/snapshot_2026-04/metadata.json

After this script runs once, the snapshot is IMMUTABLE. Any future benchmark
that claims "CRNG reproduces X/Y metrics on 7 real assets" must read from
this frozen CSV, not re-download data.

A second script, `rolling_benchmark.py`, is the honest "live" counterpart
that re-downloads each time and reports state-of-the-world. Never cite the
rolling one as evidence of a stable claim.

See SPECS.md P2 (frozen evidence per claim).

Run ONCE:
    cd crng-package && PYTHONPATH=. python benchmarks/freeze_snapshot_2026-04.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshot_2026-04")
CSV_PATH = os.path.join(SNAPSHOT_DIR, "prices.csv")
SHA_PATH = os.path.join(SNAPSHOT_DIR, "prices.sha256")
META_PATH = os.path.join(SNAPSHOT_DIR, "metadata.json")

SYMBOLS = {
    "Gold":    "GC=F",
    "EURUSD":  "EURUSD=X",
    "ETH":     "ETH-USD",
    "BTC":     "BTC-USD",
    "SP500":   "^GSPC",
    "Oil":     "CL=F",
    "USDJPY":  "JPY=X",
}


def main():
    if os.path.exists(CSV_PATH):
        print(f"Snapshot already exists at {CSV_PATH}")
        print("Per SPECS.md P2, snapshots are IMMUTABLE. Delete manually if you")
        print("really want to re-freeze (and update the README SHA reference).")
        return 1

    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed. Cannot freeze snapshot without it.")
        return 2

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    print(f"Freezing snapshot at {datetime.now(timezone.utc).isoformat()}")
    print(f"Downloading {len(SYMBOLS)} assets, 5y daily history from yfinance.")

    frames = {}
    for name, symbol in SYMBOLS.items():
        print(f"  {name} ({symbol})...", end=" ", flush=True)
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="5y", interval="1d")
        if df is None or len(df) < 100:
            print(f"insufficient data ({0 if df is None else len(df)} days)")
            continue
        frames[name] = df["Close"].dropna()
        print(f"{len(frames[name])} days")

    if not frames:
        print("No data downloaded. Aborting.")
        return 3

    # Align on union of dates, leave NaN where missing (FX and equity have
    # different holidays; crypto trades 7 days a week)
    prices = pd.DataFrame(frames)
    prices.index.name = "date"
    prices = prices.sort_index()

    # Sanitize timezone info from index for CSV round-tripping
    if prices.index.tz is not None:
        prices.index = prices.index.tz_convert("UTC").tz_localize(None)

    prices.to_csv(CSV_PATH)

    # Hash
    with open(CSV_PATH, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    with open(SHA_PATH, "w") as fh:
        fh.write(digest + "\n")

    # Metadata
    meta = {
        "snapshot_name": "snapshot_2026-04",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "source": "yfinance (Yahoo Finance)",
        "period": "5y",
        "interval": "1d",
        "column_is": "log-return series will be computed by consumers; this CSV holds CLOSE prices",
        "symbols": SYMBOLS,
        "n_rows": int(len(prices)),
        "date_range": {
            "start": str(prices.index.min().date()),
            "end":   str(prices.index.max().date()),
        },
        "columns": list(prices.columns),
        "non_nan_counts": {col: int(prices[col].notna().sum()) for col in prices.columns},
        "sha256": digest,
        "immutable": True,
        "notes": (
            "Per SPECS.md P2, this file is IMMUTABLE once written. Do not edit. "
            "To produce a new snapshot, create snapshot_2026-05/ etc. "
            "Log-returns are computed at read time by consumers "
            "(benchmarks/frozen_benchmark.py) so the raw CSV stays canonical."
        ),
    }
    with open(META_PATH, "w") as fh:
        json.dump(meta, fh, indent=2)

    print()
    print(f"Snapshot CSV  : {CSV_PATH}")
    print(f"SHA256        : {digest}")
    print(f"Metadata JSON : {META_PATH}")
    print(f"Rows          : {len(prices)}")
    print(f"Date range    : {meta['date_range']['start']} → {meta['date_range']['end']}")
    print(f"Columns       : {list(prices.columns)}")
    print()
    print("This snapshot is now IMMUTABLE. Reference it in the README by SHA256.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
