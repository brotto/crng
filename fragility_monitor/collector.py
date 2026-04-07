#!/usr/bin/env python3
"""
CRNG-Fragility Monitor — Data Collector
==========================================
Fetches commodity prices and economic indicators from public APIs.

Sources:
  - Yahoo Finance: oil, gas, VIX, indices, Baltic Dry
  - FRED API: yields, DXY, specific series (requires API key)
  - Direct download: FAO, World Bank (monthly)

Usage:
  python collector.py                  # Fetch all available data
  python collector.py --days 365       # Fetch last N days of history
  python collector.py --symbol BRENT   # Fetch specific symbol
  python collector.py --status         # Show collection status
"""

import argparse
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
from db import init_db, insert_raw, get_connection

# ══════════════════════════════════════════════════════════════════
# Symbol Definitions
# ══════════════════════════════════════════════════════════════════

SYMBOLS = {
    # ── Tier 1: Choke Point Direct (Hormuz) ──
    "BRENT": {
        "tier": 1, "category": "energy",
        "yahoo": "BZ=F",
        "unit": "USD/bbl",
        "desc": "Brent Crude Oil Futures"
    },
    "WTI": {
        "tier": 1, "category": "energy",
        "yahoo": "CL=F",
        "unit": "USD/bbl",
        "desc": "WTI Crude Oil Futures"
    },
    "NATGAS_US": {
        "tier": 1, "category": "energy",
        "yahoo": "NG=F",
        "unit": "USD/MMBtu",
        "desc": "Henry Hub Natural Gas Futures"
    },
    "NATGAS_EU": {
        "tier": 1, "category": "energy",
        "yahoo": "TTF=F",
        "unit": "EUR/MWh",
        "desc": "TTF European Natural Gas"
    },

    # ── Tier 2: Financial Stress ──
    "VIX": {
        "tier": 2, "category": "financial",
        "yahoo": "^VIX",
        "unit": "index",
        "desc": "CBOE Volatility Index (fear gauge)"
    },
    "DXY": {
        "tier": 2, "category": "financial",
        "yahoo": "DX-Y.NYB",
        "unit": "index",
        "desc": "US Dollar Index"
    },
    "YIELD_2Y": {
        "tier": 2, "category": "financial",
        "yahoo": "^IRX",  # 13-week T-bill as proxy
        "fred": "DGS2",
        "unit": "pct",
        "desc": "US 2-Year Treasury Yield"
    },
    "YIELD_10Y": {
        "tier": 2, "category": "financial",
        "yahoo": "^TNX",
        "unit": "pct",
        "desc": "US 10-Year Treasury Yield"
    },
    "GOLD": {
        "tier": 2, "category": "financial",
        "yahoo": "GC=F",
        "unit": "USD/oz",
        "desc": "Gold Futures (safe haven)"
    },

    # ── Tier 3: Physical Economy ──
    "BALTIC_DRY": {
        "tier": 3, "category": "shipping",
        "yahoo": "^BDIY",  # May not be available
        "unit": "index",
        "desc": "Baltic Dry Index (physical trade proxy)"
    },
    "SOX": {
        "tier": 3, "category": "tech",
        "yahoo": "^SOX",
        "unit": "index",
        "desc": "PHLX Semiconductor Index"
    },
    "SP500": {
        "tier": 3, "category": "market",
        "yahoo": "^GSPC",
        "unit": "index",
        "desc": "S&P 500"
    },

    # ── Tier 4: AI Bubble ──
    "NASDAQ": {
        "tier": 4, "category": "tech",
        "yahoo": "^IXIC",
        "unit": "index",
        "desc": "NASDAQ Composite"
    },
    "NVDA": {
        "tier": 4, "category": "tech",
        "yahoo": "NVDA",
        "unit": "USD",
        "desc": "NVIDIA (AI bellwether)"
    },
    "MSFT": {
        "tier": 4, "category": "tech",
        "yahoo": "MSFT",
        "unit": "USD",
        "desc": "Microsoft"
    },
    "META": {
        "tier": 4, "category": "tech",
        "yahoo": "META",
        "unit": "USD",
        "desc": "Meta Platforms"
    },
}


# ══════════════════════════════════════════════════════════════════
# Yahoo Finance Collector
# ══════════════════════════════════════════════════════════════════

def fetch_yahoo(symbol_key, days=365):
    """Fetch historical data from Yahoo Finance."""
    import yfinance as yf

    sym = SYMBOLS[symbol_key]
    ticker_id = sym.get("yahoo")
    if not ticker_id:
        return None

    try:
        ticker = yf.Ticker(ticker_id)
        end = datetime.now()
        start = end - timedelta(days=days)

        hist = ticker.history(start=start.strftime("%Y-%m-%d"),
                              end=end.strftime("%Y-%m-%d"))

        if hist.empty:
            print(f"    {symbol_key}: no data from Yahoo ({ticker_id})")
            return None

        records = []
        for idx, row in hist.iterrows():
            date = idx.strftime("%Y-%m-%d")
            close = row.get("Close")
            if close is not None and close > 0:
                records.append((date, float(close)))

        print(f"    {symbol_key}: {len(records)} days from Yahoo ({ticker_id})")
        return records

    except Exception as e:
        print(f"    {symbol_key}: Yahoo error — {e}")
        return None


# ══════════════════════════════════════════════════════════════════
# FRED Collector (optional, requires API key)
# ══════════════════════════════════════════════════════════════════

def fetch_fred(symbol_key, days=365):
    """Fetch data from FRED API (if key available)."""
    fred_key = os.environ.get("FRED_API_KEY")
    if not fred_key:
        return None

    sym = SYMBOLS[symbol_key]
    series_id = sym.get("fred")
    if not series_id:
        return None

    try:
        from fredapi import Fred
        fred = Fred(api_key=fred_key)
        end = datetime.now()
        start = end - timedelta(days=days)

        data = fred.get_series(series_id, start, end)
        records = []
        for date, value in data.items():
            if value is not None:
                records.append((date.strftime("%Y-%m-%d"), float(value)))

        print(f"    {symbol_key}: {len(records)} days from FRED ({series_id})")
        return records

    except Exception as e:
        print(f"    {symbol_key}: FRED error — {e}")
        return None


# ══════════════════════════════════════════════════════════════════
# Unified Collector
# ══════════════════════════════════════════════════════════════════

def collect_all(days=365, symbols=None):
    """Collect all symbols from best available source."""
    captured_at = datetime.utcnow().isoformat() + "Z"

    if symbols is None:
        symbols = list(SYMBOLS.keys())

    print(f"\n[COLLECTOR] Fetching {len(symbols)} symbols ({days} days of history)")
    print(f"  Captured at: {captured_at}")
    print(f"{'='*60}")

    total_records = 0
    success = 0
    failed = []

    for sym_key in symbols:
        sym = SYMBOLS[sym_key]

        # Try FRED first (more reliable for financial data)
        records = fetch_fred(sym_key, days)

        # Fallback to Yahoo Finance
        if records is None:
            records = fetch_yahoo(sym_key, days)

        if records is None or len(records) == 0:
            failed.append(sym_key)
            continue

        # Store in DB
        for date, value in records:
            insert_raw(date, sym_key, sym['tier'], sym['category'],
                      value, "yahoo", unit=sym.get('unit'),
                      captured_at=captured_at)

        total_records += len(records)
        success += 1

    print(f"{'='*60}")
    print(f"[COLLECTOR] Done: {success}/{len(symbols)} symbols, {total_records} total records")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print()

    return success, failed


def show_status():
    """Show collection status."""
    conn = get_connection()

    print(f"\n{'='*70}")
    print(f"  FRAGILITY MONITOR — DATA STATUS")
    print(f"{'='*70}")

    rows = conn.execute("""
    SELECT symbol, tier, category, COUNT(*) as n,
           MIN(date) as first, MAX(date) as last,
           source
    FROM raw_data
    GROUP BY symbol
    ORDER BY tier, symbol
    """).fetchall()

    if not rows:
        print("  No data collected yet. Run: python collector.py --days 365")
        print(f"{'='*70}\n")
        conn.close()
        return

    current_tier = 0
    tier_names = {1: "CHOKE POINT (Hormuz)", 2: "FINANCIAL STRESS",
                  3: "PHYSICAL ECONOMY", 4: "AI BUBBLE"}

    for r in rows:
        if r['tier'] != current_tier:
            current_tier = r['tier']
            print(f"\n  Tier {current_tier}: {tier_names.get(current_tier, '?')}")
            print(f"  {'Symbol':<15} {'Days':>5} {'First':>12} {'Last':>12} {'Source':<8}")
            print(f"  {'-'*55}")

        print(f"  {r['symbol']:<15} {r['n']:>5} {r['first']:>12} {r['last']:>12} {r['source']:<8}")

    # Latest values
    print(f"\n  LATEST VALUES:")
    latest = conn.execute("""
    SELECT r.symbol, r.value, r.unit, r.date
    FROM raw_data r
    INNER JOIN (
        SELECT symbol, MAX(date) as max_date FROM raw_data GROUP BY symbol
    ) l ON r.symbol = l.symbol AND r.date = l.max_date
    ORDER BY r.tier, r.symbol
    """).fetchall()

    for r in latest:
        unit = r['unit'] or ''
        print(f"    {r['symbol']:<15} {r['value']:>12.2f} {unit:<10} ({r['date']})")

    conn.close()
    print(f"{'='*70}\n")


if __name__ == "__main__":
    init_db()

    parser = argparse.ArgumentParser(description="CRNG-Fragility Monitor: Data Collector")
    parser.add_argument("--days", type=int, default=365, help="Days of history to fetch")
    parser.add_argument("--symbol", help="Fetch specific symbol only")
    parser.add_argument("--status", action="store_true", help="Show collection status")
    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.symbol:
        collect_all(days=args.days, symbols=[args.symbol.upper()])
    else:
        collect_all(days=args.days)
