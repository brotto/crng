#!/usr/bin/env python3
"""
CRNG-Fragility Monitor — Database Layer
=========================================
SQLite database for commodity prices, economic indicators, and CRNG metrics.
Designed for scientific rigor: every data point is timestamped and immutable.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "fragility_monitor.db")


def get_connection():
    """Get database connection with WAL mode for concurrent access."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize all tables."""
    conn = get_connection()
    c = conn.cursor()

    # ── Raw Price/Indicator Data ───────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS raw_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        captured_at TEXT NOT NULL,
        date TEXT NOT NULL,                 -- YYYY-MM-DD
        symbol TEXT NOT NULL,               -- e.g. BRENT, HH_GAS, VIX, UREA
        tier INTEGER NOT NULL,              -- 1=choke, 2=financial, 3=physical, 4=ai_bubble
        category TEXT NOT NULL,             -- energy, fertilizer, helium, financial, food, tech
        value REAL NOT NULL,
        source TEXT NOT NULL,               -- fred, yahoo, worldbank
        unit TEXT,                          -- USD/bbl, USD/MMBtu, index, etc.
        raw_json TEXT,
        UNIQUE(date, symbol, source)
    )
    """)

    # ── CRNG Metrics (computed daily) ──────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS crng_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        computed_at TEXT NOT NULL,
        date TEXT NOT NULL,
        symbol TEXT NOT NULL,
        -- Cycle base
        ma20 REAL,                          -- 20-day moving average
        std20 REAL,                         -- 20-day rolling std
        -- CRNG indicators
        z_score REAL,                       -- (value - ma20) / std20
        kurtosis_60 REAL,                   -- 60-day rolling kurtosis
        ci_lo REAL,                         -- Fat-tailed CI lower
        ci_hi REAL,                         -- Fat-tailed CI upper
        ci_mult REAL,                       -- CI multiplier (kurtosis-adjusted)
        -- Regime
        regime TEXT,                        -- normal, stressed, critical
        UNIQUE(date, symbol)
    )
    """)

    # ── Concurrence Events ─────────────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS concurrence (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        computed_at TEXT NOT NULL,
        date TEXT NOT NULL,
        tier1_stressed INTEGER,             -- Count of Tier 1 symbols with |Z| > 2
        tier1_symbols TEXT,                 -- Comma-separated list
        max_kurtosis REAL,                  -- Highest kurtosis among Tier 1
        max_kurtosis_symbol TEXT,
        alert_level TEXT,                   -- green, yellow, orange, red
        lockstep_ratio REAL,                -- Energy-GDP correlation (rolling)
        notes TEXT,
        UNIQUE(date)
    )
    """)

    # ── Lockstep Tracker ───────────────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS lockstep (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        computed_at TEXT NOT NULL,
        period_start TEXT NOT NULL,
        period_end TEXT NOT NULL,
        energy_change_pct REAL,
        gdp_change_pct REAL,
        correlation REAL,
        divergence REAL,                    -- |energy - gdp| change
        notes TEXT
    )
    """)

    conn.commit()
    conn.close()
    print(f"Fragility Monitor DB initialized: {DB_PATH}")


def insert_raw(date, symbol, tier, category, value, source, unit=None, captured_at=None):
    """Insert a raw data point."""
    if captured_at is None:
        captured_at = datetime.utcnow().isoformat() + "Z"

    conn = get_connection()
    try:
        conn.execute("""
        INSERT OR REPLACE INTO raw_data
        (captured_at, date, symbol, tier, category, value, source, unit)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (captured_at, date, symbol, tier, category, value, source, unit))
        conn.commit()
    finally:
        conn.close()


def insert_metrics(date, symbol, metrics, computed_at=None):
    """Insert computed CRNG metrics."""
    if computed_at is None:
        computed_at = datetime.utcnow().isoformat() + "Z"

    conn = get_connection()
    try:
        conn.execute("""
        INSERT OR REPLACE INTO crng_metrics
        (computed_at, date, symbol, ma20, std20, z_score,
         kurtosis_60, ci_lo, ci_hi, ci_mult, regime)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            computed_at, date, symbol,
            metrics.get('ma20'), metrics.get('std20'), metrics.get('z_score'),
            metrics.get('kurtosis_60'), metrics.get('ci_lo'), metrics.get('ci_hi'),
            metrics.get('ci_mult'), metrics.get('regime')
        ))
        conn.commit()
    finally:
        conn.close()


def insert_concurrence(date, data, computed_at=None):
    """Insert concurrence event."""
    if computed_at is None:
        computed_at = datetime.utcnow().isoformat() + "Z"

    conn = get_connection()
    try:
        conn.execute("""
        INSERT OR REPLACE INTO concurrence
        (computed_at, date, tier1_stressed, tier1_symbols,
         max_kurtosis, max_kurtosis_symbol, alert_level,
         lockstep_ratio, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            computed_at, date,
            data.get('tier1_stressed'), data.get('tier1_symbols'),
            data.get('max_kurtosis'), data.get('max_kurtosis_symbol'),
            data.get('alert_level'), data.get('lockstep_ratio'),
            data.get('notes')
        ))
        conn.commit()
    finally:
        conn.close()


def get_raw_series(symbol, start_date=None, end_date=None):
    """Get raw data series for a symbol."""
    conn = get_connection()
    query = "SELECT date, value FROM raw_data WHERE symbol = ?"
    params = [symbol]
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    query += " ORDER BY date"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [(r['date'], r['value']) for r in rows]


def get_latest_metrics():
    """Get most recent metrics for all symbols."""
    conn = get_connection()
    rows = conn.execute("""
    SELECT m.* FROM crng_metrics m
    INNER JOIN (
        SELECT symbol, MAX(date) as max_date
        FROM crng_metrics GROUP BY symbol
    ) latest ON m.symbol = latest.symbol AND m.date = latest.max_date
    ORDER BY m.symbol
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_latest_concurrence():
    """Get most recent concurrence assessment."""
    conn = get_connection()
    row = conn.execute("""
    SELECT * FROM concurrence ORDER BY date DESC LIMIT 1
    """).fetchone()
    conn.close()
    return dict(row) if row else None


if __name__ == "__main__":
    init_db()
