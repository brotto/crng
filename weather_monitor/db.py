#!/usr/bin/env python3
"""
Weather Monitor — Database Layer
=================================
SQLite database for storing forecasts, observations, and CRNG-Cast predictions.
Designed for scientific rigor: every data point is timestamped and immutable.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "weather_monitor.db")


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

    # ── WF1: Forecast Tracking ──────────────────────────────────
    # Captures Open-Meteo forecasts and tracks if they change
    c.execute("""
    CREATE TABLE IF NOT EXISTS wf1_forecasts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        captured_at TEXT NOT NULL,           -- UTC timestamp of API call
        target_date TEXT NOT NULL,           -- Date being forecast (YYYY-MM-DD)
        target_hour INTEGER NOT NULL,        -- Hour being forecast (0-23)
        city TEXT NOT NULL,                  -- City key (sp, nyc, london)
        source TEXT NOT NULL DEFAULT 'open-meteo',
        temperature_2m REAL,
        relative_humidity_2m REAL,
        precipitation REAL,
        precipitation_probability REAL,
        cloud_cover REAL,
        wind_speed_10m REAL,
        wind_direction_10m REAL,
        wind_gusts_10m REAL,
        pressure_msl REAL,
        weather_code INTEGER,
        raw_json TEXT,                       -- Full API response for audit
        UNIQUE(captured_at, target_date, target_hour, city)
    )
    """)

    # ── WF2: Observation Tracking ───────────────────────────────
    # Captures actual observations (archive API, delayed 48h+)
    c.execute("""
    CREATE TABLE IF NOT EXISTS wf2_observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        captured_at TEXT NOT NULL,           -- When we fetched this data
        observed_date TEXT NOT NULL,         -- Date of observation
        observed_hour INTEGER NOT NULL,      -- Hour of observation
        city TEXT NOT NULL,
        source TEXT NOT NULL DEFAULT 'open-meteo-archive',
        temperature_2m REAL,
        relative_humidity_2m REAL,
        precipitation REAL,
        cloud_cover REAL,
        wind_speed_10m REAL,
        wind_direction_10m REAL,
        pressure_msl REAL,
        weather_code INTEGER,
        raw_json TEXT,
        UNIQUE(observed_date, observed_hour, city, source)
    )
    """)

    # ── WF3: CRNG-Cast Predictions ──────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS wf3_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        predicted_at TEXT NOT NULL,          -- When prediction was made
        target_date TEXT NOT NULL,
        target_hour INTEGER NOT NULL,
        city TEXT NOT NULL,
        model_version TEXT NOT NULL DEFAULT 'v1.2',
        temperature_pred REAL,
        temperature_ci_lo REAL,
        temperature_ci_hi REAL,
        cloud_factor REAL,
        diurnal_dt REAL,                    -- Diurnal component used
        stochastic_dt REAL,                 -- CRNG stochastic component
        base_temperature REAL,              -- T at prediction time
        base_hour INTEGER,                  -- Hour used as base
        historical_days_used INTEGER,        -- N days of history
        is_adjustment INTEGER DEFAULT 0,    -- 0=initial, 1=3h readjustment
        adjustment_reason TEXT,             -- Why readjusted (deviation, cloud change, etc)
        raw_params TEXT,                    -- JSON of all model params
        UNIQUE(predicted_at, target_date, target_hour, city)
    )
    """)

    # ── Comparison/Scorecard ────────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS scorecard (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        evaluated_at TEXT NOT NULL,
        target_date TEXT NOT NULL,
        target_hour INTEGER NOT NULL,
        city TEXT NOT NULL,
        observed_temp REAL,
        forecast_temp_midnight REAL,        -- Open-Meteo forecast (midnight capture)
        forecast_temp_latest REAL,          -- Open-Meteo latest forecast (possibly adjusted)
        crng_temp_midnight REAL,            -- CRNG-Cast initial prediction
        crng_temp_latest REAL,              -- CRNG-Cast latest (possibly readjusted)
        error_forecast_midnight REAL,
        error_forecast_latest REAL,
        error_crng_midnight REAL,
        error_crng_latest REAL,
        forecast_drifted INTEGER,           -- 1 if Open-Meteo changed prediction
        forecast_drift_amount REAL,
        crng_readjusted INTEGER,
        crng_readjust_amount REAL,
        within_crng_ci INTEGER,             -- 1 if observed within CI
        UNIQUE(target_date, target_hour, city)
    )
    """)

    # ── Error Statistics ────────────────────────────────────────
    c.execute("""
    CREATE TABLE IF NOT EXISTS error_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        computed_at TEXT NOT NULL,
        city TEXT NOT NULL,
        period_start TEXT NOT NULL,
        period_end TEXT NOT NULL,
        n_observations INTEGER,
        -- Open-Meteo stats
        om_mae REAL,                        -- Mean Absolute Error
        om_rmse REAL,
        om_bias REAL,                       -- Systematic bias
        om_error_kurtosis REAL,             -- Kurtosis of errors
        -- CRNG-Cast stats
        crng_mae REAL,
        crng_rmse REAL,
        crng_bias REAL,
        crng_error_kurtosis REAL,
        crng_ci_coverage REAL,              -- % within CI
        crng_ci_avg_width REAL,
        -- Comparison
        crng_wins INTEGER,
        om_wins INTEGER,
        draws INTEGER
    )
    """)

    conn.commit()
    conn.close()
    print(f"Database initialized: {DB_PATH}")


def insert_forecast(city, target_date, target_hour, data, captured_at=None):
    """Insert a forecast data point (WF1)."""
    if captured_at is None:
        captured_at = datetime.utcnow().isoformat() + "Z"

    conn = get_connection()
    try:
        conn.execute("""
        INSERT OR REPLACE INTO wf1_forecasts
        (captured_at, target_date, target_hour, city,
         temperature_2m, relative_humidity_2m, precipitation,
         precipitation_probability, cloud_cover, wind_speed_10m,
         wind_direction_10m, wind_gusts_10m, pressure_msl, weather_code, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            captured_at, target_date, target_hour, city,
            data.get('temperature_2m'), data.get('relative_humidity_2m'),
            data.get('precipitation'), data.get('precipitation_probability'),
            data.get('cloud_cover'), data.get('wind_speed_10m'),
            data.get('wind_direction_10m'), data.get('wind_gusts_10m'),
            data.get('pressure_msl'), data.get('weather_code'),
            str(data)
        ))
        conn.commit()
    finally:
        conn.close()


def insert_observation(city, observed_date, observed_hour, data, captured_at=None, source='open-meteo-archive'):
    """Insert an observation data point (WF2)."""
    if captured_at is None:
        captured_at = datetime.utcnow().isoformat() + "Z"

    conn = get_connection()
    try:
        conn.execute("""
        INSERT OR REPLACE INTO wf2_observations
        (captured_at, observed_date, observed_hour, city, source,
         temperature_2m, relative_humidity_2m, precipitation,
         cloud_cover, wind_speed_10m, wind_direction_10m,
         pressure_msl, weather_code, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            captured_at, observed_date, observed_hour, city, source,
            data.get('temperature_2m'), data.get('relative_humidity_2m'),
            data.get('precipitation'), data.get('cloud_cover'),
            data.get('wind_speed_10m'), data.get('wind_direction_10m'),
            data.get('pressure_msl'), data.get('weather_code'),
            str(data)
        ))
        conn.commit()
    finally:
        conn.close()


def insert_prediction(city, target_date, target_hour, pred_data, predicted_at=None):
    """Insert a CRNG-Cast prediction (WF3)."""
    if predicted_at is None:
        predicted_at = datetime.utcnow().isoformat() + "Z"

    conn = get_connection()
    try:
        conn.execute("""
        INSERT OR REPLACE INTO wf3_predictions
        (predicted_at, target_date, target_hour, city, model_version,
         temperature_pred, temperature_ci_lo, temperature_ci_hi,
         cloud_factor, diurnal_dt, stochastic_dt,
         base_temperature, base_hour, historical_days_used,
         is_adjustment, adjustment_reason, raw_params)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            predicted_at, target_date, target_hour, city,
            pred_data.get('model_version', 'v1.2'),
            pred_data.get('temperature_pred'),
            pred_data.get('temperature_ci_lo'),
            pred_data.get('temperature_ci_hi'),
            pred_data.get('cloud_factor'),
            pred_data.get('diurnal_dt'),
            pred_data.get('stochastic_dt'),
            pred_data.get('base_temperature'),
            pred_data.get('base_hour'),
            pred_data.get('historical_days_used', 3),
            pred_data.get('is_adjustment', 0),
            pred_data.get('adjustment_reason'),
            str(pred_data)
        ))
        conn.commit()
    finally:
        conn.close()


def get_forecasts_for_date(city, target_date):
    """Get all forecast captures for a given date (WF1)."""
    conn = get_connection()
    rows = conn.execute("""
    SELECT * FROM wf1_forecasts
    WHERE city = ? AND target_date = ?
    ORDER BY captured_at, target_hour
    """, (city, target_date)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_forecast_drift(city, target_date, target_hour):
    """Check if forecast changed for a specific hour (WF1)."""
    conn = get_connection()
    rows = conn.execute("""
    SELECT captured_at, temperature_2m, cloud_cover, precipitation_probability
    FROM wf1_forecasts
    WHERE city = ? AND target_date = ? AND target_hour = ?
    ORDER BY captured_at
    """, (city, target_date, target_hour)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_observations(city, date):
    """Get observations for a date (WF2)."""
    conn = get_connection()
    rows = conn.execute("""
    SELECT * FROM wf2_observations
    WHERE city = ? AND observed_date = ?
    ORDER BY observed_hour
    """, (city, date)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_predictions(city, target_date):
    """Get CRNG-Cast predictions for a date (WF3)."""
    conn = get_connection()
    rows = conn.execute("""
    SELECT * FROM wf3_predictions
    WHERE city = ? AND target_date = ?
    ORDER BY target_hour, predicted_at
    """, (city, target_date)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
