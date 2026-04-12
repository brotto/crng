#!/usr/bin/env python3
"""
WF1 — Forecast Capture & Drift Detection
==========================================
Captures Open-Meteo forecasts and detects if they change over time.

Schedule:
  - Midnight: Full day capture (24 hourly forecasts)
  - Every 3h:  Re-capture to detect drift

Usage:
  python wf1_capture_forecasts.py              # Capture current forecasts
  python wf1_capture_forecasts.py --drift      # Check for drift vs previous captures
  python wf1_capture_forecasts.py --report     # Print drift report
"""

import argparse
import json
import sys
import os
import urllib.request
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
from db import get_connection, insert_forecast, get_forecast_drift, init_db

# Cities to monitor
CITIES = {
    "sp":     {"lat": -23.55, "lon": -46.63, "tz": "America/Sao_Paulo", "name": "Sao Paulo"},
    "nyc":    {"lat":  40.71, "lon": -74.01, "tz": "America/New_York",  "name": "New York"},
    "london": {"lat":  51.51, "lon":  -0.13, "tz": "Europe/London",     "name": "London"},
}

FORECAST_VARS = [
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "precipitation_probability", "cloud_cover", "wind_speed_10m",
    "wind_direction_10m", "wind_gusts_10m", "pressure_msl", "weather_code"
]


def fetch_forecast(city_key):
    """Fetch 24h hourly forecast for a city."""
    city = CITIES[city_key]
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={city['lat']}&longitude={city['lon']}"
        f"&timezone={city['tz']}"
        f"&hourly={','.join(FORECAST_VARS)}"
        f"&forecast_days=1"
    )
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  ERROR fetching {city_key}: {e}")
        return None


def capture_forecasts():
    """Capture forecasts for all cities and store in DB."""
    captured_at = datetime.utcnow().isoformat() + "Z"
    print(f"\n[WF1] Capturing forecasts at {captured_at}")

    for city_key in CITIES:
        data = fetch_forecast(city_key)
        if data is None or "error" in data:
            print(f"  SKIP {city_key}: API error")
            continue

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        count = 0

        for i, t in enumerate(times):
            target_date = t.split("T")[0]
            target_hour = int(t.split("T")[1].split(":")[0])

            row = {}
            for var in FORECAST_VARS:
                vals = hourly.get(var, [])
                row[var] = vals[i] if i < len(vals) else None

            insert_forecast(city_key, target_date, target_hour, row, captured_at)
            count += 1

        print(f"  {CITIES[city_key]['name']}: {count} hourly forecasts stored")

    print(f"[WF1] Capture complete.\n")


def check_drift():
    """Check if forecasts have drifted since last capture."""
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\n[WF1] Drift analysis for {today}")
    print(f"{'='*65}")

    drift_found = False
    for city_key in CITIES:
        print(f"\n  {CITIES[city_key]['name']}:")
        for hour in range(24):
            captures = get_forecast_drift(city_key, today, hour)
            if len(captures) < 2:
                continue

            temps = [c['temperature_2m'] for c in captures if c['temperature_2m'] is not None]
            if len(temps) < 2:
                continue

            drift = max(temps) - min(temps)
            if drift > 0.1:  # Threshold: >0.1C change
                drift_found = True
                first = captures[0]
                last = captures[-1]
                print(f"    {hour:02d}:00 — DRIFT {drift:+.1f}°C "
                      f"({first['temperature_2m']:.1f} → {last['temperature_2m']:.1f}°C) "
                      f"over {len(captures)} captures")

    if not drift_found:
        print(f"\n  Nenhum drift significativo detectado (threshold: >0.1°C)")

    print(f"{'='*65}\n")


def print_report():
    """Print full forecast report for today."""
    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_connection()

    print(f"\n[WF1] Forecast Report — {today}")
    print(f"{'='*75}")

    for city_key in CITIES:
        rows = conn.execute("""
        SELECT target_hour,
               MIN(temperature_2m) as t_min,
               MAX(temperature_2m) as t_max,
               COUNT(DISTINCT captured_at) as n_captures,
               GROUP_CONCAT(DISTINCT SUBSTR(captured_at, 12, 5)) as capture_times
        FROM wf1_forecasts
        WHERE city = ? AND target_date = ?
        GROUP BY target_hour
        ORDER BY target_hour
        """, (city_key, today)).fetchall()

        print(f"\n  {CITIES[city_key]['name']}:")
        print(f"  {'Hora':>5} {'T_min':>7} {'T_max':>7} {'Drift':>7} {'Captures':>9}")
        print(f"  {'-'*40}")
        for r in rows:
            drift = (r['t_max'] - r['t_min']) if r['t_min'] and r['t_max'] else 0
            drift_s = f"{drift:.1f}°C" if drift > 0.1 else "—"
            print(f"  {r['target_hour']:>4}h {r['t_min']:>6.1f} {r['t_max']:>6.1f} {drift_s:>7} {r['n_captures']:>9}")

    conn.close()
    print(f"{'='*75}\n")


if __name__ == "__main__":
    init_db()

    parser = argparse.ArgumentParser(description="WF1: Forecast Capture & Drift Detection")
    parser.add_argument("--drift", action="store_true", help="Check for forecast drift")
    parser.add_argument("--report", action="store_true", help="Print forecast report")
    args = parser.parse_args()

    if args.drift:
        check_drift()
    elif args.report:
        print_report()
    else:
        capture_forecasts()
