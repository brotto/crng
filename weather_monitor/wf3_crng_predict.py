#!/usr/bin/env python3
"""
WF3 — CRNG-Cast Predictions
=============================
Generates CRNG-Cast predictions and readjusts every 3h.

Schedule:
  - Midnight: Generate predictions for all 24 hours
  - Every 3h:  Check current conditions, readjust if needed

Readjustment triggers:
  - Temperature deviation > 2.0°C from prediction
  - Cloud cover change > 30 percentage points
  - Precipitation event not predicted (or vice-versa)

Usage:
  python wf3_crng_predict.py                   # Generate predictions for today
  python wf3_crng_predict.py --adjust          # 3h readjustment check
  python wf3_crng_predict.py --report          # Print prediction report
  python wf3_crng_predict.py --score 2026-04-06  # Score predictions vs observations
"""

import argparse
import json
import math
import sys
import os
import urllib.request
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
from db import (get_connection, insert_prediction, get_predictions,
                get_observations, init_db)

CITIES = {
    "sp":     {"lat": -23.55, "lon": -46.63, "tz": "America/Sao_Paulo", "name": "Sao Paulo",
               "kurtosis": 6.39, "target_k": 15.0},
    "nyc":    {"lat":  40.71, "lon": -74.01, "tz": "America/New_York",  "name": "New York",
               "kurtosis": 4.71, "target_k": 8.0},
    "london": {"lat":  51.51, "lon":  -0.13, "tz": "Europe/London",     "name": "London",
               "kurtosis": 4.98, "target_k": 10.0},
}

ARCHIVE_VARS = [
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "cloud_cover", "wind_speed_10m", "pressure_msl", "weather_code"
]

READJUST_TEMP_THRESHOLD = 2.0      # °C deviation triggers readjustment
READJUST_CLOUD_THRESHOLD = 30.0    # pp cloud change triggers readjustment
HISTORICAL_DAYS = 5                 # Days of history for diurnal model


def fetch_archive(city_key, start_date, end_date):
    """Fetch archive data for diurnal model building."""
    city = CITIES[city_key]
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={city['lat']}&longitude={city['lon']}"
        f"&timezone={city['tz']}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly={','.join(ARCHIVE_VARS)}"
    )
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  ERROR fetching archive: {e}")
        return None


def fetch_current(city_key):
    """Fetch current conditions."""
    city = CITIES[city_key]
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={city['lat']}&longitude={city['lon']}"
        f"&timezone={city['tz']}"
        f"&current=temperature_2m,relative_humidity_2m,cloud_cover,"
        f"wind_speed_10m,precipitation,weather_code"
    )
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  ERROR fetching current: {e}")
        return None


def build_diurnal_model(city_key, n_days=HISTORICAL_DAYS):
    """Build diurnal temperature model from historical data.

    Returns dict of {hour: {'mean_temp': float, 'std': float, 'mean_dt': float}}
    """
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=n_days + 1)).strftime("%Y-%m-%d")

    data = fetch_archive(city_key, start_date, end_date)
    if data is None or "error" in data:
        return None

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    clouds = hourly.get("cloud_cover", [])

    # Group by hour across days
    hour_data = {h: [] for h in range(24)}
    dt_data = {h: [] for h in range(24)}
    cloud_data = {h: [] for h in range(24)}

    for i, t in enumerate(times):
        hour = int(t.split("T")[1].split(":")[0])
        if temps[i] is not None:
            hour_data[hour].append(temps[i])
            if clouds[i] is not None:
                cloud_data[hour].append(clouds[i])

    # Calculate hour-to-hour deltas for each day
    day_temps = {}
    for i, t in enumerate(times):
        date = t.split("T")[0]
        hour = int(t.split("T")[1].split(":")[0])
        if date not in day_temps:
            day_temps[date] = {}
        if temps[i] is not None:
            day_temps[date][hour] = temps[i]

    for date, ht in day_temps.items():
        for h in range(1, 24):
            if h in ht and (h-1) in ht:
                dt_data[h].append(ht[h] - ht[h-1])

    model = {}
    for h in range(24):
        model[h] = {
            'mean_temp': sum(hour_data[h]) / len(hour_data[h]) if hour_data[h] else None,
            'std': _std(hour_data[h]) if len(hour_data[h]) > 1 else 1.0,
            'mean_dt': sum(dt_data[h]) / len(dt_data[h]) if dt_data[h] else 0,
            'std_dt': _std(dt_data[h]) if len(dt_data[h]) > 1 else 1.0,
            'kurtosis_dt': _kurtosis(dt_data[h]) if len(dt_data[h]) > 3 else 3.0,
            'mean_cloud': sum(cloud_data[h]) / len(cloud_data[h]) if cloud_data[h] else 50,
            'n_samples': len(hour_data[h]),
        }

    return model


def _std(values):
    if len(values) < 2:
        return 0
    mean = sum(values) / len(values)
    return math.sqrt(sum((x - mean)**2 for x in values) / (len(values) - 1))


def _kurtosis(values):
    if len(values) < 4:
        return 3.0
    n = len(values)
    mean = sum(values) / n
    m2 = sum((x - mean)**2 for x in values) / n
    m4 = sum((x - mean)**4 for x in values) / n
    return m4 / (m2**2) if m2 > 0 else 3.0


def predict_full_day(city_key, model, current_temp=None, current_hour=None, current_cloud=None):
    """Generate predictions for all 24 hours using diurnal model."""
    city = CITIES[city_key]
    predictions = {}

    if current_temp is None:
        # Use model's midnight temperature as base
        current_temp = model[0]['mean_temp'] if model[0]['mean_temp'] else 20.0
        current_hour = 0

    for target_h in range(24):
        if target_h <= current_hour:
            continue  # Don't predict the past

        # Accumulate diurnal deltas from current_hour to target
        diurnal_dt = 0
        for h in range(current_hour + 1, target_h + 1):
            diurnal_dt += model[h]['mean_dt']

        # Cloud adjustment
        cloud_expected = model[target_h]['mean_cloud']
        if current_cloud is not None:
            cloud_diff = current_cloud - cloud_expected
            # More clouds than expected: reduce warming (if warming phase) or reduce cooling
            is_warming = model[target_h]['mean_dt'] > 0
            if is_warming and cloud_diff > 20:
                cloud_factor = max(0.6, 1.0 - (cloud_diff / 200))
            elif not is_warming and cloud_diff > 20:
                cloud_factor = min(1.2, 1.0 + (cloud_diff / 300))
            else:
                cloud_factor = 1.0
        else:
            cloud_factor = 1.0

        diurnal_dt_adjusted = diurnal_dt * cloud_factor

        # Temperature prediction
        t_pred = current_temp + diurnal_dt_adjusted

        # CI based on historical variability + fat tails
        hours_ahead = target_h - current_hour
        sigma_accum = 0
        for h in range(current_hour + 1, target_h + 1):
            sigma_accum += model[h]['std_dt'] ** 2
        sigma_accum = math.sqrt(sigma_accum) if sigma_accum > 0 else 1.0

        # Fat-tail multiplier: wider CI for higher kurtosis
        k = city.get('kurtosis', 3.0)
        ci_mult = 1.645 * (1 + (k - 3) / 10)  # Wider for fatter tails
        ci_lo = t_pred - ci_mult * sigma_accum
        ci_hi = t_pred + ci_mult * sigma_accum

        predictions[target_h] = {
            'temperature_pred': round(t_pred, 1),
            'temperature_ci_lo': round(ci_lo, 1),
            'temperature_ci_hi': round(ci_hi, 1),
            'cloud_factor': round(cloud_factor, 3),
            'diurnal_dt': round(diurnal_dt_adjusted, 2),
            'stochastic_dt': 0,  # v1.2 uses purely diurnal for now
            'base_temperature': current_temp,
            'base_hour': current_hour,
            'historical_days_used': HISTORICAL_DAYS,
            'model_version': 'v1.2',
            'is_adjustment': 0,
            'adjustment_reason': None,
        }

    return predictions


def generate_predictions():
    """Generate full-day predictions for all cities."""
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\n[WF3] Generating CRNG-Cast predictions for {today}")

    for city_key in CITIES:
        city = CITIES[city_key]
        print(f"\n  Building diurnal model for {city['name']}...")
        model = build_diurnal_model(city_key)

        if model is None:
            print(f"    SKIP: couldn't build model")
            continue

        # Get current conditions
        current = fetch_current(city_key)
        current_temp = None
        current_hour = None
        current_cloud = None

        if current and "current" in current:
            c = current["current"]
            current_temp = c.get("temperature_2m")
            current_cloud = c.get("cloud_cover")
            time_str = c.get("time", "")
            if "T" in time_str:
                current_hour = int(time_str.split("T")[1].split(":")[0])

        print(f"    Current: {current_temp}°C at {current_hour}:00, clouds={current_cloud}%")

        predictions = predict_full_day(
            city_key, model,
            current_temp=current_temp,
            current_hour=current_hour,
            current_cloud=current_cloud
        )

        # Store predictions
        for target_h, pred in predictions.items():
            insert_prediction(city_key, today, target_h, pred)

        # Print summary
        print(f"    Predictions ({len(predictions)} hours):")
        for h in sorted(predictions.keys()):
            p = predictions[h]
            print(f"      {h:02d}:00 → {p['temperature_pred']}°C "
                  f"[{p['temperature_ci_lo']} — {p['temperature_ci_hi']}] "
                  f"cloud_f={p['cloud_factor']}")

    print(f"\n[WF3] Predictions complete.\n")


def readjust():
    """3h readjustment: check current conditions and re-predict if needed."""
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\n[WF3] Readjustment check for {today}")

    conn = get_connection()

    for city_key in CITIES:
        city = CITIES[city_key]

        # Get current conditions
        current = fetch_current(city_key)
        if not current or "current" not in current:
            print(f"  SKIP {city['name']}: no current data")
            continue

        c = current["current"]
        current_temp = c.get("temperature_2m")
        current_cloud = c.get("cloud_cover")
        time_str = c.get("time", "")
        current_hour = int(time_str.split("T")[1].split(":")[0]) if "T" in time_str else None

        if current_hour is None or current_temp is None:
            continue

        # Get our prediction for this hour
        pred_row = conn.execute("""
        SELECT temperature_pred, temperature_ci_lo, temperature_ci_hi
        FROM wf3_predictions
        WHERE city = ? AND target_date = ? AND target_hour = ?
        ORDER BY predicted_at DESC LIMIT 1
        """, (city_key, today, current_hour)).fetchone()

        reasons = []
        if pred_row:
            pred_temp = pred_row['temperature_pred']
            temp_dev = abs(current_temp - pred_temp)

            if temp_dev > READJUST_TEMP_THRESHOLD:
                reasons.append(f"temp_deviation={temp_dev:.1f}C")

        # Check cloud change
        # (compare current cloud vs what we expected)
        model = build_diurnal_model(city_key)
        if model and current_cloud is not None:
            expected_cloud = model[current_hour]['mean_cloud']
            cloud_diff = abs(current_cloud - expected_cloud)
            if cloud_diff > READJUST_CLOUD_THRESHOLD:
                reasons.append(f"cloud_change={cloud_diff:.0f}pp")

        if reasons:
            reason_str = "; ".join(reasons)
            print(f"\n  {city['name']}: READJUSTING ({reason_str})")
            print(f"    Current: {current_temp}°C, clouds={current_cloud}%")

            if model:
                predictions = predict_full_day(
                    city_key, model,
                    current_temp=current_temp,
                    current_hour=current_hour,
                    current_cloud=current_cloud
                )

                for target_h, pred in predictions.items():
                    pred['is_adjustment'] = 1
                    pred['adjustment_reason'] = reason_str
                    insert_prediction(city_key, today, target_h, pred)

                for h in sorted(predictions.keys()):
                    p = predictions[h]
                    print(f"      {h:02d}:00 → {p['temperature_pred']}°C "
                          f"[{p['temperature_ci_lo']} — {p['temperature_ci_hi']}]")
        else:
            print(f"  {city['name']}: No readjustment needed "
                  f"(T={current_temp}°C, clouds={current_cloud}%)")

    conn.close()
    print(f"\n[WF3] Readjustment complete.\n")


def score_predictions(target_date):
    """Score predictions against observations."""
    conn = get_connection()
    print(f"\n[WF3] CRNG-Cast Scorecard — {target_date}")
    print(f"{'='*80}")

    for city_key in CITIES:
        city = CITIES[city_key]

        # Get initial (midnight) predictions
        preds = conn.execute("""
        SELECT target_hour, temperature_pred, temperature_ci_lo, temperature_ci_hi,
               MIN(predicted_at) as first_pred
        FROM wf3_predictions
        WHERE city = ? AND target_date = ? AND is_adjustment = 0
        GROUP BY target_hour
        ORDER BY target_hour
        """, (city_key, target_date)).fetchall()

        # Get latest (adjusted) predictions
        adj_preds = conn.execute("""
        SELECT target_hour, temperature_pred, temperature_ci_lo, temperature_ci_hi,
               MAX(predicted_at) as last_pred
        FROM wf3_predictions
        WHERE city = ? AND target_date = ?
        GROUP BY target_hour
        ORDER BY target_hour
        """, (city_key, target_date)).fetchall()

        # Get observations
        obs = conn.execute("""
        SELECT observed_hour, temperature_2m
        FROM wf2_observations
        WHERE city = ? AND observed_date = ?
        ORDER BY observed_hour
        """, (city_key, target_date)).fetchall()

        # Get midnight forecasts (Open-Meteo)
        forecasts = conn.execute("""
        SELECT target_hour, temperature_2m,
               MIN(captured_at) as first_capture
        FROM wf1_forecasts
        WHERE city = ? AND target_date = ?
        GROUP BY target_hour
        ORDER BY target_hour
        """, (city_key, target_date)).fetchall()

        if not preds or not obs:
            print(f"\n  {city['name']}: Dados insuficientes")
            continue

        obs_map = {r['observed_hour']: r['temperature_2m'] for r in obs}
        adj_map = {r['target_hour']: r for r in adj_preds}
        fc_map = {r['target_hour']: r['temperature_2m'] for r in forecasts}

        print(f"\n  {city['name']} (K={city['kurtosis']}):")
        print(f"  {'Hora':>5} {'CRNG0':>7} {'CRNGadj':>8} {'OM':>7} {'Obs':>7} "
              f"{'ErrC0':>6} {'ErrCA':>6} {'ErrOM':>6} {'CI':>5}")
        print(f"  {'-'*62}")

        err_crng0 = []
        err_crng_adj = []
        err_om = []
        ci_hits = 0
        ci_total = 0

        for r in preds:
            h = r['target_hour']
            t_obs = obs_map.get(h)
            if t_obs is None:
                continue

            t_c0 = r['temperature_pred']
            t_ca = adj_map[h]['temperature_pred'] if h in adj_map else t_c0
            t_om = fc_map.get(h)
            ci_lo = adj_map[h]['temperature_ci_lo'] if h in adj_map else r['temperature_ci_lo']
            ci_hi = adj_map[h]['temperature_ci_hi'] if h in adj_map else r['temperature_ci_hi']

            ec0 = abs(t_c0 - t_obs)
            eca = abs(t_ca - t_obs)
            eo = abs(t_om - t_obs) if t_om else None
            in_ci = ci_lo <= t_obs <= ci_hi

            err_crng0.append(ec0)
            err_crng_adj.append(eca)
            if eo is not None:
                err_om.append(eo)
            ci_total += 1
            if in_ci:
                ci_hits += 1

            ci_s = "Y" if in_ci else "N"
            eo_s = f"{eo:.1f}" if eo else "—"

            print(f"  {h:>4}h {t_c0:>6.1f} {t_ca:>7.1f} "
                  f"{t_om if t_om else 0:>6.1f} {t_obs:>6.1f} "
                  f"{ec0:>5.1f} {eca:>5.1f} {eo_s:>6} {ci_s:>5}")

        if err_crng0:
            mae_c0 = sum(err_crng0) / len(err_crng0)
            mae_ca = sum(err_crng_adj) / len(err_crng_adj)
            mae_om = sum(err_om) / len(err_om) if err_om else 0
            ci_pct = (ci_hits / ci_total * 100) if ci_total else 0
            k_err = _kurtosis(err_crng_adj) if len(err_crng_adj) > 3 else 0

            print(f"  {'-'*62}")
            print(f"  MAE:  CRNG(0)={mae_c0:.2f}  CRNG(adj)={mae_ca:.2f}  OM={mae_om:.2f}")
            print(f"  CI coverage: {ci_pct:.0f}% ({ci_hits}/{ci_total})")
            print(f"  Error kurtosis: {k_err:.2f}")

            winner = "CRNG" if mae_ca < mae_om else "OM"
            print(f"  Winner: {winner}")

    conn.close()
    print(f"{'='*80}\n")


if __name__ == "__main__":
    init_db()

    parser = argparse.ArgumentParser(description="WF3: CRNG-Cast Predictions")
    parser.add_argument("--adjust", action="store_true", help="3h readjustment check")
    parser.add_argument("--report", action="store_true", help="Print predictions report")
    parser.add_argument("--score", metavar="DATE", help="Score predictions vs observations")
    args = parser.parse_args()

    if args.adjust:
        readjust()
    elif args.score:
        score_predictions(args.score)
    else:
        generate_predictions()
