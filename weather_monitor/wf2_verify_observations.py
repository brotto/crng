#!/usr/bin/env python3
"""
WF2 — Observation Verification
================================
Fetches actual weather observations and compares against
the ORIGINAL (midnight) forecasts, before any drift/adjustment.

Data Sources (priority order):
  1. INMET (Brazilian cities only) — real station data
  2. Open-Meteo Archive API — global fallback

Schedule:
  - Every 3h: Fetch archive for hours that have passed
  - Daily (next day): Full day verification with 24h+ delay

Usage:
  python wf2_verify_observations.py                       # Fetch today's observations
  python wf2_verify_observations.py --date 2026-04-06     # Fetch specific date
  python wf2_verify_observations.py --compare 2026-04-06  # Compare forecast vs observed
  python wf2_verify_observations.py --delay 48            # Use 48h delay for reliable data
"""

import argparse
import json
import sys
import os
import urllib.request
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
from db import (get_connection, insert_observation, get_observations,
                get_forecasts_for_date, init_db)

# ── City Definitions ───────────────────────────────────────────────
CITIES = {
    "sp":     {"lat": -23.55, "lon": -46.63, "tz": "America/Sao_Paulo", "name": "Sao Paulo"},
    "nyc":    {"lat":  40.71, "lon": -74.01, "tz": "America/New_York",  "name": "New York"},
    "london": {"lat":  51.51, "lon":  -0.13, "tz": "Europe/London",     "name": "London"},
}

# ── INMET Station Mapping ─────────────────────────────────────────
# Only Brazilian cities have INMET stations
# A701 = SP Mirante de Santana (principal), A771 = SP Interlagos (backup)
INMET_STATIONS = {
    "sp": [
        {"code": "A701", "name": "SP Mirante de Santana", "lat": -23.50, "lon": -46.62},
        {"code": "A771", "name": "SP Interlagos",         "lat": -23.72, "lon": -46.68},
    ],
    # NYC and London don't have INMET stations (Brazil-only network)
}

INMET_API_BASE = "https://apitempo.inmet.gov.br"
INMET_HEADERS = {
    "User-Agent": "CRNG-Cast-WeatherMonitor/1.0 (research; contact@crng-cast.org)",
    "Accept": "application/json",
}

# ── Open-Meteo Archive Variables ──────────────────────────────────
ARCHIVE_VARS = [
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "cloud_cover", "wind_speed_10m", "wind_direction_10m",
    "pressure_msl", "weather_code"
]


# ══════════════════════════════════════════════════════════════════
# INMET Data Source
# ══════════════════════════════════════════════════════════════════

def fetch_inmet_station(station_code, date):
    """
    Fetch hourly observation data from INMET for a specific station and date.

    INMET API endpoint: /estacao/{date}/{date}/{station_code}
    Returns list of hourly observations or None on failure.

    Known issues:
    - API sometimes returns HTTP 204 (No Content) — treat as temporary failure
    - Requires User-Agent header (connection reset without it)
    - Data availability: typically 1-2h delay from real-time
    """
    # INMET date format: YYYY-MM-DD
    url = f"{INMET_API_BASE}/estacao/{date}/{date}/{station_code}"

    try:
        req = urllib.request.Request(url, headers=INMET_HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = resp.getcode()
            if status == 204:
                return None  # No data available

            raw = resp.read().decode()
            if not raw or raw.strip() == "":
                return None

            data = json.loads(raw)
            if isinstance(data, list) and len(data) > 0:
                return data
            return None

    except urllib.error.HTTPError as e:
        if e.code == 204:
            return None
        print(f"    INMET HTTP {e.code} for station {station_code}: {e.reason}")
        return None
    except Exception as e:
        print(f"    INMET error for station {station_code}: {e}")
        return None


def parse_inmet_observation(record):
    """
    Parse a single INMET observation record into our standard format.

    INMET field mapping:
      TEM_INS -> temperature_2m (instantaneous temperature, °C)
      UMD_INS -> relative_humidity_2m (instantaneous humidity, %)
      CHUVA   -> precipitation (mm)
      PRE_INS -> pressure_msl (hPa) — actually station pressure, close enough
      VEN_VEL -> wind_speed_10m (m/s → km/h conversion needed)
      VEN_DIR -> wind_direction_10m (degrees)
      HR_MEDICAO -> observation time (HHmm format)
      DT_MEDICAO -> observation date (YYYY-MM-DD)
    """
    try:
        row = {}

        # Temperature (°C) — INMET uses TEM_INS (instantaneous)
        temp = record.get("TEM_INS")
        if temp is not None and temp != "":
            row["temperature_2m"] = float(temp)
        else:
            # Try TEM_MAX or TEM_MIN as fallback
            temp = record.get("TEM_MAX") or record.get("TEM_MIN")
            if temp is not None and temp != "":
                row["temperature_2m"] = float(temp)
            else:
                row["temperature_2m"] = None

        # Humidity (%)
        hum = record.get("UMD_INS")
        if hum is not None and hum != "":
            row["relative_humidity_2m"] = float(hum)
        else:
            row["relative_humidity_2m"] = None

        # Precipitation (mm)
        prec = record.get("CHUVA")
        if prec is not None and prec != "":
            row["precipitation"] = float(prec)
        else:
            row["precipitation"] = None

        # Cloud cover — INMET doesn't provide this directly
        row["cloud_cover"] = None

        # Wind speed (INMET: m/s → convert to km/h for consistency with Open-Meteo)
        wind = record.get("VEN_VEL")
        if wind is not None and wind != "":
            row["wind_speed_10m"] = float(wind) * 3.6  # m/s → km/h
        else:
            row["wind_speed_10m"] = None

        # Wind direction (degrees)
        wdir = record.get("VEN_DIR")
        if wdir is not None and wdir != "":
            row["wind_direction_10m"] = float(wdir)
        else:
            row["wind_direction_10m"] = None

        # Pressure (hPa) — INMET gives station pressure, not MSL
        pres = record.get("PRE_INS")
        if pres is not None and pres != "":
            row["pressure_msl"] = float(pres)
        else:
            row["pressure_msl"] = None

        # Weather code — INMET doesn't use WMO codes
        row["weather_code"] = None

        # Extract hour from HR_MEDICAO (format: "HHmm" or "HHMM")
        hr = record.get("HR_MEDICAO", "")
        if hr and len(hr) >= 2:
            hour = int(hr[:2])
            # INMET reports in UTC — convert to local time handled by caller
        else:
            hour = None

        # Extract date
        date = record.get("DT_MEDICAO", "")

        return row, date, hour

    except (ValueError, TypeError) as e:
        print(f"    INMET parse error: {e}")
        return None, None, None


def fetch_inmet_observations(city_key, date):
    """
    Try to fetch observations from INMET for a Brazilian city.

    Returns list of (obs_hour, row_dict) tuples, or None if INMET unavailable.
    Tries primary station first, then backup stations.
    """
    stations = INMET_STATIONS.get(city_key)
    if not stations:
        return None  # No INMET stations for this city

    for station in stations:
        print(f"    Trying INMET station {station['code']} ({station['name']})...")
        data = fetch_inmet_station(station["code"], date)

        if data is None:
            print(f"    INMET {station['code']}: no data available")
            continue

        observations = []
        for record in data:
            row, obs_date, obs_hour = parse_inmet_observation(record)
            if row is None or row.get("temperature_2m") is None or obs_hour is None:
                continue

            # INMET reports in UTC — adjust to local timezone
            # SP is UTC-3, so INMET hour 15:00 UTC = 12:00 local
            # However, INMET's HR_MEDICAO appears to already be in local time
            # (based on API documentation). We store as-is.
            observations.append((obs_hour, row))

        if observations:
            print(f"    INMET {station['code']}: {len(observations)} hourly observations found")
            return observations

        print(f"    INMET {station['code']}: data returned but no valid temperature readings")

    return None  # All stations failed


# ══════════════════════════════════════════════════════════════════
# Open-Meteo Archive Data Source (fallback)
# ══════════════════════════════════════════════════════════════════

def fetch_archive(city_key, date):
    """Fetch archive (observation) data from Open-Meteo for a specific date."""
    city = CITIES[city_key]
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={city['lat']}&longitude={city['lon']}"
        f"&timezone={city['tz']}"
        f"&start_date={date}&end_date={date}"
        f"&hourly={','.join(ARCHIVE_VARS)}"
    )
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"    ERROR fetching Open-Meteo archive {city_key}: {e}")
        return None


def parse_open_meteo_observations(data, city_key, max_hour=None):
    """
    Parse Open-Meteo archive response into list of (obs_hour, row_dict).
    Returns list or None on error.

    IMPORTANT: Open-Meteo archive API returns model projections for future
    hours of the current day, NOT real station observations. The max_hour
    parameter filters out any hour >= max_hour to prevent storing fake data.
    """
    if data is None or "error" in data:
        return None

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    observations = []

    for i, t in enumerate(times):
        obs_date = t.split("T")[0]
        obs_hour = int(t.split("T")[1].split(":")[0])

        # Filter out future hours (archive API returns model projections, not observations)
        if max_hour is not None and obs_hour >= max_hour:
            continue

        row = {}
        for var in ARCHIVE_VARS:
            vals = hourly.get(var, [])
            row[var] = vals[i] if i < len(vals) else None

        # Skip if temperature is None (no data yet)
        if row.get('temperature_2m') is None:
            continue

        observations.append((obs_hour, row))

    return observations if observations else None


# ══════════════════════════════════════════════════════════════════
# Unified Capture (INMET → Open-Meteo fallback)
# ══════════════════════════════════════════════════════════════════

def capture_observations(target_date=None, delay_hours=24):
    """
    Capture observations using best available source.

    Priority:
      1. INMET (Brazilian cities only) — real station data, gold standard
      2. Open-Meteo Archive — global, but mixes model + station data

    CRITICAL: Open-Meteo archive API returns model projections for future
    hours of the current day. We calculate max_hour per city based on current
    UTC time to filter out fake "observations" that are actually forecasts.
    """
    if target_date is None:
        target_dt = datetime.now() - timedelta(hours=delay_hours)
        target_date = target_dt.strftime("%Y-%m-%d")

    captured_at = datetime.utcnow().isoformat() + "Z"
    today = datetime.utcnow().strftime("%Y-%m-%d")
    print(f"\n[WF2] Capturing observations for {target_date} (captured at {captured_at})")

    # Timezone offsets from UTC (negative = behind UTC)
    TZ_OFFSETS = {"sp": -3, "nyc": -4, "london": 1}  # Approximate, DST-aware would be better

    for city_key in CITIES:
        source = None
        observations = None

        # ── Calculate max_hour for today (prevent storing future projections) ──
        max_hour = None
        if target_date == today or target_date == datetime.now().strftime("%Y-%m-%d"):
            # Current local hour for this city
            utc_hour = datetime.utcnow().hour
            offset = TZ_OFFSETS.get(city_key, 0)
            local_hour = (utc_hour + offset) % 24
            # Only trust hours at least 1h in the past (data delay)
            max_hour = max(0, local_hour - 1)
            print(f"  {CITIES[city_key]['name']}: filtering to hours 0-{max_hour} (current local ~{local_hour}h)")

        # ── Try INMET first (Brazilian cities only) ──
        if city_key in INMET_STATIONS:
            print(f"  {CITIES[city_key]['name']}: trying INMET...")
            observations = fetch_inmet_observations(city_key, target_date)
            if observations:
                source = "inmet"
                # INMET returns real station data, but still filter future if needed
                if max_hour is not None:
                    observations = [(h, row) for h, row in observations if h < max_hour]
                print(f"  {CITIES[city_key]['name']}: using INMET data ({len(observations)} hours)")

        # ── Fallback to Open-Meteo Archive ──
        if observations is None:
            if city_key in INMET_STATIONS:
                print(f"  {CITIES[city_key]['name']}: INMET unavailable, falling back to Open-Meteo archive")
            else:
                print(f"  {CITIES[city_key]['name']}: fetching Open-Meteo archive...")

            data = fetch_archive(city_key, target_date)
            observations = parse_open_meteo_observations(data, city_key, max_hour=max_hour)
            if observations:
                source = "open-meteo-archive"

        # ── Store observations ──
        if observations is None or len(observations) == 0:
            print(f"  {CITIES[city_key]['name']}: NO DATA from any source!")
            continue

        count = 0
        for obs_hour, row in observations:
            insert_observation(city_key, target_date, obs_hour, row,
                             captured_at=captured_at, source=source)
            count += 1

        source_label = "INMET" if source == "inmet" else "Open-Meteo Archive"
        print(f"  {CITIES[city_key]['name']}: {count} hourly observations stored (source: {source_label})")

    print(f"[WF2] Capture complete.\n")


# ══════════════════════════════════════════════════════════════════
# Comparison: Forecast vs Observed
# ══════════════════════════════════════════════════════════════════

def compare_forecast_vs_observed(target_date):
    """Compare midnight forecast against actual observations."""
    conn = get_connection()
    print(f"\n[WF2] Forecast vs Observation — {target_date}")
    print(f"{'='*80}")

    for city_key in CITIES:
        # Get earliest (midnight) forecast for each hour
        midnight_forecasts = conn.execute("""
        SELECT target_hour, temperature_2m, cloud_cover, precipitation_probability,
               MIN(captured_at) as first_capture
        FROM wf1_forecasts
        WHERE city = ? AND target_date = ?
        GROUP BY target_hour
        ORDER BY target_hour
        """, (city_key, target_date)).fetchall()

        # Get latest forecast for each hour (to detect drift)
        latest_forecasts = conn.execute("""
        SELECT target_hour, temperature_2m, cloud_cover,
               MAX(captured_at) as last_capture
        FROM wf1_forecasts
        WHERE city = ? AND target_date = ?
        GROUP BY target_hour
        ORDER BY target_hour
        """, (city_key, target_date)).fetchall()

        # Get observations (prefer INMET, but take whatever is available)
        observations = conn.execute("""
        SELECT observed_hour, temperature_2m, cloud_cover, precipitation, source,
               CASE
                   WHEN source = 'inmet' THEN 1
                   ELSE 2
               END as source_priority
        FROM wf2_observations
        WHERE city = ? AND observed_date = ?
        ORDER BY observed_hour, source_priority
        """, (city_key, target_date)).fetchall()

        if not midnight_forecasts or not observations:
            print(f"\n  {CITIES[city_key]['name']}: Dados insuficientes")
            continue

        # Build obs map — prefer INMET over Open-Meteo for same hour
        obs_map = {}
        for r in observations:
            h = r['observed_hour']
            if h not in obs_map or r['source'] == 'inmet':
                obs_map[h] = dict(r)

        latest_map = {r['target_hour']: dict(r) for r in latest_forecasts}

        # Identify sources used
        sources_used = set(obs_map[h]['source'] for h in obs_map)
        source_str = ", ".join(s.upper() for s in sorted(sources_used))

        print(f"\n  {CITIES[city_key]['name']} (source: {source_str}):")
        print(f"  {'Hora':>5} {'F(00h)':>7} {'F(lat)':>7} {'Obs':>7} {'Err00':>6} {'ErrLat':>7} {'Drift':>6} {'Src':>5}")
        print(f"  {'-'*58}")

        errors_midnight = []
        errors_latest = []
        drift_count = 0

        for r in midnight_forecasts:
            hour = r['target_hour']
            obs = obs_map.get(hour)
            lat = latest_map.get(hour)

            if obs is None or obs['temperature_2m'] is None:
                continue

            t_obs = obs['temperature_2m']
            t_f00 = r['temperature_2m']
            t_flat = lat['temperature_2m'] if lat else t_f00

            err00 = abs(t_f00 - t_obs) if t_f00 else None
            errlat = abs(t_flat - t_obs) if t_flat else None
            drift = abs(t_flat - t_f00) if t_f00 and t_flat else 0

            if err00 is not None:
                errors_midnight.append(err00)
            if errlat is not None:
                errors_latest.append(errlat)
            if drift > 0.1:
                drift_count += 1

            drift_s = f"{drift:.1f}" if drift > 0.1 else "—"
            err00_s = f"{err00:.1f}" if err00 else "—"
            errlat_s = f"{errlat:.1f}" if errlat else "—"
            src_s = obs.get('source', '?')[:5].upper()

            print(f"  {hour:>4}h {t_f00:>6.1f} {t_flat:>6.1f} {t_obs:>6.1f} {err00_s:>6} {errlat_s:>7} {drift_s:>6} {src_s:>5}")

        if errors_midnight:
            mae00 = sum(errors_midnight) / len(errors_midnight)
            maelat = sum(errors_latest) / len(errors_latest) if errors_latest else 0
            print(f"  {'-'*58}")
            print(f"  MAE (midnight): {mae00:.2f}°C | MAE (latest): {maelat:.2f}°C | Drifts: {drift_count}")

    conn.close()
    print(f"{'='*80}\n")


# ══════════════════════════════════════════════════════════════════
# INMET Status Check
# ══════════════════════════════════════════════════════════════════

def check_inmet_status():
    """Quick check if INMET API is responding with data."""
    print(f"\n[WF2] INMET API Status Check")
    print(f"{'='*50}")

    # Try listing stations first
    try:
        url = f"{INMET_API_BASE}/estacoes/T"
        req = urllib.request.Request(url, headers=INMET_HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            n_stations = len(data) if isinstance(data, list) else 0
            print(f"  Station listing: OK ({n_stations} stations)")
    except Exception as e:
        print(f"  Station listing: FAILED ({e})")
        print(f"  INMET API appears completely down")
        print(f"{'='*50}\n")
        return False

    # Try fetching data for SP Mirante (A701)
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    for station in INMET_STATIONS.get("sp", []):
        data = fetch_inmet_station(station["code"], yesterday)
        if data:
            print(f"  Station {station['code']} data: OK ({len(data)} records for {yesterday})")
            print(f"  INMET API is OPERATIONAL")
            print(f"{'='*50}\n")
            return True
        else:
            print(f"  Station {station['code']} data: NO DATA (204 or empty)")

    print(f"  INMET API: stations list OK but data endpoints returning empty")
    print(f"  Status: PARTIALLY DOWN (will use Open-Meteo fallback)")
    print(f"{'='*50}\n")
    return False


if __name__ == "__main__":
    init_db()

    parser = argparse.ArgumentParser(description="WF2: Observation Verification")
    parser.add_argument("--date", help="Target date (YYYY-MM-DD)")
    parser.add_argument("--compare", help="Compare forecast vs observed for date")
    parser.add_argument("--delay", type=int, default=24, help="Hours of delay for reliable archive data")
    parser.add_argument("--inmet-status", action="store_true", help="Check INMET API status")
    args = parser.parse_args()

    if args.inmet_status:
        check_inmet_status()
    elif args.compare:
        compare_forecast_vs_observed(args.compare)
    else:
        capture_observations(target_date=args.date, delay_hours=args.delay)
