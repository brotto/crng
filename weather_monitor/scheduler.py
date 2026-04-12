#!/usr/bin/env python3
"""
Weather Monitor — Scheduler
=============================
Unified scheduler for all 3 workflows.

Designed to be called by cron or Claude Code schedules.

Schedules:
  Midnight (00:05):  WF1 capture + WF3 predictions + WF2 previous day observations
  Every 3h:          WF1 re-capture + WF3 readjust + WF2 catch-up
  Daily (06:00):     XLSX report for previous day

Usage:
  python scheduler.py midnight     # Midnight run (WF1 + WF3 + WF2 prev day)
  python scheduler.py check        # 3-hourly check (WF1 + WF3 adjust)
  python scheduler.py report       # Generate daily report
  python scheduler.py all          # Run everything (for testing)
  python scheduler.py status       # Show DB status
"""

import argparse
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
from db import get_connection, init_db


def run_midnight():
    """Midnight workflow: full capture + predictions + previous day obs."""
    print(f"\n{'='*60}")
    print(f"  MIDNIGHT RUN — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    # WF1: Capture today's forecasts
    from wf1_capture_forecasts import capture_forecasts
    capture_forecasts()

    # WF3: Generate today's predictions
    from wf3_crng_predict import generate_predictions
    generate_predictions()

    # WF2: Capture yesterday's observations (24h delay = reliable)
    from wf2_verify_observations import capture_observations
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    capture_observations(target_date=yesterday, delay_hours=24)

    print(f"\n{'='*60}")
    print(f"  MIDNIGHT RUN COMPLETE")
    print(f"{'='*60}\n")


def run_check():
    """3-hourly check: re-capture forecasts + readjust predictions."""
    print(f"\n{'='*60}")
    print(f"  3H CHECK — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    # WF1: Re-capture forecasts (detect drift)
    from wf1_capture_forecasts import capture_forecasts, check_drift
    capture_forecasts()
    check_drift()

    # WF3: Readjust predictions if needed
    from wf3_crng_predict import readjust
    readjust()

    # WF2: Capture today's observations so far
    from wf2_verify_observations import capture_observations
    today = datetime.now().strftime("%Y-%m-%d")
    capture_observations(target_date=today, delay_hours=0)

    print(f"\n{'='*60}")
    print(f"  3H CHECK COMPLETE")
    print(f"{'='*60}\n")


def run_report():
    """Generate daily report for previous day."""
    from report_xlsx import generate_report
    from wf2_verify_observations import compare_forecast_vs_observed
    from wf3_crng_predict import score_predictions

    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Score results
    compare_forecast_vs_observed(yesterday)
    score_predictions(yesterday)

    # Generate XLSX
    generate_report(yesterday)


def show_status():
    """Show database status."""
    conn = get_connection()

    print(f"\n{'='*50}")
    print(f"  WEATHER MONITOR STATUS")
    print(f"{'='*50}")

    tables = {
        'wf1_forecasts':    ('WF1 Forecasts',       'captured_at',  'target_date'),
        'wf2_observations': ('WF2 Observations',    'captured_at',  'observed_date'),
        'wf3_predictions':  ('WF3 CRNG Predictions','predicted_at', 'target_date'),
        'scorecard':        ('Scorecards',          'evaluated_at', 'target_date'),
    }

    for table, (label, ts_col, date_col) in tables.items():
        try:
            count = conn.execute(f"SELECT COUNT(*) as n FROM {table}").fetchone()['n']
            if count == 0:
                print(f"  {label:<25} (empty)")
                continue
            latest = conn.execute(f"SELECT MAX({ts_col}) as t FROM {table}").fetchone()['t']
            dates = conn.execute(f"SELECT MIN({date_col}) as d1, MAX({date_col}) as d2 FROM {table}").fetchone()
            sources = ""
            if table == 'wf2_observations':
                srcs = conn.execute("SELECT DISTINCT source FROM wf2_observations").fetchall()
                sources = f"  src: {', '.join(r['source'] for r in srcs)}"
            print(f"  {label:<25} {count:>6} rows  ({dates['d1']} → {dates['d2']}){sources}")
        except Exception as e:
            print(f"  {label:<25} (error: {e})")

    # Cities with data
    cities = conn.execute(
        "SELECT DISTINCT city FROM wf1_forecasts"
    ).fetchall()
    print(f"\n  Cities: {', '.join(r['city'] for r in cities) if cities else 'none'}")

    # Date range
    dates = conn.execute(
        "SELECT MIN(target_date) as d1, MAX(target_date) as d2 FROM wf1_forecasts"
    ).fetchone()
    if dates['d1']:
        print(f"  Date range: {dates['d1']} → {dates['d2']}")

    conn.close()
    print(f"{'='*50}\n")


if __name__ == "__main__":
    init_db()

    parser = argparse.ArgumentParser(description="Weather Monitor Scheduler")
    parser.add_argument("action", choices=["midnight", "check", "report", "all", "status"],
                       help="Action to run")
    args = parser.parse_args()

    if args.action == "midnight":
        run_midnight()
    elif args.action == "check":
        run_check()
    elif args.action == "report":
        run_report()
    elif args.action == "all":
        run_midnight()
        run_check()
        run_report()
    elif args.action == "status":
        show_status()
