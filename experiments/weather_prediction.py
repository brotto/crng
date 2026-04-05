"""
WEATHER PREDICTION: CRNG vs PRNG PERTURBATIONS
=================================================

Can CRNG improve weather prediction in restricted regions?

Real weather data from Open-Meteo (ERA5 reanalysis):
- São Paulo (tropical): 2014-2023, 3652 days
- New York (continental): 2014-2023, 3652 days
- London (maritime): 2014-2023, 3652 days

Method:
1. Analyze real weather series: kurtosis, vol clustering, regime structure
2. Build naive persistence model (tomorrow = today + noise)
3. Compare PRNG vs CRNG perturbations for forward simulation
4. Measure: which produces more realistic forecast distributions?

Sources (verifiable):
- Open-Meteo Historical API (ERA5 reanalysis, ECMWF)
  https://archive-api.open-meteo.com/v1/archive
- Cross-validation with NOAA GHCN-Daily possible via:
  https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/

Ale Brotto — 2026-04-05
"""

import numpy as np
from scipy import stats as sp_stats
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crng import ContingencyRNG

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


COLORS = {
    'bg': '#0D1117', 'text': '#E6EDF3', 'grid': '#21262D',
    'real': '#2ECC71', 'prng': '#3498DB', 'crng': '#E74C3C',
    'accent': '#FFD700',
}


# ============================================================
# DATA LOADING
# ============================================================

def load_weather(filepath):
    """Load Open-Meteo JSON data."""
    with open(filepath) as f:
        data = json.load(f)

    daily = data['daily']
    temps_mean = np.array([t if t is not None else np.nan for t in daily['temperature_2m_mean']])
    temps_max = np.array([t if t is not None else np.nan for t in daily['temperature_2m_max']])
    temps_min = np.array([t if t is not None else np.nan for t in daily['temperature_2m_min']])
    precip = np.array([p if p is not None else 0 for p in daily['precipitation_sum']])
    wind = np.array([w if w is not None else np.nan for w in daily['wind_speed_10m_max']])
    dates = daily['time']

    # Fill NaNs with linear interpolation
    for arr in [temps_mean, temps_max, temps_min, wind]:
        mask = np.isnan(arr)
        if mask.any():
            arr[mask] = np.interp(np.where(mask)[0], np.where(~mask)[0], arr[~mask])

    return {
        'dates': dates,
        'temp_mean': temps_mean,
        'temp_max': temps_max,
        'temp_min': temps_min,
        'precip': precip,
        'wind': wind,
    }


# ============================================================
# ANALYSIS
# ============================================================

def analyze_weather(data, city):
    """Full statistical analysis of weather time series."""

    print(f"\n{'='*70}")
    print(f"  {city.upper()} — WEATHER ANALYSIS")
    print(f"  {data['dates'][0]} to {data['dates'][-1]} ({len(data['dates'])} days)")
    print(f"{'='*70}")

    temp = data['temp_mean']

    # Daily changes (the key variable for prediction)
    delta_t = np.diff(temp)

    print(f"\n  --- Temperature Statistics ---")
    print(f"  Mean temp:    {np.mean(temp):.1f}°C")
    print(f"  Std temp:     {np.std(temp):.1f}°C")
    print(f"  Min temp:     {np.min(temp):.1f}°C")
    print(f"  Max temp:     {np.max(temp):.1f}°C")

    print(f"\n  --- Daily Change (ΔT) Statistics ---")
    print(f"  Mean ΔT:      {np.mean(delta_t):.3f}°C")
    print(f"  Std ΔT:       {np.std(delta_t):.3f}°C")
    print(f"  Kurtosis ΔT:  {sp_stats.kurtosis(delta_t, fisher=False):.2f}")
    print(f"  Skewness ΔT:  {sp_stats.skew(delta_t):.3f}")
    print(f"  Max |ΔT|:     {np.max(np.abs(delta_t)):.1f}°C")

    # Volatility clustering
    abs_delta = np.abs(delta_t)
    vol_acf = np.corrcoef(abs_delta[:-1], abs_delta[1:])[0, 1]
    print(f"  Vol clustering: {vol_acf:.4f} ({'YES' if vol_acf > 0.05 else 'NO'})")

    # Hurst exponent of temperature
    def hurst(x):
        n = len(x)
        ks = range(10, min(n // 4, 200))
        rs_vals = []
        for k in ks:
            chunks = [x[i:i+k] for i in range(0, n-k, k)]
            rs = []
            for ch in chunks:
                if len(ch) < k:
                    continue
                m = np.mean(ch)
                dev = np.cumsum(ch - m)
                r = np.max(dev) - np.min(dev)
                s = np.std(ch)
                if s > 0:
                    rs.append(r / s)
            if rs:
                rs_vals.append((np.log(k), np.log(np.mean(rs))))
        if len(rs_vals) < 3:
            return 0.5
        xv, yv = zip(*rs_vals)
        slope, _, _, _, _ = sp_stats.linregress(xv, yv)
        return slope

    h = hurst(temp)
    print(f"  Hurst (temp):   {h:.4f} ({'persistent' if h > 0.55 else 'random' if h > 0.45 else 'anti-persistent'})")

    # Precipitation extremes
    precip = data['precip']
    precip_nonzero = precip[precip > 0]
    if len(precip_nonzero) > 10:
        print(f"\n  --- Precipitation ---")
        print(f"  Rainy days:     {len(precip_nonzero)}/{len(precip)} ({len(precip_nonzero)/len(precip)*100:.1f}%)")
        print(f"  Mean (rainy):   {np.mean(precip_nonzero):.1f}mm")
        print(f"  Max precip:     {np.max(precip):.1f}mm")
        print(f"  Kurtosis:       {sp_stats.kurtosis(precip_nonzero, fisher=False):.2f}")

    return {
        'temp': temp,
        'delta_t': delta_t,
        'delta_k': sp_stats.kurtosis(delta_t, fisher=False),
        'delta_std': np.std(delta_t),
        'vol_acf': vol_acf,
        'hurst': h,
    }


def forecast_experiment(data, city, stats):
    """
    Forecast experiment: persistence model + noise perturbations.

    Method:
    - Train on first 7 years (2014-2020), test on last 3 (2021-2023)
    - Persistence model: T(t+1) = T(t) + noise
    - Compare PRNG noise vs CRNG noise
    - Evaluate: MAE, RMSE, forecast distribution realism
    """

    print(f"\n{'='*70}")
    print(f"  {city.upper()} — FORECAST EXPERIMENT")
    print(f"{'='*70}")

    temp = data['temp_mean']
    train_end = int(len(temp) * 0.7)  # ~7 years train
    temp_train = temp[:train_end]
    temp_test = temp[train_end:]
    delta_train = np.diff(temp_train)

    print(f"\n  Train: {train_end} days, Test: {len(temp_test)} days")
    print(f"  Train ΔT: mean={np.mean(delta_train):.3f}, std={np.std(delta_train):.3f}, "
          f"K={sp_stats.kurtosis(delta_train, fisher=False):.2f}")

    n_test = len(temp_test)
    n_sims = 200  # ensemble size
    horizons = [1, 3, 7, 14, 30]

    # Real test deltas (ground truth)
    delta_test = np.diff(temp_test)

    # Generate noise series
    np.random.seed(42)
    prng_noise = np.random.randn(n_sims, n_test) * np.std(delta_train)

    rng = ContingencyRNG(seed=42, target_kurtosis=max(stats['delta_k'], 4.0),
                         vol_clustering=max(stats['vol_acf'], 0.1),
                         n_oscillators=7)
    # Pre-generate CRNG values and center them
    crng_raw = np.array([rng.next() for _ in range(n_sims * n_test)])
    crng_centered = crng_raw - np.mean(crng_raw)
    crng_scaled = crng_centered / np.std(crng_centered) * np.std(delta_train)
    crng_noise = crng_scaled.reshape(n_sims, n_test)

    # Forward simulation
    results = {}

    for noise_label, noise in [('PRNG', prng_noise), ('CRNG', crng_noise)]:
        forecasts = np.zeros((n_sims, n_test))
        for s in range(n_sims):
            forecasts[s, 0] = temp_test[0]
            for t in range(1, n_test):
                forecasts[s, t] = forecasts[s, t-1] + noise[s, t]

        # Ensemble mean
        ensemble_mean = np.mean(forecasts, axis=0)

        # Evaluate at different horizons
        horizon_results = {}
        for h in horizons:
            if h >= n_test:
                continue

            # MAE and RMSE of ensemble mean at horizon h
            # For each starting point, forecast h days ahead
            maes = []
            for start in range(0, n_test - h, h):
                # Simulate h steps from actual start temperature
                sim_paths = np.zeros((n_sims,))
                for s in range(n_sims):
                    t_curr = temp_test[start]
                    for step in range(h):
                        t_curr += noise[s, start + step]
                    sim_paths[s] = t_curr

                pred = np.mean(sim_paths)
                actual = temp_test[start + h]
                maes.append(abs(pred - actual))

            mae = np.mean(maes)
            horizon_results[h] = {'mae': mae}

        # Distribution realism: compare simulated deltas to real test deltas
        sim_deltas = np.diff(ensemble_mean)
        if len(sim_deltas) > 10 and len(delta_test) > 10:
            sim_k = sp_stats.kurtosis(sim_deltas, fisher=False)

            # Individual simulation deltas (all simulations)
            all_sim_deltas = np.diff(forecasts, axis=1).flatten()
            all_sim_k = sp_stats.kurtosis(all_sim_deltas, fisher=False)

            # KS test: simulated deltas vs real deltas
            ks_stat, ks_p = sp_stats.ks_2samp(
                (delta_test - np.mean(delta_test)) / (np.std(delta_test) + 1e-10),
                (all_sim_deltas - np.mean(all_sim_deltas)) / (np.std(all_sim_deltas) + 1e-10)
            )
        else:
            sim_k = 0
            all_sim_k = 0
            ks_stat = 1
            ks_p = 0

        results[noise_label] = {
            'horizons': horizon_results,
            'ensemble_k': all_sim_k,
            'ks_stat': ks_stat,
            'ks_p': ks_p,
            'forecasts': forecasts,
        }

    # Print results
    real_k = sp_stats.kurtosis(delta_test, fisher=False)

    print(f"\n  Real test ΔT kurtosis: {real_k:.2f}")
    print(f"\n  {'Metric':25s} {'PRNG':>12} {'CRNG':>12} {'Winner':>10}")
    print(f"  {'-'*60}")

    # Kurtosis match
    prng_k = results['PRNG']['ensemble_k']
    crng_k = results['CRNG']['ensemble_k']
    k_winner = 'CRNG' if abs(crng_k - real_k) < abs(prng_k - real_k) else 'PRNG'
    print(f"  {'Simulated ΔT Kurtosis':25s} {prng_k:>12.2f} {crng_k:>12.2f} {k_winner:>10}")

    # KS test
    prng_ks_p = results['PRNG']['ks_p']
    crng_ks_p = results['CRNG']['ks_p']
    ks_winner = 'CRNG' if crng_ks_p > prng_ks_p else 'PRNG'
    prng_match = 'MATCH' if prng_ks_p > 0.05 else 'differ'
    crng_match = 'MATCH' if crng_ks_p > 0.05 else 'differ'
    print(f"  {'KS test p-value':25s} {prng_ks_p:>12.4f} {crng_ks_p:>12.4f} {ks_winner:>10}")
    print(f"  {'KS test result':25s} {prng_match:>12} {crng_match:>12}")

    # MAE by horizon
    crng_wins = 0
    prng_wins = 0

    print(f"\n  --- MAE by Forecast Horizon ---")
    print(f"  {'Horizon':10s} {'PRNG MAE':>12} {'CRNG MAE':>12} {'Winner':>10}")
    print(f"  {'-'*50}")

    for h in horizons:
        if h not in results['PRNG']['horizons']:
            continue
        p_mae = results['PRNG']['horizons'][h]['mae']
        c_mae = results['CRNG']['horizons'][h]['mae']
        winner = 'CRNG' if c_mae < p_mae else 'PRNG'
        if winner == 'CRNG':
            crng_wins += 1
        else:
            prng_wins += 1
        print(f"  {h:>3}d       {p_mae:>12.2f}°C {c_mae:>12.2f}°C {winner:>10}")

    # Kurtosis match counts
    if abs(crng_k - real_k) < abs(prng_k - real_k):
        crng_wins += 1
    else:
        prng_wins += 1

    # KS match counts
    if crng_ks_p > prng_ks_p:
        crng_wins += 1
    else:
        prng_wins += 1

    total = crng_wins + prng_wins
    print(f"\n  CRNG wins: {crng_wins}/{total} ({crng_wins/total*100:.0f}%)")
    print(f"  PRNG wins: {prng_wins}/{total} ({prng_wins/total*100:.0f}%)")

    return results, real_k


def chart_forecast_comparison(all_results):
    """Generate comparison charts across all cities."""

    cities = list(all_results.keys())
    n_cities = len(cities)

    fig, axes = plt.subplots(1, n_cities, figsize=(6*n_cities, 6), facecolor=COLORS['bg'])
    if n_cities == 1:
        axes = [axes]

    for i, city in enumerate(cities):
        ax = axes[i]
        ax.set_facecolor(COLORS['bg'])

        results, real_k = all_results[city]
        prng_k = results['PRNG']['ensemble_k']
        crng_k = results['CRNG']['ensemble_k']

        # Bar chart: kurtosis comparison
        bars = ax.bar([0, 1, 2], [real_k, prng_k, crng_k],
                      color=[COLORS['real'], COLORS['prng'], COLORS['crng']],
                      width=0.6, edgecolor='white', linewidth=1.5, alpha=0.85)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Real\nWeather', 'PRNG\nForecast', 'CRNG\nForecast'],
                           fontsize=10, color=COLORS['text'])
        ax.set_ylabel('ΔT Kurtosis', fontsize=11, color=COLORS['text'])
        ax.set_title(city, fontsize=14, color=COLORS['text'], fontweight='bold')

        # Value labels
        for bar, val in zip(bars, [real_k, prng_k, crng_k]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.1f}', ha='center', color=COLORS['text'], fontsize=12,
                    fontweight='bold')

        # KS p-values
        prng_p = results['PRNG']['ks_p']
        crng_p = results['CRNG']['ks_p']
        ax.text(0.5, -0.15, f'KS: PRNG p={prng_p:.3f} | CRNG p={crng_p:.3f}',
                transform=ax.transAxes, ha='center', fontsize=9,
                color=COLORS['accent'])

        for spine in ax.spines.values():
            spine.set_color(COLORS['grid'])
        ax.tick_params(colors=COLORS['text'])

    fig.suptitle('Weather ΔT Kurtosis: Real vs Simulated',
                 color=COLORS['accent'], fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig('charts/weather_kurtosis_comparison.png', dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'])
    plt.close()
    print("  ✓ weather_kurtosis_comparison.png")


def chart_mae_comparison(all_results):
    """MAE comparison across horizons and cities."""

    cities = list(all_results.keys())
    horizons = [1, 3, 7, 14, 30]

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    x = np.arange(len(horizons))
    width = 0.12
    offsets = np.linspace(-width * len(cities), width * len(cities), len(cities) * 2)

    for i, city in enumerate(cities):
        results, _ = all_results[city]

        prng_maes = [results['PRNG']['horizons'].get(h, {}).get('mae', 0) for h in horizons]
        crng_maes = [results['CRNG']['horizons'].get(h, {}).get('mae', 0) for h in horizons]

        ax.bar(x + offsets[i*2], prng_maes, width, alpha=0.7,
               color=COLORS['prng'], edgecolor='white', linewidth=0.5,
               label=f'{city} PRNG' if i == 0 else '')
        ax.bar(x + offsets[i*2+1], crng_maes, width, alpha=0.7,
               color=COLORS['crng'], edgecolor='white', linewidth=0.5,
               label=f'{city} CRNG' if i == 0 else '')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{h}d' for h in horizons], fontsize=12, color=COLORS['text'])
    ax.set_xlabel('Forecast Horizon', fontsize=12, color=COLORS['text'])
    ax.set_ylabel('MAE (°C)', fontsize=12, color=COLORS['text'])
    ax.set_title('Forecast MAE: PRNG vs CRNG Perturbations',
                 fontsize=14, color=COLORS['text'], fontweight='bold')

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['prng'], label='PRNG (Gaussian)'),
        Patch(facecolor=COLORS['crng'], label='CRNG (Fat-tailed)'),
    ]
    ax.legend(handles=legend_elements, fontsize=10, facecolor=COLORS['bg'],
              edgecolor=COLORS['grid'], labelcolor=COLORS['text'])

    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])
    ax.tick_params(colors=COLORS['text'])

    plt.tight_layout()
    plt.savefig('charts/weather_mae_comparison.png', dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'])
    plt.close()
    print("  ✓ weather_mae_comparison.png")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    os.makedirs('charts', exist_ok=True)

    print("=" * 70)
    print("  WEATHER PREDICTION: CRNG vs PRNG")
    print("  Can contingent noise improve weather forecasts?")
    print("=" * 70)

    cities = {
        'São Paulo': 'data/weather_saopaulo.json',
        'New York': 'data/weather_newyork.json',
        'London': 'data/weather_london.json',
    }

    all_stats = {}
    all_results = {}

    for city, filepath in cities.items():
        data = load_weather(filepath)
        stats = analyze_weather(data, city)
        all_stats[city] = stats

        results, real_k = forecast_experiment(data, city, stats)
        all_results[city] = (results, real_k)

    # Overall scorecard
    print(f"\n\n{'#'*70}")
    print(f"#  OVERALL SCORECARD")
    print(f"{'#'*70}")

    print(f"\n  {'City':15s} {'Real K':>8} {'PRNG K':>8} {'CRNG K':>8} {'K Winner':>10} {'PRNG KS p':>10} {'CRNG KS p':>10} {'KS Winner':>10}")
    print(f"  {'-'*85}")

    total_crng = 0
    total_prng = 0

    for city in cities:
        results, real_k = all_results[city]
        prng_k = results['PRNG']['ensemble_k']
        crng_k = results['CRNG']['ensemble_k']
        prng_p = results['PRNG']['ks_p']
        crng_p = results['CRNG']['ks_p']

        k_win = 'CRNG' if abs(crng_k - real_k) < abs(prng_k - real_k) else 'PRNG'
        ks_win = 'CRNG' if crng_p > prng_p else 'PRNG'

        if k_win == 'CRNG': total_crng += 1
        else: total_prng += 1
        if ks_win == 'CRNG': total_crng += 1
        else: total_prng += 1

        print(f"  {city:15s} {real_k:>8.2f} {prng_k:>8.2f} {crng_k:>8.2f} {k_win:>10} {prng_p:>10.4f} {crng_p:>10.4f} {ks_win:>10}")

    total = total_crng + total_prng
    print(f"\n  CRNG wins: {total_crng}/{total} ({total_crng/total*100:.0f}%)")
    print(f"  PRNG wins: {total_prng}/{total} ({total_prng/total*100:.0f}%)")

    # Charts
    print(f"\n{'='*70}")
    print(f"  GENERATING CHARTS")
    print(f"{'='*70}")

    chart_forecast_comparison(all_results)
    chart_mae_comparison(all_results)

    # Data source citation
    print(f"\n{'='*70}")
    print(f"  DATA SOURCES (verifiable)")
    print(f"{'='*70}")
    print(f"  1. Open-Meteo Historical API (ERA5 reanalysis, ECMWF)")
    print(f"     https://archive-api.open-meteo.com/v1/archive")
    print(f"     License: CC BY 4.0")
    print(f"  2. Cross-validation: NOAA GHCN-Daily")
    print(f"     https://www.ncei.noaa.gov/products/land-based-station/")
    print(f"     global-historical-climatology-network-daily")
    print(f"  3. Cross-validation: NOAA ISD (hourly)")
    print(f"     https://www.ncei.noaa.gov/products/land-based-station/")
    print(f"     integrated-surface-database")

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT COMPLETE")
    print(f"{'='*70}")
