# Weather Prediction Experiment — Fat Tails in Climate Data

## CRNG v0.3.0 — Weather Forecast Validation

### What's New

CRNG noise has been tested against **10 years of real weather data** from three cities across radically different climates. The result: CRNG captures the statistical shape of real climate variability that Gaussian perturbations systematically miss.

### Experiment

**Data:** 10 years of daily temperature (2014–2023) from ERA5 reanalysis (ECMWF) via Open-Meteo API.

**Cities:**
- São Paulo (tropical, South Atlantic Magnetic Anomaly)
- New York (continental, strong seasonality)
- London (maritime temperate)

**Method:** Persistence model + 200-member ensemble with PRNG (Gaussian) vs CRNG (fat-tailed) perturbations. Evaluated on kurtosis match, KS test, and MAE at 1/3/7/14/30-day horizons.

### Key Result: Kurtosis Match 3/3

Real weather has fat-tailed daily temperature changes (kurtosis >> 3). PRNG always produces K ≈ 3.0. CRNG matches the real kurtosis:

| City | Real K | PRNG K | CRNG K |
|:---|:---|:---|:---|
| São Paulo | 6.39 | 2.99 | **6.45** |
| New York | 4.71 | 2.99 | **5.79** |
| London | 4.98 | 2.99 | **5.57** |

### Distribution Scorecard: CRNG 5/6 (83%)

CRNG wins 5 of 6 distribution metrics (kurtosis + KS test) across all three cities.

### Forecast MAE

São Paulo: CRNG wins 4/5 horizons (where fat tails matter most).
New York & London: mixed results — strong seasonality dilutes the fat-tail advantage.

### Implication

Gaussian perturbations in weather ensemble models systematically underestimate extreme temperature events. Replacing the noise source with fat-tailed CRNG improves distributional realism — especially in tropical climates where it matters most.

### Reproduce

```bash
pip install crng
cd experiments/
python3 weather_prediction.py
```

Data sources (verifiable):
- Open-Meteo Historical API (ERA5, CC BY 4.0): https://archive-api.open-meteo.com/v1/archive
- NOAA GHCN-Daily: https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily
- NOAA ISD: https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database
