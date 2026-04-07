# CRNG-Fragility Monitor -- Detecting Seneca Cliffs in Global Markets

## CRNG v0.4.0 -- Systemic Fragility Detection

### What's New

CRNG fat-tail analysis has been applied to **15 commodities and financial indicators** across four systemic tiers. The result: current market conditions (April 2026) show kurtosis levels **2x higher than COVID-19** and **10x higher than the 2008 GFC** -- a statistical signature consistent with systems approaching a Seneca Cliff.

### Motivation

Inspired by Steve Keen's analysis of Strait of Hormuz vulnerability. The CRNG approach does not attempt to model the mechanics of geopolitical disruption. Instead, it detects the **statistical signature** of systems accumulating stress -- fat tails in daily returns that precede catastrophic collapse.

### Monitored Assets (4 Tiers)

| Tier | Name | Assets |
|:---|:---|:---|
| Tier 1 | Choke Point | Brent, WTI, NatGas (Hormuz-dependent) |
| Tier 2 | Financial Stress | VIX, Gold, DXY, US 10Y Yields |
| Tier 3 | Physical Economy | SOX (Semiconductors), S&P 500 |
| Tier 4 | AI Bubble | NASDAQ, NVDA, MSFT, META |

### Method

- **Rolling kurtosis** (60-day window) on daily log-returns for each asset
- **Z-scores** relative to each asset's historical baseline
- **Fat-tailed confidence intervals** (not Gaussian) for anomaly detection
- **Concurrence detection** across Tier 1 assets (Hormuz chokepoint stress)
- **Alert system:** GREEN (normal) -> YELLOW (elevated) -> ORANGE (high) -> RED (critical), triggered by concurrent stress + kurtosis thresholds

### Key Finding: Unprecedented Fat Tails (April 2026)

Average kurtosis across 5 key commodities = **+9.05**

| Period | Avg Kurtosis | Context |
|:---|:---|:---|
| **April 2026** | **+9.05** | Current readings |
| COVID-19 (2020) | +4.58 | Global pandemic |
| GFC (2008) | +0.94 | Lehman Brothers collapse |

Current levels are 2x COVID and 10x GFC.

### Individual Asset Readings

| Asset | Kurtosis | Z-Score | Status |
|:---|:---|:---|:---|
| NATGAS_US | +16.47 | -- | Extreme (unprecedented in any historical crisis) |
| GOLD | +7.39 | -- | Critical (safe haven rush, $4,694/oz) |
| BRENT | +6.43 | -- | Critical |
| MSFT | +7.13 | -- | Critical (AI bubble fat tails) |
| WTI | +4.67 | +2.20 | Stressed (above confidence interval) |
| META | +3.75 | -- | Elevated (AI bubble fat tails) |

Natural Gas kurtosis at +16.47 has no precedent in any historical crisis window in our database (back to 1990).

### What Fat Tails Mean

A kurtosis of +16 does not mean prices are high or low. It means the **distribution of daily changes has extreme outliers** -- the system is producing moves that should be astronomically rare under normal conditions. This is the statistical fingerprint of a system under structural stress, where small perturbations can cascade into large regime shifts (the Seneca Cliff pattern: slow buildup, rapid collapse).

### Architecture

```
fragility_monitor/
  collector.py       -- Yahoo Finance + FRED data collection
  analyzer.py        -- CRNG metrics (rolling kurtosis, Z-scores, CI)
  create_visuals.py  -- Automated chart generation
  data/
    fragility.db     -- SQLite (S&P 500 since 1927, VIX since 1990)
```

### Reproduce

```bash
cd crng-package/fragility_monitor
pip install yfinance fredapi pandas scipy matplotlib
python3 collector.py --days 365
python3 analyzer.py --report
python3 analyzer.py --backtest
```

### Backtest Validation

The CRNG-Fragility framework was backtested against known crisis periods. Elevated kurtosis concurrence across tiers correctly flags:
- 2008 GFC (financial stress precedes equity collapse)
- 2020 COVID (energy + volatility spike simultaneously)
- 2022 Ukraine/energy crisis (Tier 1 choke point stress)

In each case, the fat-tail signature appeared **before** the peak drawdown, not after.

### Implication

Standard risk models (VaR, portfolio variance) assume thin-tailed return distributions. When kurtosis across multiple asset classes simultaneously exceeds historical crisis levels, these models systematically underestimate tail risk. The CRNG-Fragility Monitor provides a complementary signal: not a prediction of *what* will happen, but a measurement of how far the system has drifted from its baseline statistical regime.

Data sources (verifiable):
- Yahoo Finance: https://finance.yahoo.com
- FRED (Federal Reserve Economic Data): https://fred.stlouisfed.org
- CBOE VIX Historical Data: https://www.cboe.com/tradable_products/vix/
