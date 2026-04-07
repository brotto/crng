CRNG-Fragility Monitor Is Detecting Unprecedented Fat-Tail Signals in Global Markets

The numbers are worse than 2008 and 2020 combined. Not in price — in tail risk.

[IMAGE: 01_kurtosis_comparison.png]

I built a fragility monitor based on CRNG (Contingent Random Number Generator) theory. It tracks excess kurtosis across 15 global commodities and indices in real time. Kurtosis measures how likely extreme moves are relative to what Gaussian models predict. Higher kurtosis means fatter tails — more "impossible" moves waiting to happen.

The current readings stopped me cold.

2026 average excess kurtosis across monitored symbols: +9.05.
COVID crash (March 2020): +4.58.
Global Financial Crisis (2008): +0.94.

Read that again. The current fat-tail regime is 2x COVID and 10x the GFC. This is not a price forecast. It is a measurement of how unstable the statistical structure of these markets has become.

The single most extreme reading comes from natural gas.

[IMAGE: 02_natgas_kurtosis.png]

NATGAS excess kurtosis: +16.47. I have no historical precedent for this number. For context, a normal distribution has excess kurtosis of zero. COVID peak for energy was around +5. NATGAS is currently at +16.47. The distribution of daily returns has effectively abandoned any resemblance to normality. Tail events that Gaussian VaR models assign near-zero probability to are occurring regularly.

Gold is at $4,694/oz with excess kurtosis of +7.39. WTI crude has a Z-score of +2.20 — sitting above the upper bound of its historical confidence interval. Six of the 15 symbols I monitor are in what the system classifies as "critical" regime.

[IMAGE: 03_dashboard.png]

The dashboard above shows the current state. Green/yellow/red classification based on kurtosis thresholds calibrated against historical crisis periods. Six symbols red. Five yellow. Only four in normal regime.

This work was partly inspired by Steve Keen's analysis of Strait of Hormuz dependency. The concentration of global oil supply through a single chokepoint is a textbook fat-tail generator — low probability, extreme consequence.

[IMAGE: 04_hormuz_dependency.png]

But Hormuz is just one node. The fragility monitor reveals that fat tails are not isolated to energy. They are correlated across commodities, metals, and volatility indices simultaneously. This cross-asset tail correlation is itself the signal. Individual fat tails are manageable. Synchronized fat tails across asset classes are how systemic crises begin.

Why does CRNG detect this when standard models do not? Because standard risk models assume returns are drawn from thin-tailed distributions (Gaussian or at best Student-t with fixed degrees of freedom). They treat kurtosis as a static parameter. CRNG treats the tail structure as dynamic — it can thicken and thin in response to real contingency pressure in the system. When CRNG kurtosis spikes, it means the generating process itself has changed regime, not just that a few outlier days occurred.

[IMAGE: 05_kurtosis_buildup.gif]

The animation above shows the kurtosis buildup over the past 90 days. The acceleration is visible. This is not a sudden spike — it is a progressive regime shift. The tails have been fattening steadily, which is more concerning than a single shock event because it suggests structural rather than transient fragility.

What does WTI look like when you overlay fat-tail confidence intervals instead of Gaussian ones?

[IMAGE: 06_oil_ci.png]

The CRNG-calibrated confidence interval for Brent crude is substantially wider than the Gaussian CI. Price has already breached the Gaussian upper bound multiple times — events that "should not happen" under normal assumptions. Under CRNG intervals, these moves are expected. The model is not surprised. Gaussian VaR is.

To be clear about what this is and is not. This is not a prediction that markets will crash. It is a measurement that the probability distribution of returns has shifted to a regime where extreme moves — in either direction — are far more likely than any Gaussian-based risk model currently accounts for. It means VaR is underestimated. It means options are likely underpriced on the tails. It means portfolio hedges sized using normal assumptions are too small.

Six symbols in critical regime. Average kurtosis 10x the GFC. NATGAS at levels never recorded. Cross-asset tail correlation rising.

The data does not tell you what will happen. It tells you the system is fragile.

pip install crng | github.com/brotto/crng


---

IMAGE/GIF PLACEMENT MAP

Position 1 (after first subtitle paragraph):
[IMAGE: 01_kurtosis_comparison.png]
Path: /Users/alebrotto/Deriv MCP/crng-package/fragility_monitor/assets/01_kurtosis_comparison.png
Format: 5:2 image (PNG)
Description: Bar chart comparing crisis kurtosis — GFC vs COVID vs 2026

Position 2 (after "natural gas" introduction):
[IMAGE: 02_natgas_kurtosis.png]
Path: /Users/alebrotto/Deriv MCP/crng-package/fragility_monitor/assets/02_natgas_kurtosis.png
Format: 5:2 image (PNG)
Description: NATGAS kurtosis spike visualization

Position 3 (after dashboard description):
[IMAGE: 03_dashboard.png]
Path: /Users/alebrotto/Deriv MCP/crng-package/fragility_monitor/assets/03_dashboard.png
Format: 5:2 image (PNG)
Description: Current alert dashboard with green/yellow/red classification

Position 4 (after Hormuz/Keen reference):
[IMAGE: 04_hormuz_dependency.png]
Path: /Users/alebrotto/Deriv MCP/crng-package/fragility_monitor/assets/04_hormuz_dependency.png
Format: 5:2 image (PNG)
Description: Strait of Hormuz supply dependency chart

Position 5 (after regime shift explanation):
[IMAGE: 05_kurtosis_buildup.gif]
Path: /Users/alebrotto/Deriv MCP/crng-package/fragility_monitor/assets/05_kurtosis_buildup.gif
Format: 5:2 animated GIF
Description: Animated kurtosis buildup over 90 days

Position 6 (after Brent CI explanation):
[IMAGE: 06_oil_ci.png]
Path: /Users/alebrotto/Deriv MCP/crng-package/fragility_monitor/assets/06_oil_ci.png
Format: 5:2 image (PNG)
Description: Brent crude with CRNG fat-tail CI vs Gaussian CI
