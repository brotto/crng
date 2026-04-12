Two Lorenz attractors. Same equations. Same chaos. Only the noise differs.

Left: Gaussian (PRNG). Right: Contingent (CRNG).

The numbers tell the story:

Gaussian creates 20 artificial regime changes.
CRNG — only 11.

Gaussian vol clustering collapses to 0.07.
CRNG holds at 0.98.

The overlay reveals it all: Gaussian diffuses the butterfly into noise. CRNG preserves the deterministic structure.

Chaos is deterministic. The noise you add should respect that.

Try it yourself: brotto.tech/lorenz

pip install crng

#chaos #lorenz #python #opensource #crng #datascience
