# Errata — Next Catastrophe Predictor: Bug de Transformação Quantil

**Data:** 2026-04-10
**Gravidade:** P1 (crítico — invalida o método)
**Arquivo original:** `experiments/next_catastrophe.py`
**Arquivo após correção:** `experiments/_DEPRECATED_next_catastrophe.py` (não executável)
**Origem do achado:** Codex review 2026-04-10, item P1.2

---

## O que estava errado

O arquivo `experiments/next_catastrophe.py` implementava quatro métodos para
prever o intervalo até o próximo evento catastrófico, partindo de 82 eventos
históricos (1900–2025). O **Método 2: CRNG-Modulated Sampling** (linhas 182–189
do original) continha este código:

```python
rng = ContingencyRNG(seed=42, target_kurtosis=8.0, vol_clustering=0.35, n_oscillators=7)

for _ in range(n_simulations):
    crng_val = rng.next()
    # Map CRNG output to a gap using the empirical distribution shape
    quantile = min(max(crng_val, 0.001), 0.999)
    gap = np.percentile(gaps, quantile * 100)
    crng_next.append(gap)
```

A intenção era usar o CRNG como "modulador" sobre a distribuição empírica de gaps,
preservando a estrutura de clustering e caudas gordas.

O problema é que essa transformação assume que `rng.next()` retorna um valor
uniforme em `[0,1]` — um quantil. **Não retorna.** `ContingencyRNG.next()` produz
valores centrados em zero, com std variável (tipicamente da ordem de 1.0–3.0) e
cauda gorda. A distribuição depende da kurtose alvo e do estado interno do
oscilador.

## O impacto empírico

Medição local (n = 100.000 samples, seed=42, target_kurtosis=8.0, vol_clustering=0.35,
n_oscillators=7):

```python
from crng import ContingencyRNG
import numpy as np

rng = ContingencyRNG(seed=42, target_kurtosis=8.0, vol_clustering=0.35, n_oscillators=7)
samples = np.array([rng.next() for _ in range(100_000)])

clipped_low  = np.sum(samples < 0.001)          # valores abaixo do piso
clipped_high = np.sum(samples > 0.999)          # valores acima do teto
in_range     = np.sum((samples >= 0.001) & (samples <= 0.999))

print(f"Abaixo de 0.001: {clipped_low/1e5:.1%}")
print(f"Entre 0.001 e 0.999: {in_range/1e5:.1%}")
print(f"Acima de 0.999: {clipped_high/1e5:.1%}")
```

Resultado típico (os números exatos variam com seed/target mas a ordem de grandeza
é constante): mais de 50% das amostras caem abaixo de 0.001 e são colapsadas para
o piso; o restante se divide entre o teto e a faixa intermediária. Isso significa
que o "quantil" passado para `np.percentile(gaps, q*100)` é dominado pelos dois
extremos: o menor gap histórico e o maior. A distribuição de gaps "CRNG-modulada"
resultante é essencialmente bi-modal em (min_gap, max_gap), nada a ver com a
distribuição de gaps original nem com o CRNG.

**Nenhuma das conclusões do Método 2 pode ser atribuída ao CRNG.** Elas são
artefatos de clipping.

## A transformação correta

Para mapear o output do CRNG em quantis `[0,1]` de forma estatisticamente válida,
é preciso passar pela CDF empírica de um lote grande de amostras:

```python
# Gerar um lote grande e construir a CDF empírica
batch = rng.generate(100_000)          # ou np.array([rng.next() for _ in range(100_000)])
sorted_batch = np.sort(batch)

def crng_to_quantile(x):
    """Mapeia um output CRNG x ∈ ℝ em quantil u ∈ (0,1) via CDF empírica."""
    idx = np.searchsorted(sorted_batch, x)
    return (idx + 0.5) / len(sorted_batch)
```

Ou, equivalentemente, se o objetivo é simplesmente gerar quantis para amostragem:

```python
# Gera N quantis uniformes(ish) através do CRNG preservando correlação temporal
raw = rng.generate(n_simulations)
ranks = np.argsort(np.argsort(raw))
quantiles = (ranks + 0.5) / len(raw)
# Agora quantiles ∈ (0,1) e pode ir para np.percentile
```

A segunda forma preserva a estrutura temporal do CRNG (autocorrelação, clustering)
enquanto garante que as entradas para `percentile` sejam uniformemente distribuídas.

## Por que o erro passou

O método estava rotulado no próprio código como "EXPERIMENTAL / PRIVATE — NOT for
publication". Isso é honesto e não foi usado em posts públicos do projeto. Porém
o experimento gerava dashboards e narrativas internas sobre "janelas prováveis de
próximo evento" que influenciaram a forma como o subprojeto `catastrophic_events`
foi descrito em material derivado (mind maps, rascunhos).

O erro também é ilustrativo de uma armadilha geral: **qualquer função que retorna
valores em ℝ não pode ser interpretada como quantil sem passar pela CDF**. Isso
vai para `SPECS.md` apêndice A1 como armadilha catalogada.

## Ação tomada

1. `experiments/next_catastrophe.py` renomeado para `_DEPRECATED_next_catastrophe.py`.
2. Topo do arquivo agora contém esta errata e um `raise RuntimeError` que impede
   execução acidental.
3. Nenhum post público precisa ser retirado (não havia post sobre esse experimento
   especificamente), mas qualquer menção futura a "CRNG prevendo catástrofes" deve
   citar esta errata.
4. O subprojeto `catastrophic_events.py` (análise de gaps, sem predição) permanece
   como estava — ele não contém o bug. Só o experimento de predição construído
   em cima dele (Método 2) está invalidado.
5. `SPECS.md` apêndice A1 foi escrito a partir deste achado.

## O que este bug não invalida

Importante separar o que foi invalidado do que não foi:

- **Invalidado:** Método 2 do `next_catastrophe.py` e qualquer conclusão derivada
  dele sobre janelas preditivas "CRNG-moduladas".
- **NÃO invalidado:** a análise empírica de gaps (distribuição, FFT, hazard rate)
  que antecede o Método 2. Esses métodos usam matemática clássica sobre a série
  histórica, sem envolver o CRNG.
- **NÃO invalidado:** a observação de que a distribuição de gaps tem estrutura
  quasi-periódica — isso é uma propriedade dos dados, independente de como o CRNG
  é ou não usado em cima.
- **NÃO invalidado:** os testes KS "20/20" entre gaps e CRNG no experimento
  precursor `catastrophic_events.py`, desde que esses testes comparem diretamente
  o output `generate()` com os gaps (o que é o uso semanticamente correto —
  `generate()` retorna uma distribuição, e KS compara duas distribuições). Isso
  precisa ser verificado no arquivo em questão, tarefa separada.

---

**Assinado:** Ale Brotto (autor original) + compliance-officer (revisão)
**Status:** corrigido, deprecated, documentado.
