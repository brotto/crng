# Errata pública — "CRNG reproduz 42 de 49 métricas de mercado (86%)" (2026-04-10)

> Vários posts publicados em março/abril de 2026 citam, como cabeça de série,
> a afirmação de que "o CRNG reproduz 42 de 49 métricas de 7 ativos reais —
> 86% — enquanto um PRNG gaussiano padrão fica em 0%". Esta errata retira essa
> afirmação. Ela foi gerada por um pipeline de benchmark com dois defeitos
> estruturais e, com os defeitos corrigidos, o número muda substancialmente.
>
> — Ale Brotto, 2026-04-10

## TL;DR

1. **Dados não-congelados.** O benchmark original re-baixava os sete ativos do
   yfinance a cada execução. O número "42/49" refere-se a um snapshot dos
   dados que ninguém pode recuperar hoje — cada execução subsequente produz
   uma amostra ligeiramente diferente e uma pontuação ligeiramente diferente.
   Por definição, não é evidência reproduzível.

2. **Bug semântico em `stats()`.** O método `ContingencyRNG.stats()` estava
   computando kurtosis e clustering sobre `np.diff(values)` em vez de
   `values`. Por convenção do projeto — agora formalizada no `SPECS.md`
   princípio P5 — `generate()` já devolve log-retornos. O `diff` extra estava
   medindo a kurtosis das *diferenças* entre retornos consecutivos, que é
   uma quantidade matematicamente diferente e artificialmente inflada. Os
   números reportados como "CRNG kurtosis" na tabela original são,
   portanto, de uma grandeza que não é a kurtosis dos retornos do CRNG.

## O número correto, congelado

O projeto agora mantém um snapshot imutável, verificado por SHA256:

- **Arquivo:** `crng-package/benchmarks/snapshot_2026-04/prices.csv`
- **SHA256:** `82f8b5e5abe2f9d084769898b8d3b6ffefc5cfbd1c2757531df76d049ec9fff5`
- **Janela:** 2021-04-10 → 2026-04-10 (5 anos, 7 ativos: Gold, EURUSD, ETH,
  BTC, SP500, Oil, USDJPY)
- **Relatório:** `crng-package/benchmarks/snapshot_2026-04/frozen_benchmark_report.json`

Com o bug de `stats()` corrigido e a amostra congelada, as comparações são:

### Kurtosis (alvo = retornos reais, n por ativo = ver relatório)

| Ativo    | Real K | CRNG K | iid Gauss K | Mais perto do real |
|:---------|------:|-------:|------------:|:-------------------:|
| Gold     | 15.39 |   8.09 |        3.04 | CRNG |
| S&P 500  |  9.47 |   7.17 |        3.04 | CRNG |
| ETH      |  8.31 |   5.95 |        3.01 | CRNG |
| Oil      |  8.26 |   6.75 |        3.04 | CRNG |
| BTC      |  6.96 |   5.77 |        3.01 | CRNG |
| USDJPY   |  5.95 |   3.06 |        3.03 | CRNG |
| EURUSD   |  4.89 |   7.73 |        3.03 | iid  |

CRNG está mais perto da kurtosis real em 6 de 7 ativos. Ele *supera* o alvo
no EURUSD — o ativo mais gaussiano do conjunto. Isso é reportado como um
erro honesto, não escondido.

### Frequência de eventos |z| > 3σ (% das observações)

| Ativo    | Real  | CRNG  | iid Gauss | Mais perto |
|:---------|------:|------:|----------:|:----------:|
| Gold     | 1.11  | 0.95  | 0.16      | CRNG |
| S&P 500  | 1.04  | 0.96  | 0.16      | CRNG |
| ETH      | 1.70  | 0.93  | 0.11      | CRNG |
| Oil      | 1.27  | 0.88  | 0.16      | CRNG |
| BTC      | 1.97  | 0.93  | 0.11      | CRNG |
| USDJPY   | 1.39  | 0.23  | 0.15      | CRNG |
| EURUSD   | 1.00  | 0.92  | 0.15      | CRNG |

CRNG está mais perto em 7 de 7. Um gerador iid gaussiano subestima a
frequência de eventos três-sigma por um fator entre 6,5× (S&P 500 e
EURUSD, empatados em 13/2 eventos exatos) e 18,0× (BTC, 36/2 eventos
exatos), calculado sobre as contagens inteiras do
`frozen_benchmark_report.json`.

### Autocorrelação de |retornos| no lag 1 (clustering)

| Ativo    | Real   | CRNG    | iid Gauss | Mais perto |
|:---------|------:|--------:|----------:|:----------:|
| Gold     | +0.103 | +0.043 | +0.035 | CRNG |
| S&P 500  | +0.177 | +0.008 | +0.035 | iid  |
| ETH      | +0.168 | +0.024 | +0.006 | CRNG |
| Oil      | +0.121 | +0.017 | +0.035 | iid  |
| BTC      | +0.145 | +0.021 | +0.006 | CRNG |
| USDJPY   | +0.102 | −0.031 | +0.033 | iid  |
| EURUSD   | +0.124 | −0.044 | +0.033 | iid  |

**Esta é a fraqueza honesta.** Tanto o CRNG quanto o iid subestimam o
clustering real, e em 4 de 7 ativos (S&P 500, Oil, USDJPY, EURUSD) o ruído
residual do iid fica mais próximo da ACF real do que o CRNG. O mecanismo
de clustering do CRNG atual é mais fraco do que o target nominal sugere.

### Resumo do snapshot

| Métrica            | CRNG ganha | iid ganha |
|:-------------------|:----------:|:---------:|
| Kurtosis           | 6          | 1         |
| Cauda 3σ           | 7          | 0         |
| ACF(\|retornos\|)  | 3          | 4         |

**Total CRNG:** 16 de 21 comparações (76,2%), contra os 42/49 (86%)
originais. A redução em relação a 86% vem de duas correções: (a) o bug
semântico de `stats()` inflava artificialmente a kurtosis medida, e
(b) o número antigo era medido sobre dados não-congelados e sem regra
a priori de seleção de modo. O novo número é reproduzível a partir do
SHA256 acima.

**Nota histórica sobre este arquivo:** a primeira versão publicada desta
errata (2026-04-10, manhã) trazia o total errado "17 de 21 (81%)" por um
erro aritmético na linha de ACF (4/3 em vez de 3/4). O compliance-officer
adversarial identificou o defeito na auditoria da tarde do mesmo dia. Os
valores desta versão são os corretos — batem linha por linha com o JSON
congelado `benchmarks/snapshot_2026-04/frozen_benchmark_report.json`.

## O que esta errata NÃO retira

- O CRNG *continua* claramente superior ao iid gaussiano em kurtosis e em
  frequência de caudas em quase todos os ativos testados. A magnitude da
  vantagem é menor do que o post original sugeria, mas existe e é
  reproduzível.
- A arquitetura (oscilladores irracionais + ressonância + cascata) e a
  transição de fase em função do parâmetro de amplificação permanecem
  intactas — nada do que foi corrigido mudou o núcleo do gerador.
- As ferramentas de análise e visualização continuam válidas.

## O que esta errata retira, explicitamente

- A afirmação numérica "42/49" ou "86% das métricas" em qualquer contexto.
- A tabela antiga com valores tipo "Gold CRNG K=11.2" e similares — aqueles
  valores eram produto do `stats()` bugado.
- Qualquer gráfico ou post que tenha a string "CRNG matches 86% of market
  metrics" ou equivalente como headline. Esses foram marcados com banner
  de retração no repositório.

## Como verificar

```bash
cd crng-package
PYTHONPATH=. python3 benchmarks/frozen_benchmark.py
```

Imprime a tabela acima e grava o relatório JSON completo em
`benchmarks/snapshot_2026-04/frozen_benchmark_report.json`. O script aborta
se a SHA256 do CSV não bater com `prices.sha256`, garantindo que a
evidência não tenha sido alterada.

## O contexto maior

Esta errata faz parte de uma revisão de disciplina do projeto introduzida
em 2026-04-10, após uma auditoria externa (Codex review) apontar sete
problemas de P1/P2. As correções e princípios que ficam são:

- `SPECS.md`: sete princípios inegociáveis, incluindo "o CRNG é descritivo,
  não preditivo" (P1), "toda evidência pública deve vir de um snapshot
  congelado com SHA256" (P2), e "target ≠ achieved, sempre reportados lado
  a lado" (P4).
- `REVIEWS/codex_review_2026-04.md`: rastreamento dos sete achados e das
  correções aplicadas.
- Agente `compliance-officer` adversarial que revisa todo trabalho antes
  de aterrissar.

— Ale Brotto
  2026-04-10 (versão corrigida da tarde; pendente de re-auditoria do
  compliance-officer após erro aritmético detectado na versão da manhã)
