# Codex Review — Abril 2026

**Data do review:** 2026-04-10
**Revisor:** OpenAI Codex (auditoria externa solicitada)
**Escopo:** crng-package (CRNG, TSON, exemplos, posts)
**Verdict geral:** núcleo gerador sólido; camada de validação pública e subprojeto
TSON com inconsistências que justificam erratas e reestruturação.

---

## Achados P1 (críticos)

### P1.1 — Benchmark 42/49 não reprodutível

**Arquivo:** `examples/real_world_comparison.py` linhas 39–74, 366–384
**Problema:** O script baixa dados ao vivo via `yfinance` e, para cada ativo,
escolhe entre o preset e o modelo auto-calibrado *usando a própria kurtose-alvo
do ativo como critério de seleção*, antes de pontuar as 7 métricas. Isso é
data snooping clássico (P3 de SPECS.md) combinado com amostra dinâmica (P2).
**Consequência:** o headline "42/49 (86%)" do README e dos posts não corresponde
a um experimento estável — muda por execução e é viesado por construção.
**Correção:** congelar `benchmarks/snapshot_2026-04.parquet`, introduzir regra
de seleção a priori, reportar apenas o experimento congelado como evidência,
manter `rolling_benchmark.py` como ferramenta exploratória rotulada como tal.

### P1.2 — Bug de transformação quantil no `next_catastrophe.py`

**Arquivo:** `experiments/next_catastrophe.py` linhas 182–189
**Problema:** O código trata `rng.next()` como se fosse uma variável uniforme
em [0,1], aplicando `clip(0.001, 0.999)` e passando direto para `np.percentile`.
Mas `ContingencyRNG.next()` retorna valores centrados em zero com cauda gorda,
não quantis. Medição empírica mostra que >50% das amostras colam no piso 0.001.
**Consequência:** a "previsão CRNG-modulada" é dominada por artefato de
clipping, não por transformação probabilística válida.
**Correção:** aposentar o subprojeto como deprecated, publicar errata explicando
o bug e a transformação quantil correta (CDF empírica via rank/n).

### P1.3 — Inconsistência entre fórmula fechada e simulação no TSON

**Arquivo:** `tson/equations.py` linhas 105–120
**Problema:** A função `expected_mesmitude_instant` retorna `√(π/(2Π)) + 1/2 ≈ 1.753`
para `Π=1` e o comentário afirma que isso prova que a primeira mesmitude ocorre
no segundo instante. Mas `simulation_results.json` (model4_pure_tson, 10⁶ trials)
reporta `mean = 2.421`, não 1.753. A fórmula fechada é a aproximação clássica
do birthday problem que só vale no regime `Π→0, N→∞` — inválida para `Π=1`.
**Consequência:** a conclusão filosófica central do subprojeto TSON ("primeiro
mesmitude no segundo instante") está apoiada numa fórmula que não se aplica ao
regime em questão. A interpretação correta: 2 é a *moda* e `1−1/e ≈ 63.2%` é
`P(ℵ|N=2)`, mas a *expectativa* é ≈ 2.42.
**Correção:** reescrever `equations.py` para separar moda, probabilidade e
expectativa explicitamente; publicar errata no Apple Notes; atualizar o
`theory_tson_formalization.md` da memória do autor.

---

## Achados P2 (sérios)

### P2.1 — API nos posts não existe

**Arquivo:** `posts/article_lake_vs_ocean.md` linhas 117–122
**Problema:** Posts instanciam `ContingencyRNG(seed=42, preset='gold')`, mas
`__init__` não tem parâmetro `preset`.
**Correção:** substituir por `gold(seed=42)` em todos os posts.

### P2.2 — `gaussian()` preset anuncia "no clustering" mas tem ACF ~0.20

**Arquivo:** `crng/__init__.py` linhas 408–410
**Problema:** O docstring diz "K≈3, no fat tails, no clustering", mas medição
local com n=100k mostra `vol_clustering_acf ≈ 0.206`. O preset ainda passa pelo
oscilador e ressonância, carregando ACF residual.
**Consequência:** quando usado como baseline contra CRNG, o `gaussian()`
subestima o ganho do CRNG (baseline já tem clustering parcial) ou superestima,
dependendo da métrica. Ruim em qualquer direção.
**Correção:** introduzir `iid_gaussian()` como baseline puro (numpy
`default_rng().standard_normal`, sem oscilador); atualizar docstring de
`gaussian()` para dizer explicitamente que NÃO é baseline.

### P2.3 — Preset `btc()` reporta target 219, achieved ~91

**Arquivo:** `README.md` linhas 104–112
**Problema:** Tabela de presets mostra "Target K = 219" para BTC, sem coluna
achieved. Medição local com n=100k mostra kurtose ≈ 91. A direção está certa
(cauda gorda emerge), mas o fraseado do README sugere fidelidade alcançada.
**Correção:** refazer tabela com colunas `target_K` e `achieved_K` lado a lado;
adicionar nota explicando que presets são ponto de partida e que modelos formais
(com train/validation window) vivem em `models/*.yaml`.

### P2.4 — Semântica inconsistente do `generate()`

**Arquivos:** `crng/__init__.py` `from_data()` vs `stats()` vs `examples/real_world_comparison.py`
**Problema:** `from_data()` mede kurtose em returns de log-prices; `stats()` mede
kurtose em first-differences do output; `real_world_comparison.py` trata o output
do generate() como returns direto. As três interpretações coexistem.
**Consequência:** qualquer métrica reportada depende de qual interpretação o
leitor aplica. Ambiguidade vaza para claims.
**Correção:** decidir uma vez que `generate(n)` produz log-returns, aplicar em
todos os consumidores, documentar em SPECS.md (P5).

---

## Verificação executada pelo revisor

```bash
cd crng-package && PYTHONPATH=. pytest -q
# → 25 passed
```

Medições locais do revisor:
- `gaussian().stats(100_000)` → kurtosis ≈ 2.8, vol_clustering_acf ≈ 0.206
- `btc().stats(100_000)` → kurtosis ≈ 91 (não 219)
- `next_catastrophe.py`: >50% dos `rng.next()` colam no piso 0.001 após clip

---

## Resposta do projeto

Ver `SPECS.md` (criado 2026-04-10) para os sete princípios inegociáveis que
regem o projeto a partir desta data, o subagente `compliance-officer` (criado
na mesma data) para enforcement automático em cada empreitada, e as erratas
individuais em `REVIEWS/errata/` para cada achado específico.

Plano de correção emergencial:
1. Aposentar `next_catastrophe.py` como deprecated, publicar errata [P1.2].
2. Reescrever `tson/equations.py` com separação de moda/probabilidade/expectativa,
   errata no Apple Notes [P1.3].
3. Introduzir `crng.iid_gaussian()` como baseline honesto [P2.2].
4. Refazer tabela de presets do README com target/achieved [P2.3].
5. Corrigir posts para usar `gold(seed=42)` [P2.1].
6. Congelar `benchmarks/snapshot_2026-04.parquet`, introduzir regra de seleção
   a priori, reescrever linguagem do README como descritiva [P1.1, P5].
7. Criar `models/btc_v1.yaml` como primeiro modelo formal auditável.

Depois do emergencial, refazer cada análise sob a nova disciplina.
