# SPECS — Princípios Inegociáveis do Workspace CRNG/TSON/Deriv

> **Status:** vigente desde 2026-04-10
> **Autoridade:** este documento sobrescreve qualquer convenção informal em posts,
> README ou CLAUDE.md quando houver conflito.
> **Enforcement:** o subagente `compliance-officer` audita toda empreitada
> não-trivial contra este documento. Violação = rejeição, não negociação.

---

## Por que este documento existe

Em abril de 2026 uma revisão externa (Codex) encontrou 3 achados P1 (críticos) e
4 achados P2 (sérios) nos artefatos públicos do projeto. Todos os achados eram
consequência do mesmo padrão de falha: **atalhos práticos aceitos sob a justificativa
de economia de tokens, velocidade ou pragmatismo, em detrimento dos princípios
descritivo-gerativos do projeto**.

Este documento elimina a ambiguidade sob a qual esses atalhos foram tomados.

Precisão sem enviesamento é o fundamento. Economia de tokens nunca é justificativa
para atalho metodológico. Trabalho minucioso é o padrão, não a exceção.

---

## Os sete princípios inegociáveis

### P1 — Separação rigorosa entre descrição e predição

**O CRNG é uma ferramenta descritivo-generativa.** Isso significa que sua função é
reproduzir o *fingerprint estatístico* de uma série histórica: kurtose, volatility
clustering, tail index, Hurst, autocorrelação de |retornos|, permutation entropy.

Descrever não é prever. Prever exige train/test split temporal genuíno.

**Linguagem permitida** para claims descritivos:
- "reproduz"
- "reconstroi"
- "captura o fingerprint estatístico de"
- "gera séries cujas estatísticas coincidem com"
- "reproduz 42/49 métricas da amostra X"

**Linguagem proibida** em contexto descritivo:
- "prevê"
- "forecasts"
- "antecipa"
- "previsão"
- "modelo preditivo"
- qualquer formulação que sugira que o CRNG tem poder de saber o que vai acontecer

**Para usar linguagem preditiva**, o experimento precisa ter:
1. Janela de treino `[t₀, t₁]` explícita.
2. Janela de teste `[t₁+1, t₂]` explícita, disjunta da de treino.
3. Modelo calibrado *exclusivamente* em `[t₀, t₁]`.
4. Métricas reportadas *exclusivamente* em `[t₁+1, t₂]`.
5. Declaração explícita do modelo antes do teste (não escolhido depois).

Sem essas cinco condições, o claim é descritivo e a linguagem tem que refletir isso.

---

### P2 — Evidência congelada para cada claim numérico

Todo número reportado publicamente (README, posts, artigos, Apple Notes, slides)
precisa apontar para um **artefato determinístico versionado**: parquet, CSV,
JSON, ou YAML dentro do próprio repositório.

Um claim não pode apontar para um script que baixa dados ao vivo. Mesmo que o
script seja honesto e rode o mesmo método sempre, o *número reportado* é uma
afirmação sobre uma amostra específica e precisa dessa amostra preservada.

**Padrão aceitável:**

```markdown
> Na janela 2021-01-01 a 2026-04-01, nos 7 ativos listados em
> `benchmarks/snapshot_2026-04.parquet` (sha256: abc...), o CRNG reproduziu
> 42/49 métricas do fingerprint. Método disponível em
> `examples/frozen_benchmark.py` para auditoria, e em `examples/rolling_benchmark.py`
> para re-execução em qualquer janela futura.
```

**Padrão inaceitável:**

```markdown
> CRNG wins 42/49 metrics (86%).
```

(sem referência a janela, sem referência a artefato, sem sha, sem possibilidade
de terceiros reproduzirem o número exato.)

**Observação sobre dinamismo.** O princípio P2 não proíbe explorar dados ao vivo.
Proíbe fazer *claim estável* sobre dados ao vivo. O projeto pode ter simultaneamente:
- `frozen_benchmark.py` → lê parquet congelado, produz número estável, é o que
  aparece no README.
- `rolling_benchmark.py` → baixa dados ao vivo, produz relatório "estado atual",
  é explicitamente rotulado como dinâmico e nunca é citado como evidência de claim.

Os dois coexistem. O erro é confundir o segundo com o primeiro.

---

### P3 — Seleção de modelo a priori

Quando um experimento compara múltiplos modelos (preset vs auto-calibrado, v1 vs v2,
preset_A vs preset_B), a regra de seleção precisa ser declarada **antes** de ver
as métricas de avaliação.

**Exemplo aceitável:**

```python
# Selection rule (a priori, declared in script header):
#   - crypto assets → use btc() or eth() presets depending on ticker
#   - metals → use gold() preset
#   - FX → use eurusd() preset
#   - everything else → use from_data() calibrated on first 60% of window
SELECTION_RULE = {
    "BTC": btc, "ETH": eth,
    "Gold": gold,
    "EURUSD": eurusd, "USDJPY": eurusd,
    "SP500": "auto", "Oil": "auto",
}
```

**Exemplo inaceitável:**

```python
# For each asset, test both preset and auto, pick whichever is closer
# to the real kurtosis, then score the remaining 6 metrics.
test_preset = compute_all_stats(generate_returns_crng(preset_fn, n, seed=42))
test_auto = compute_all_stats(generate_returns_calibrated(prices, n, seed=42))
use_preset = abs(test_preset["Kurtosis"] - k_real) < abs(test_auto["Kurtosis"] - k_real)
```

Esse segundo padrão é **data snooping**. O modelo sempre ganha porque foi escolhido
sabendo quem ia ganhar. Seja qual for o tamanho do projeto, essa construção é
automaticamente REJECT pelo `compliance-officer`.

---

### P4 — Target ≠ Achieved, sempre reportados lado a lado

Toda tabela de presets, modelo calibrado ou experimento de calibração reporta duas
colunas distintas:

- **Target**: o valor que foi pedido ao construtor.
- **Achieved**: o valor efetivamente medido no output do generator, com n≥100.000
  e seed fixo declarado.

**Exemplo obrigatório no README:**

| Preset | Target K | Achieved K (n=100k, seed=42) | ACF1 alvo | ACF1 achieved |
|:-------|:--------:|:----------------------------:|:---------:|:-------------:|
| `gold()` | 9.26 | TBD | 0.30 | TBD |
| `btc()` | 219 | ~91 | 0.50 | TBD |

Valores exatos preenchidos após primeira bateria de medição reprodutível.
A célula "Target" nunca aparece sozinha. Se só existe target, a linha não vai
para o README até achieved ser medido.

---

### P5 — Semântica única por artefato

A pergunta "o output de `ContingencyRNG.generate(n)` representa retornos, níveis
ou alguma outra coisa?" tem exatamente uma resposta canônica no projeto. Essa
resposta é:

> **`generate(n)` produz retornos log-escala.** Para trajetória de preços,
> `prices = np.exp(np.cumsum(returns))`.

Todos os consumidores (`stats()`, `real_world_comparison.py`, `fragility_monitor`,
notebooks, exemplos, testes) devem respeitar essa semântica. Se algum arquivo
toma diffs do output antes de computar estatísticas, isso é aplicar diff duas
vezes e é bug semântico — **REJECT**.

Se no futuro for necessário mudar a semântica canônica, a mudança precisa:
1. Alterar `SPECS.md` primeiro.
2. Alterar todos os consumidores no mesmo commit.
3. Registrar errata no `REVIEWS/` com rationale.
4. Ser aprovada pelo `compliance-officer` explicitamente.

---

### P6 — Baseline iid Gaussiano honesto

O baseline para qualquer comparação "CRNG vs PRNG" é:

```python
def iid_gaussian(seed=42):
    """True iid Gaussian baseline — numpy default_rng, no oscillator."""
    import numpy as np
    class _IIDWrapper:
        def __init__(self, seed):
            self._rng = np.random.default_rng(seed)
        def generate(self, n):
            return self._rng.standard_normal(n)
        def next(self):
            return float(self._rng.standard_normal())
    return _IIDWrapper(seed)
```

Este, e somente este, é o baseline de comparação.

**NÃO** é baseline:
- `crng.gaussian()` — ainda roda pelo oscilador/ressonância/cascata. Medições
  locais mostraram `vol_clustering_acf ≈ 0.20` a lag 1, longe de iid. Serve como
  *referência interna* ("o que acontece quando o próprio CRNG tenta imitar iid"),
  mas não como baseline de comparação contra PRNG.

O docstring de `gaussian()` deve dizer, literalmente:
```
NOTA: gaussian() NÃO é o baseline de comparação CRNG vs PRNG.
Use iid_gaussian() para esse fim. gaussian() mantém o oscilador ativo
e carrega ACF residual ~0.20.
```

---

### P7 — Precisão numérica na linguagem

Resultados numéricos em qualquer texto público (README, post, artigo, note, slide,
commit message) seguem estas regras:

1. **Zero aproximações vagas**: proibido "~", "perto de", "mais ou menos",
   "aproximadamente", "por volta de" quando o número é computável.
2. **Se o número flutua com seed/amostra**: reportar `μ ± σ` **e** `n` (tamanho
   da amostra) **e** número de seeds/repetições.
3. **Arredondamento nunca favorece a tese**. Se um número calculado é 41.7/49,
   reportar 41.7/49 ou "~42/49 (exato: 41.7 em 10 seeds, σ=1.1)". Não reportar
   "42/49 (86%)" como se fosse um inteiro fixo.
4. **Unidades sempre explícitas**: dias, pontos-base, log-return, std units.
5. **sha ou hash de artefato quando o claim depende de dado congelado**: opcional
   mas recomendado para evitar drift silencioso.

---

## Regras operacionais de processo

### Invocação do compliance-officer

O subagente `compliance-officer` deve ser invocado:

1. **Antes** de iniciar qualquer empreitada que envolva:
   - Editar README
   - Escrever post, artigo, errata ou nota pública
   - Adicionar/modificar modelo em `models/`
   - Criar/atualizar snapshot em `benchmarks/`
   - Tocar em `tson/equations.py` ou qualquer arquivo TSON
   - Mudar a semântica de qualquer função pública do pacote

2. **Depois** de concluir a mesma empreitada, antes de qualquer commit ou publicação.

O retorno do agente é vinculante. `REJECT` = não prossegue. `APPROVE WITH NOTES`
= prossegue e abre tarefa de follow-up para as notas. `APPROVE` = prossegue limpo.

Mudanças triviais (typo, reformat, comentário) estão isentas — mas *o agente decide
se é trivial*, não o autor da mudança.

### Erratas

Qualquer correção de claim previamente publicado vai para `crng-package/REVIEWS/errata/`
como arquivo `.md` datado, e a entrada original é ou (a) corrigida com nota
`> **Errata 2026-MM-DD:** ver REVIEWS/errata/NNNN_...` ou (b) marcada como deprecated
se for estrutural demais para corrigir no lugar.

### Estrutura de diretórios congelada

```
crng-package/
  benchmarks/              # artefatos determinísticos — NUNCA sobrescrever snapshots
    snapshot_2026-04.parquet
    snapshot_2026-04.sha256
    README.md              # descreve cada snapshot
  models/                  # modelos formais versionados — imutáveis por versão
    btc_v1.yaml
    gold_v1.yaml
    ...
  REVIEWS/                 # revisões externas e erratas
    codex_review_2026-04.md
    errata/
      2026-04-10_tson_expected_value.md
      2026-04-10_next_catastrophe_quantile_bug.md
```

Arquivos em `benchmarks/` e `models/` são imutáveis por versão. Para atualizar,
criar `snapshot_2026-05.parquet` ou `btc_v2.yaml`, nunca editar o anterior.

### Campos obrigatórios em `models/*.yaml`

```yaml
name: btc
version: 1
created: 2026-04-10
author: ale-brotto

train_window:
  start: 2018-01-01
  end:   2023-12-31

target_fingerprint:
  kurtosis: 219
  vol_acf_lag1: 0.50
  tail_3sigma_pct: ...
  hurst: ...

achieved_fingerprint:
  n_samples: 100000
  seed: 42
  kurtosis: ...
  vol_acf_lag1: ...
  tail_3sigma_pct: ...
  hurst: ...

validation_window:
  start: 2024-01-01
  end:   2026-04-01

validation_fingerprint:
  # métricas do mesmo modelo, medidas em dados da validation_window
  # que o modelo NÃO viu durante calibração
  kurtosis: ...
  vol_acf_lag1: ...
  ...

hyperparameters:
  target_kurtosis: 219
  vol_clustering: 0.5
  n_oscillators: 4
  cascade_threshold: 1.2
  cascade_memory: 20

notes: >
  Livre para contexto qualitativo.
```

Os seis blocos (`train_window`, `target_fingerprint`, `achieved_fingerprint`,
`validation_window`, `validation_fingerprint`, `version`) são obrigatórios.
Um yaml sem qualquer um deles é rejeitado.

---

## Apêndice: armadilhas específicas já catalogadas

### A1. `rng.next()` não é uniforme em [0,1]

`ContingencyRNG.next()` retorna um valor centrado em zero, com std variável e
cauda gorda. **Tratá-lo como quantil é bug**. Para mapear CRNG em quantis, gerar
um lote grande, ordenar e usar `(rank + 0.5) / n`:

```python
samples = rng.generate(100_000)
ranks = np.argsort(np.argsort(samples))
quantiles = (ranks + 0.5) / len(samples)
# agora quantiles ∈ (0,1) e pode ir para np.percentile, scipy.stats.X.ppf, etc.
```

### A2. TSON — expectativa vs moda vs probabilidade de aparição

Para o modelo `ℵ(N, 1) = 1 - exp(-N(N-1)/2)`:

| Grandeza | Valor | Interpretação |
|:---------|:-----:|:--------------|
| `P(ℵ \| N=2)` | ≈ 0.632 (1−1/e) | probabilidade de haver mesmitude com N=2 |
| `Moda(N*)` | 2 | instante mais provável da primeira mesmitude |
| `E[N*]` | ≈ 2.421 | média do primeiro instante de mesmitude (Monte Carlo 10⁶) |
| `√(π/2) + 1/2` | ≈ 1.753 | aproximação inválida aqui (só serve no regime N grande) |

Qualquer texto que diga "primeiro mesmitude no segundo instante" precisa qualificar
se está falando de moda, probabilidade de aparição, ou expectativa. Os três são
leituras diferentes do mesmo fenômeno.

### A3. Presets não são modelos

`btc()`, `gold()`, `eth()`, `eurusd()` são chutes de hiperparâmetro escolhidos à
mão. Não têm train window registrada, não têm achieved fingerprint, não têm
validação out-of-sample. Servem como ponto de partida rápido em exemplos e notebooks.

**Não são modelos**. Modelos vivem em `models/*.yaml` com todos os campos obrigatórios.
Claims sérios (README, artigos, artigos científicos) devem referenciar um modelo
formal, não um preset.

---

## Histórico de mudanças

| Data | Autor | Mudança |
|:-----|:------|:--------|
| 2026-04-10 | Ale Brotto (+ Claude assist) | Criação inicial em resposta à revisão Codex. |
