# Errata — TSON: Fórmula Fechada da Expectativa vs Simulação

**Data:** 2026-04-10
**Gravidade:** P1 (crítico — conclusão central do subprojeto estava apoiada em fórmula inválida)
**Arquivo:** `crng-package/tson/equations.py` (função `expected_mesmitude_instant`)
**Artigo afetado no Apple Notes:** "TSON — Formalização Matemática da Origem do Nada"
**Memória afetada:** `.claude/projects/-Users-alebrotto-Deriv-MCP/memory/theory_tson_formalization.md`
**Origem do achado:** Codex review 2026-04-10, item P1.3

---

## O que estava errado

A versão anterior da função `expected_mesmitude_instant(Pi=1.0)` retornava:

```python
return np.sqrt(np.pi / (2.0 * Pi)) + 0.5
```

que para `Π = 1` dá `1.7533141373...`. O comentário afirmava:

> "With Π=1: E[N*] ≈ 1.753. Rounded up: the first mesmitude occurs at instant 2.
>  Interpretation: The SECOND instant creates the first Being.
>  This is the mathematical proof of Ale's thesis..."

Simultaneamente, o arquivo `tson/simulation_results.json` gerado por Monte Carlo
(modelo `model4_pure_tson`, 10⁶ trials) reportava:

```json
{
  "mean": 2.42073,
  "median": 2.0,
  "mode": 2,
  "distribution": {"2": 631848, "3": 318091, "4": 47582, "5": 2441, "6": 38}
}
```

A diferença entre a fórmula fechada (1.75) e a simulação (2.42) é de 38% — não é
erro de Monte Carlo, é erro estrutural.

## Por que a fórmula estava errada

A expressão `√(π/(2Π)) + 1/2` é a aproximação assintótica clássica do birthday
problem, válida no regime em que:
1. O "espaço" é grande (Π → 0), e
2. O número de tentativas antes da primeira colisão é grande (N → ∞).

No regime TSON canônico (Π = 1, N ≲ 5), nenhuma dessas condições é satisfeita.
A aproximação falha por completo.

## A fórmula correta

Para a variável aleatória `N*` (primeiro instante de mesmitude), dada a CDF
`F(N) = 1 - exp(-N(N-1)·Π/2)`, a expectativa exata é:

```
E[N*] = Σ_{N=0}^∞ P(N* > N)
      = Σ_{N=0}^∞ [1 - F(N)]
      = Σ_{N=0}^∞ exp(-N(N-1)·Π/2)
```

Para Π = 1, cada termo decai super-exponencialmente:

| N | exp(-N(N-1)/2) | Acumulado |
|:-:|:--------------:|:---------:|
| 0 | 1.000000 | 1.000000 |
| 1 | 1.000000 | 2.000000 |
| 2 | 0.367879 | 2.367879 |
| 3 | 0.049787 | 2.417667 |
| 4 | 0.002479 | 2.420145 |
| 5 | 0.0000454 | 2.420191 |
| 6 | 3.06e-7 | 2.420191 |

**Valor exato (8 termos já dão precisão de máquina):**
`E[N*] = 2.420190968307...`

Monte Carlo com 10⁶ trials → 2.42073 (dentro de 1σ do valor analítico).

A série converge em formato de theta-function; não há expressão fechada mais
simples, mas o cálculo numérico é trivial (loop de 10 termos).

## A interpretação correta — três quantidades distintas

O erro aconteceu porque três leituras diferentes do mesmo fenômeno foram
conflatadas num único número. As três leituras corretas:

| Quantidade | Valor (Π=1) | O que significa |
|:-----------|:-----------:|:----------------|
| **Moda(N\*)** | 2 | Instante mais provável da primeira mesmitude |
| **P(ℵ \| N=2)** | 1 − 1/e ≈ 0.6321 | Prob. de a mesmitude já ter ocorrido com 2 instantes |
| **E[N\*]** | 2.420190... | Valor esperado do primeiro instante de mesmitude |
| **Mediana** | 2 | 50% dos casos têm primeira mesmitude até N=2 |
| *(descartado)* | 1.7533 | Aproximação birthday — inválida em Π=1 |

A frase **"a primeira mesmitude ocorre no segundo instante"** é verdadeira como
*moda* e como *probabilidade de aparição em N=2*, mas **falsa** como *expectativa*.

A interpretação rigorosa, reescrita:

> *Com potência Π=1 no vazio, a primeira mesmitude é esperada no instante
> 2.42 em média. Em 63.2% dos casos ela ocorre já no segundo instante (moda);
> nos outros 36.8%, ela se atrasa — em ~32% vai para o terceiro instante, ~5%
> para o quarto, e caudas minúsculas além disso. O segundo instante é portanto
> o ponto-Arché mais provável, mas não é o único. O Ser tende a emergir no
> segundo instante, mas às vezes precisa de um terceiro.*

Isso altera a poesia do resultado, mas não a substância filosófica. A tese
central — "um instante sozinho é Nada, dois instantes criam a possibilidade
de Ser" — continua verdadeira como *estrutura modal*. Apenas a *expectativa*
é ligeiramente maior que 2, refletindo que o mundo "não precisa" do terceiro
instante em 63% dos casos, mas precisa nos outros 37%.

## O que foi corrigido

### 1. `tson/equations.py`

- `expected_mesmitude_instant(Pi)` agora computa a série exata (10–50 termos,
  corte automático quando `term < 1e-15`).
- Adicionada `birthday_approximation(Pi)` separada, documentada como inválida
  em Π=1 e preservada apenas para comparação didática.
- Adicionada `mesmitude_pmf(N, Pi)` para a função massa de probabilidade.
- Adicionada `mode_mesmitude_instant(Pi)` para a moda.
- `print_core_results` agora reporta as três quantidades lado a lado com rótulos
  explícitos.
- Cabeçalho do arquivo contém bloco de errata visível.

### 2. Texto do artigo no Apple Notes

A nota "TSON — Formalização Matemática da Origem do Nada" precisa de uma adição
no topo (antes do Resumo) com uma seção **"Errata 2026-04-10"** curta apontando
para este documento e indicando as três leituras corretas. O resto do corpo
permanece; apenas a frase "primeiro mesmitude no segundo instante" precisa ser
qualificada como moda/probabilidade, não expectativa, onde quer que apareça.

### 3. Memória do autor

`.claude/projects/-Users-alebrotto-Deriv-MCP/memory/theory_tson_formalization.md`
precisa de uma linha de errata apontando para este arquivo.

## O que este erro NÃO invalida

- **A função de potência Π(x) = x^x** permanece matematicamente correta,
  incluindo o ponto crítico em `x = 1/e` e o valor mínimo `e^(-1/e)`.
- **A equação de mesmitude `ℵ(N,Π) = 1 - exp(-N(N-1)Π/2)`** permanece correta
  como CDF — o erro foi apenas em como extrair a expectativa dela.
- **A equação Arche `Ω(N)`** permanece correta; seu output numérico não dependia
  de `expected_mesmitude_instant`.
- **A conexão Euler tríplice** (mínimo de potência em `1/e`, P(ℵ|N=2) = 1-1/e,
  normalização em `1/(1+e)`) permanece verdadeira — os três aparecimentos de `e`
  são independentes da aproximação birthday.
- **A tese filosófica central** (um instante = Nada, dois instantes = possibilidade
  de Ser, e como limiar da emergência) permanece sustentada pela moda e pela
  probabilidade de aparição em N=2.

O que muda é apenas a precisão da asserção "no segundo instante": ela é verdadeira
como ponto modal, mas a expectativa está entre 2 e 3, mais perto de 2.

---

**Assinado:** Ale Brotto (autor) + compliance-officer (revisão)
**Status:** corrigido em `equations.py`, errata emitida, Apple Notes a ser atualizado
  como próximo passo no todo list.
