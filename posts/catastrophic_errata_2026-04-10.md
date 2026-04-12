# Errata pública — "CRNG prevê a próxima catástrofe" (2026-04-10)

> A série de posts *catastrophic_** publicada em março/abril de 2026 afirmava
> que o CRNG, calibrado sobre 125 anos de eventos catastróficos, indicava a
> data provável do próximo grande evento. Esta errata declara formalmente que
> aquela conclusão estava errada por três motivos estruturais e não deve
> continuar sendo citada como evidência de capacidade preditiva do CRNG.
>
> — Ale Brotto, 2026-04-10

## TL;DR

1. **O bug.** `experiments/next_catastrophe.py` chamava `rng.next()` e tratava
   o retorno como se fosse um número em [0, 1]. `rng.next()` devolve um
   *log-retorno* centrado em zero, não uma quantile. O mapeamento "amostra →
   data" estava, portanto, operando sobre o domínio errado.

2. **O escopo.** Mesmo que o mapeamento estivesse correto, o CRNG é, por
   construção, um gerador *descritivo*: ele reproduz a *assinatura
   estatística* (kurtosis, caudas, clustering) de uma amostra real. Ele não
   contém um modelo causal dos eventos catastróficos e, portanto, não pode
   fazer previsões sobre quando o próximo evento ocorrerá. Essa separação
   descritivo/preditivo é agora o princípio P1 do `SPECS.md` do projeto.

3. **O "data snooping".** A data específica que o post mencionava foi obtida
   após experimentar variantes do pipeline e reportar a que parecia mais
   interessante. Isso é seleção posterior (post-hoc model selection), que
   invalida qualquer intervalo de confiança associado ao resultado.

## O que o post dizia

> *"Nosso algoritmo, calibrado sobre 82 eventos catastróficos entre 1900 e
> 2025, aponta o próximo evento para [data]. A probabilidade estimada é de
> [X]%."*

## O que o código realmente fazia

Reconstruindo o pipeline, o núcleo do script era aproximadamente:

```python
rng = ContingencyRNG(seed=42, target_kurtosis=9.26)
amostras = [rng.next() for _ in range(10_000)]
# tentativa de mapear amostras → datas em [1900, 2025+h]
datas = [1900 + (h_total) * a for a in amostras]
```

O erro está na linha `rng.next()`. A função devolve um **log-retorno** — um
número centrado em zero, podendo ser negativo, sem nenhuma semântica de
quantile. Ao ser usado como coeficiente linear em `[1900, 1900+h]`, valores
negativos levavam o "evento previsto" para antes de 1900, e valores grandes
o empurravam para muito depois de 2025. O que o pipeline efetivamente
descartava ao filtrar "datas válidas" era um subconjunto não-uniforme do
output do gerador, cujo formato estatístico não corresponde à distribuição
real de intervalos entre catástrofes.

## Como teria sido o mapeamento correto

Se a intenção fosse transformar saídas do CRNG em quantiles, a operação
correta é usar a CDF empírica do gerador:

```python
samples = rng.generate(100_000)
ranks = np.argsort(np.argsort(samples))
quantiles = (ranks + 0.5) / len(samples)   # cada amostra → posição em [0, 1]
```

Aplicar isso ao problema original ainda não produziria uma previsão de
próxima catástrofe — apenas produziria uma sequência de quantiles cuja
distribuição é, por construção, uniforme em [0, 1]. Isso apenas torna
visível o segundo problema: **não há ponte entre os quantiles do CRNG e a
data do próximo evento real**. O gerador não foi calibrado sobre uma
variável temporal; ele foi calibrado sobre estatísticas de distribuição. Os
dois domínios são incomensuráveis sem um modelo causal explícito, que o
CRNG não tem.

## Por que nenhum refinamento salva a previsão

Suponhamos que alguém construísse um mapeamento temporal legítimo: por
exemplo, ajustasse um processo de Poisson homogêneo sobre as 82 datas de
catástrofe e gerasse amostras desse processo usando CRNG como fonte de
uniformidades. Este seria um procedimento defensável, mas:

1. O gerador de números uniformes padrão do NumPy serviria exatamente ao
   mesmo propósito, com resultados indistinguíveis.
2. O CRNG não adicionaria nenhum sinal novo — o mérito estatístico do CRNG
   está em reproduzir *caudas gordas e clustering*, propriedades ausentes do
   Poisson.
3. A previsão resultante seria, por construção, uma projeção do processo de
   Poisson calibrado sobre 82 pontos, com intervalos de confiança largos o
   suficiente para cobrir décadas.

Em outras palavras: o único ganho informativo da previsão original estava
no acaso de uma data plausível, não em estrutura replicável.

## O que o CRNG de fato faz bem (para ficar claro)

Para não lançar o bebê junto com a água do banho: a capacidade descritiva
do CRNG permanece intacta. No benchmark congelado 2026-04 (veja
`benchmarks/snapshot_2026-04/frozen_benchmark_report.json`, SHA256
`82f8b5e5abe2f9d084769898b8d3b6ffefc5cfbd1c2757531df76d049ec9fff5`), sobre
sete ativos reais e cinco anos de retornos diários, o CRNG auto-calibrado
(`from_data`) está mais próximo da kurtosis real do que um PRNG Gaussiano
em 6 de 7 ativos, e mais próximo da frequência de eventos |z| > 3σ em 7 de
7. Essa é a afirmação defensável. Ela é **descritiva**, não preditiva.

Toda afirmação futura sobre previsão precisa passar pelo princípio P1 do
`SPECS.md`: o CRNG é um gerador descritivo, não um forecaster. Qualquer
post, thread ou commit que o trate como forecaster está, por definição,
fora do escopo do projeto e sujeito a errata como esta.

## O que foi feito no código

- `experiments/next_catastrophe.py` foi renomeado para
  `experiments/_DEPRECATED_next_catastrophe.py`.
- Adicionado um bloco `raise RuntimeError(...)` no topo do arquivo, que
  aborta qualquer tentativa de execução com uma mensagem apontando para
  esta errata.
- O código original foi preservado abaixo do `raise`, em comentário, para
  auditoria.
- Cada post da série `catastrophic_*.md` recebeu um cabeçalho de
  depreciação apontando para `REVIEWS/errata/2026-04-10_next_catastrophe_quantile_bug.md`
  e para esta errata pública.

## Para quem leu o post original

Se você viu a data anunciada, pode esquecê-la. Ela não foi produzida por
um pipeline cientificamente defensável, e o projeto não tem — e não
pretende ter — capacidade de antecipar grandes eventos sistêmicos
específicos. A tese descritiva do CRNG continua, mas não é a mesma tese.

— Ale Brotto, com revisão do agente compliance-officer
  2026-04-10
