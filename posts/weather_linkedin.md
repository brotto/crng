# LinkedIn — Caudas Gordas no Clima

---

Modelos de previsão do tempo perturbam seus ensembles com ruído gaussiano. A suposição implícita: variações diárias de temperatura seguem uma distribuição normal.

O clima real tem caudas gordas.

Coletei 10 anos de dados diários de temperatura (ERA5/ECMWF) para São Paulo, New York e Londres. Medi a curtose das variações diárias — que quantifica quão frequentes são os eventos extremos em relação ao esperado pela curva normal (K=3).

Os resultados:

São Paulo — curtose real: 6.39. Gaussiano: 2.99. CRNG: 6.45.
New York — curtose real: 4.71. Gaussiano: 2.99. CRNG: 5.79.
Londres — curtose real: 4.98. Gaussiano: 2.99. CRNG: 5.57.

O gerador gaussiano sempre produz K ≈ 3. Não consegue representar a realidade. O CRNG captura as caudas gordas em todas as três cidades.

O caso mais revelador é São Paulo — localizada sobre a Anomalia Magnética do Atlântico Sul, onde o campo magnético terrestre é anomalamente fraco. Curtose real de 6.39, CRNG de 6.45. Quase idênticas.

Na previsão de temperatura, CRNG vence 4 de 5 horizontes em São Paulo — exatamente onde as caudas são mais gordas. Em New York e Londres, resultado misto: a sazonalidade forte dilui o efeito.

Scorecard de distribuição: CRNG vence 5 de 6 métricas (83%).

A implicação para meteorologia computacional: perturbações gaussianas subestimam sistematicamente a probabilidade de eventos climáticos extremos. Substituir a fonte de ruído por algo que respeite as caudas gordas reais pode melhorar a calibração probabilística — especialmente nos cenários que mais importam.

Terceiro domínio validado para o CRNG: eventos catastróficos (20/20), geometria temporal universal (7/10), e agora clima (3/3 curtose, 83% distribuição).

pip install crng

#meteorology #statistics #python #opensource #crng #machinelearning #datascience
