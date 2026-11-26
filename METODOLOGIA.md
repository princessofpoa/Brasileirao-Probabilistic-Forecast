# Metodologia do meu simulador estocástico de rebaixamento EXCLUSIVO 2026

## 1. Visão geral
Simula 1.000.000 de "finais de campeonato" (os jogos que faltam da lista `RESTANTES`, rodada a rodada). Em cada uma:
(i) sorteia os parâmetros de força dos times a partir da posteriori; (ii) simulam-se os jogos rodada a rodada, com intensidades que
dependem dos incentivos de cada time naquele momento; (iii) aplica-se o critério de desempate; (iv) registra a posição final.
A probabilidade de queda é a fração de simulações em que o time termina entre 17º e 20º.

## 2. Força dinâmica (espaço de estados)
Gols do mandante ~ f(λ_c), λ_c = exp(μ + h + A[c,k] − D[f,l]); visitante λ_f = exp(μ + A[f,l] − D[c,k]).
Ataque A e defesa D de cada time seguem **passeio aleatório por jogo**: A[i,0] ~ N(0,τ²), A[i,k]−A[i,k−1] ~ N(0,σ²) (idem D).
τ = encolhimento à média da liga; σ = volatilidade da forma ("momento"). μ, h: priors vagos.
Inferência: moda a posteriori por Newton esparso + **aproximação de Laplace** (covariância = inversa da Hessiana).
Cada campeonato simulado sorteia seus parâmetros dessa normal ----- incerteza de parâmetros propagada.
Opcional: **saltos** raros de forma (prob. 5%/jogo, sd 0,08) — lesões, troca de técnico.

## 3. Distribuição de gols
Marginais **Conway-Maxwell-Poisson** com média fixa e ν≈1,4 (sub-dispersão: o Brasileirão tem menos placares extremos que Poisson),
mais dependência Dixon-Coles (ρ≈−0,15) nos placares baixos. Amostragem por tabelas de CDF + rejeição.

## 4. Seleção de hiperparâmetros
Backtest *walk-forward* em 220 jogos (log-score do placar, log-loss e RPS). Os pesos dos 144 modelos (σ, τ, ρ, ν) vêm de
pseudo-BMA temperado: w ∝ exp(0,5·Σ log-score). A simulação final é uma **mistura** desses modelos.
Achados: forma é só fracamente persistente (σ ótimo 0,01–0,03); mando específico por time não é suportado pelos dados;
gols sub-dispersos; o modelo dinâmico supera o estático em +0,052±0,021 nat/jogo.

## 5. Incentivos endógenos ("jogo morto")
Antes de cada rodada simulada, calcula os melhores/piores colocados possíveis de cada time (critério pontos, vitórias).
Um time está "vivo" se algum limiar relevante (título, G4, vaga na Libertadores, Sul-Americana, rebaixamento) ainda está ao seu alcance.
Time sem nada em jogo perde intensidade (ataque −δa, defesa −δd, δ~N⁺(0,06; 0,04)). **Palmeiras** (finais de Libertadores/Copa) roda elenco
nas duas últimas rodadas (~N⁺(0,10; 0,05)). *δ e rodízio são premissas a priori, não estimados por isso a análise de sensibilidade.*

## 6. Vagas na Libertadores como variável aleatória
N_T = 6 + 1[Palmeiras campeão da Libertadores, p~Beta] + 1[Grêmio rebaixado e regra Conmebol aplicável, p=0,9].
Assim, rebaixar o Grêmio abre vaga (efeito que beneficia Santos, Atlético-MG, Bahia etc.) e muda o que está em jogo para esses times.

## 7. Confronto direto
Os jogos entre os rivais diretos são simulados explicitamente (acoplamento). Como desempate: pontos, vitórias, saldo, gols pró, confronto direto
(empates de 2 times), sorteio. Cartões não são modelados. Empates exatos no 16º/17º pelos 3 primeiros critérios: 0,13% das simulações; decididos por confronto direto: 0,08%.

## 8. Desespero: valor do jogo e ameaça (camada OPCIONAL, desligada)
Ligue com `python analise_v2.py --desespero`. Sem essa opção, o modelo é o das seções 1 a 7.

**Variáveis.** Antes de cada rodada simulada, para cada clube:
- **S = P(fica | vence) − P(fica | perde)**: o *valor do jogo* (quanto aquele resultado decide a temporada);
- **T = P(cai)**: a *ameaça* (o "medo").

Os dois saem do próprio simulador: uma rodada-piloto registra, para cada clube e rodada, a distância à linha de corte
(pontos do clube menos os do 16º colocado entre os outros 19) e o resultado do jogo; daí se constroem as tabelas
P(fica | rodada, clube, distância, resultado). Como os clubes reagem a S e isso muda o próprio S, as tabelas são recalculadas
com o desespero ligado até convergir (**ponto fixo de expectativas racionais**; variação média de S: 0,0095 → 0,0015).

**Efeito nos gols** (log da média de gols):
log λ_clube += β_S·S + β_T·T + β_ST·S·T (próprio)  +  γ_S·S_adv + γ_T·T_adv + γ_ST·S_adv·T_adv (adversário).

**Calibração (versão B, principal) `estudo_desespero.py`.** Base histórica: Série A 2006–2024 (19 temporadas, 20 clubes, 4 rebaixados),
rodadas 29 a 38, 3.682 clube-jogos. Para cada temporada e rodada: ajusta-se o mesmo modelo dinâmico só com os jogos anteriores,
simula o restante 20 mil vezes **sem** motivação nenhuma, calculam-se S e T de cada clube e guarda-se o placar real.
Depois, regressão de Poisson com *offset* = gols esperados pelo modelo, controles de calibração (log dos gols esperados) e de
"jogo morto", erros-padrão robustos agrupados por rodada. Coeficientes da especificação principal (EP):

| efeito | nos gols marcados | nos gols sofridos |
|---|---|---|
| S | −0,34 (0,22) | −0,81 (0,24) |
| T | −0,18 (0,09) | +0,15 (0,07) |
| S×T | +1,10 (0,43) | +0,86 (0,50) |

Leitura: o valor do jogo **fecha a defesa** (o clube sofre bem menos gols); a ameaça sem esperança (T alto, S baixo) **piora**
o clube; e o desespero de verdade (S e T altos juntos) melhora ataque e defesa. Teste de razão de verossimilhança das
quatro variáveis S/T: p ≈ 0,0001; o resultado se mantém em 2006–2014 e 2015–2024 separadamente, nas rodadas 34–38 e com
erro agrupado por temporada; um teste placebo (embaralhando o S do adversário dentro da rodada, 200 vezes) não reproduz o efeito.
Em cada campeonato simulado os coeficientes são sorteados da normal estimada (incerteza da estimativa propagada).
O "jogo morto" passa a usar o efeito estimado (≈ −0,01 nos gols marcados e +0,05 nos sofridos) no lugar da premissa δ.

**Hipóteses testadas e não confirmadas pelos dados:** reação depois de goleada (perder por 3+ gols: −0,12 nos gols marcados no jogo
seguinte, EP 0,07; goleada em casa: +0,02, EP 0,16) e efeito de técnico novo nos 3 primeiros jogos (2014–2024: −0,14, EP 0,07;
dado de técnico tem ruído de grafia e interinos). Ambos entram só como sensibilidade.

**Versão A (a priori, para comparação):** κ·S no ataque e na defesa, κ ~ N⁺(0,10; 0,06).

## 9. Limites
Não modela lesões/suspensões específicas, arbitragem, clima, nem possíveis punições do STJD. Se ligado, o desespero é um efeito **médio**
de 19 temporadas: não captura a personalidade de um elenco ou de um técnico específico. O controle de calibração do estudo
histórico (o modelo-base exagera um pouco as diferenças de força no fim da temporada) não é aplicado no simulador.
Probabilidades são estimativas condicionais às premissas, não previsões garantidas. Intervalo de Monte Carlo ≈ ±0,1 p.p.;
a incerteza estrutural (16 hipóteses; 21 com o desespero ligado) é maior: veja `03_sensibilidade.png`.
