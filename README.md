# Simulador de rebaixamento — Brasileirão 2026

Simulação de Monte Carlo (1 milhão de campeonatos) da chance de queda de **Inter, Corinthians, Grêmio, Mirassol e Botafogo**
na reta final do Brasileirão Série A 2026.

## Como rodar
Precisa de Python 3.10 ou mais novo.

```
pip install -r requirements.txt
python carregar.py        # confere os placares com a tabela oficial (deve mostrar "clubes divergentes: 0")
python analise_v2.py      # 1 milhão de campeonatos, cerca de 4 minutos
python graficos.py        # gera os gráficos em resultados/
```

Teste rápido (menos de 1 minuto): `python analise_v2.py --n 50000 --nl 20000`

## Como atualizar depois de cada jogo
1. `jogos_2026.txt`: acrescenta uma linha com o placar, no mesmo formato. Exemplo: `37ª | 29/11 | Internacional 2 x 1 Bragantino`
2. `simular_v2.py`: apaga esse jogo da lista `RESTANTES`.
3. `dados_tabela.py` (opcional): atualiza a tabela oficial, que serve só para a conferência do `carregar.py`.
4. `analise_v2.py`: se algum jogo citado em `prox` já tiver sido disputado, troque pelo próximo jogo do clube.
5. Rodar de novo os três comandos acima.

Para acompanhar outro clube, inclua o nome em `ALVO` (em `simular_v2.py` e em `graficos.py`) e uma cor em `COR` (em `graficos.py`).

## O que o modelo considera
- Força de ataque e defesa que muda jogo a jogo (modelo de espaço de estados), com incerteza dos parâmetros e saltos raros de forma.
- Gols com menos placares extremos que a Poisson (Conway-Maxwell-Poisson) e correção de Dixon-Coles para placares baixos.
- Confronto direto entre os rivais e os critérios de desempate do regulamento (pontos, vitórias, saldo, gols pró).
- "Jogo morto" (time sem objetivo perde intensidade) e rodízio do Palmeiras.
- Vagas extras na Libertadores (Palmeiras campeão; regra da Conmebol se o Grêmio cair).
- Média sobre 144 combinações de hiperparâmetros, pesadas pelo desempenho em jogos fora da amostra.

Equações, premissas e limites: [METODOLOGIA.md](METODOLOGIA.md).

## Camada opcional: desespero
Desligada. Para ligar:

```
python analise_v2.py --desespero
python graficos_desespero.py
```

Usa `resultados/coef_desespero.json`, já incluso. Para refazer a estimativa com os dados históricos (baixados automaticamente):
`python estudo_desespero.py`

## Arquivos

| Arquivo | O que faz |
|---|---|
| `jogos_2026.txt` | placares de 2026 já disputados |
| `dados_tabela.py` | tabela oficial (conferência) e ordem dos clubes |
| `carregar.py` | lê os placares e confere com a tabela |
| `dinamico.py` | modelo de força dinâmica (ajuste e incerteza) |
| `dist.py` | distribuição de gols (Conway-Maxwell-Poisson + Dixon-Coles) |
| `simular_v2.py` | motor da simulação; lista `RESTANTES` e `ALVO` |
| `analise_v2.py` | roda tudo e grava `resultados/resultados.json` |
| `graficos.py` | gráficos 01 a 04 |
| `backtest3.py` | validação dos hiperparâmetros (opcional, demorado) |
| `estudo_desespero.py`, `graficos_desespero.py` | camada opcional de desespero |


## Aviso
Estimativas de probabilidade condicionais às premissas do modelo, não previsões garantidas.
