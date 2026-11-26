"""
Serve para gerar os gráficos da análise a partir dos resultados das simulações.
Os arquivos são salvos na pasta resultados/.
"""

import json

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# Resultados e times acompanhados
with open("resultados/resultados.json", encoding="utf-8") as arquivo:
    R = json.load(arquivo)

ALVO = [
    "Internacional",
    "Corinthians",
    "Grêmio",
    "Mirassol",
    "Botafogo",
]

# Cores usadas para identificar cada equipe nos gráficos
COR = {
    "Internacional": "#D71920",  
    "Corinthians": "#333333",    
    "Grêmio": "#0066B3",         
    "Vitória": "#E87524",        
    "Mirassol": "#F2C500",
    "Botafogo": "#9A9A9A",
}

# Configuração visual
BG = "#FFFFFF"
INK = "#252525"
SEC = "#686868"
GRID = "#E8E8E8"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "text.color": INK,
    "axes.labelcolor": SEC,
    "xtick.color": SEC,
    "ytick.color": SEC,
})


# Botafogo: cinza com hachura preto e branco
ESTILO_BARRA = {"Botafogo": dict(hatch="///", edgecolor="#000000", linewidth=0.8)}


def limpar_eixos(ax):
    """Remove as bordas desnecessárias e suaviza os eixos."""
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)

    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)


# 1. Probabilidade de permanência na Série A
fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
limpar_eixos(ax)

# Ordena os times da maior para a menor chance de permanência
ordem = sorted(
    ALVO,
    key=lambda time: R["final"][time]["p_queda"],
)

for i, time in enumerate(ordem):
    prob = 1 - R["final"][time]["p_queda"]

    faixa_min = 1 - R["faixa_estrutural"][time][1]
    faixa_max = 1 - R["faixa_estrutural"][time][0]

    ax.barh(
        i,
        prob * 100,
        color=COR[time],
        height=0.5,
        **ESTILO_BARRA.get(time, {}),
    )

    # Faixa de variação entre as hipóteses estruturais
    ax.plot(
        [faixa_min * 100, faixa_max * 100],
        [i, i],
        color=INK,
        lw=2.2,
        solid_capstyle="round",
    )

    ax.text(
        max(faixa_max * 100, prob * 100) + 1.5,
        i,
        f"{prob * 100:.1f}%  "
        f"(faixa {faixa_min * 100:.0f}–{faixa_max * 100:.0f}%)",
        va="center",
        fontsize=10,
        color=INK,
    )

ax.set_yticks(range(len(ordem)))
ax.set_yticklabels(ordem, fontsize=11)
ax.set_xlim(0, 115)
ax.set_xlabel("Chance de permanência (%)")

ax.xaxis.grid(True, color=GRID)
ax.set_axisbelow(True)

ax.set_title(
    "Chance de permanência na Série A simulada 1 milhão de vezes",
    loc="left",
    fontsize=12,
    fontweight="bold",
)

fig.text(
    0.01,
    0.01,
    f"Linha preta: menor e maior valor entre {R.get('n_hipoteses', 16)} hipóteses estruturais "
    "(incerteza do modelo, não do Monte Carlo).",
    fontsize=8,
    color=SEC,
)

fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig("resultados/01_permanencia.png")
plt.close(fig)


# 2. Evolução das probabilidades conforme o modelo ganha novas camadas
fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=160)
limpar_eixos(ax)

nomes = [etapa[0] for etapa in R["escada"]]
posicoes = np.arange(len(nomes))

for j, time in enumerate(ALVO):
    permanencia = [
        100 * (1 - etapa[1][j])
        for etapa in R["escada"]
    ]

    ax.plot(
        posicoes,
        permanencia,
        color=COR[time],
        lw=2,
        marker="o",
        ms=5,
        mec=BG,
        mew=1.2,
    )

    ax.text(
        posicoes[-1] + 0.15,
        permanencia[-1],
        f"{time} {permanencia[-1]:.0f}%",
        color=INK,
        fontsize=9,
        va="center",
    )

ax.set_xticks(posicoes)
ax.set_xticklabels([
    nome.split(".")[0]
    for nome in nomes
])

ax.set_xlim(-0.3, len(nomes) + 1.6)
ax.set_xlabel(
    f"Camada do modelo (0 = base estática; {len(nomes) - 1} = modelo completo)"
)
ax.set_ylabel("Permanência (%)")

ax.yaxis.grid(True, color=GRID)

ax.set_title(
    "O que cada camada muda na probabilidade",
    loc="left",
    fontsize=12,
    fontweight="bold",
)

fig.tight_layout()
fig.savefig("resultados/02_escada.png")
plt.close(fig)


# 3. Sensibilidade dos resultados às hipóteses do modelo
sensibilidade = R["sens"]
central = np.array(sensibilidade[0][1])

fig, axs = plt.subplots(
    1,
    len(ALVO),
    figsize=(16, max(6.2, 0.3 * len(sensibilidade) + 1.6)),
    dpi=160,
    sharey=True,
)

cenarios = [item[0] for item in sensibilidade[1:]]

for j, (ax, time) in enumerate(zip(axs, ALVO)):
    limpar_eixos(ax)

    # Diferença em pontos percentuais em relação ao cenário central
    variacao = [
        100 * (item[1][j] - central[j])
        for item in sensibilidade[1:]
    ]

    ax.barh(
        range(len(cenarios)),
        variacao,
        color=COR[time],
        height=0.6,
        **ESTILO_BARRA.get(time, {}),
    )

    ax.axvline(0, color=INK, lw=0.8)
    ax.set_title(time, fontsize=10, fontweight="bold", loc="left")

    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)

    ax.set_xlabel(
        "Δ chance de queda (p.p.)",
        fontsize=8,
    )

axs[0].set_yticks(range(len(cenarios)))
axs[0].set_yticklabels(cenarios, fontsize=8)
axs[0].invert_yaxis()

fig.suptitle(
    "Sensibilidade: o quanto cada hipótese estrutural "
    "move a chance de queda.",
    x=0.01,
    ha="left",
    fontsize=12,
    fontweight="bold",
)

fig.tight_layout()
fig.savefig("resultados/03_sensibilidade.png")
plt.close(fig)


# 4. Distribuição das posições finais
fig, ax = plt.subplots(figsize=(9.5, 5), dpi=160)
limpar_eixos(ax)

rotulos = []
for time in ALVO:
    probabilidades = np.array(R["posicao"][time]) * 100

    ax.plot(
        range(1, 21),
        probabilidades,
        color=COR[time],
        lw=2,
        marker="o",
        ms=4,
        mec=BG,
    )

    # Identifica a posição mais provável de cada equipe
    posicao_mais_provavel = int(probabilidades.argmax())
    rotulos.append([posicao_mais_provavel + 1, probabilidades[posicao_mais_provavel] + 0.9, time])

# Afasta rótulos que ficariam um em cima do outro (picos na mesma região)
colocados = []
for x, y, time in sorted(rotulos, key=lambda r: -r[1]):
    while any(abs(x - xc) < 1.6 and abs(y - yc) < 2.6 for xc, yc in colocados):
        y += 3.0
    colocados.append((x, y))
    ax.text(
        x,
        y,
        time,
        color=INK,
        fontsize=9,
        ha="center",
    )

# Destaque para a zona de rebaixamento
ax.axvspan(
    16.5,
    20.5,
    color="#D9A09B",
    alpha=0.20,
)

ax.text(
    19.5,
    ax.get_ylim()[1] * 0.5,
    "Z4",
    color=SEC,
    ha="center",
    fontsize=9,
)

ax.set_xticks(range(1, 21))
ax.set_xlabel("Posição final")
ax.set_ylabel("Probabilidade (%)")

ax.yaxis.grid(True, color=GRID)

ax.set_title(
    "Distribuição da posição final",
    loc="left",
    fontsize=12,
    fontweight="bold",
)

fig.tight_layout()
fig.savefig("resultados/04_posicoes.png")
plt.close(fig)


print("Gráficos salvos na pasta resultados.")