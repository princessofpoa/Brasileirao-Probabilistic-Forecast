"""
Gráficos da camada de desespero.
  05_desespero_dados.png   o que 19 temporadas dizem: efeito do valor do jogo (S) e da ameaça (T) nos gols
  06_desespero_mapa.png    onde cada clube está no mapa S × T nas rodadas 37 e 38
  07_desespero_impacto.png chance de queda sem e com desespero (+ cenários de reação do Inter)
"""
import csv, json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

R = json.load(open("resultados/resultados.json", encoding="utf-8"))
if "desespero" not in R:
    raise SystemExit("Rode antes:  python analise_v2.py --desespero   (a camada de desespero vem desligada por padrão)")
C = json.load(open("resultados/coef_desespero.json", encoding="utf-8"))["principal"]
H = list(csv.DictReader(open("resultados/historico_covariaveis.csv", encoding="utf-8")))

ALVO = list(R["desespero"]["efeito"])           
COR = {"Internacional": "#D71920", "Corinthians": "#333333", "Grêmio": "#0066B3",
       "Vitória": "#E87524", "Mirassol": "#F2C500", "Botafogo": "#9A9A9A"}
BORDA = {"Botafogo": "#000000"}                      
BG, INK, SEC, GRID = "#FFFFFF", "#252525", "#686868", "#E8E8E8"
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": BG, "axes.facecolor": BG, "text.color": INK,
                     "axes.labelcolor": SEC, "xtick.color": SEC, "ytick.color": SEC})


def limpar_eixos(ax):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("bottom", "left"):
        ax.spines[lado].set_color(GRID)
    ax.tick_params(length=0)


b = dict(zip(C["nomes"], C["b"])); V = np.array(C["V"]); pos = {k: i for i, k in enumerate(C["nomes"])}
S_h = np.array([float(o["S"]) for o in H]); T_h = np.array([float(o["T"]) for o in H])


def efeito(Sv, Tv, lado):
    """efeito (log) e desvio-padrão (método delta) nos gols marcados ('pro') ou sofridos ('sof')."""
    ks = ["S", "T", "SxT"] if lado == "pro" else ["S_adv", "T_adv", "SxT_adv"]
    Xg = np.stack([Sv, Tv, Sv * Tv], -1); bb = np.array([b[k] for k in ks]); ii = [pos[k] for k in ks]
    Vs = V[np.ix_(ii, ii)]
    return Xg @ bb, np.sqrt(np.einsum("...i,ij,...j->...", Xg, Vs, Xg))


fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=160, sharey=True)
NIV = [(0.1, "ameaça baixa (T = 0,1)", "#86b6ef"), (0.5, "ameaça média (T = 0,5)", "#2a78d6"), (0.9, "ameaça alta (T = 0,9)", "#0d366b")]
for ax, lado, titulo in ((axs[0], "pro", "Gols MARCADOS pelo clube"), (axs[1], "sof", "Gols SOFRIDOS pelo clube")):
    limpar_eixos(ax); ax.axhline(0, color=INK, lw=0.8)
    for Tv, rot, cor in NIV:
        perto = np.abs(T_h - Tv) < 0.12
        smax = float(np.quantile(S_h[perto], 0.99)) if perto.sum() > 30 else 0.5  
        Sg = np.linspace(0, smax, 60); e, sd = efeito(Sg, np.full_like(Sg, Tv), lado)
        ax.fill_between(Sg, 100 * (np.exp(e - 1.96 * sd) - 1), 100 * (np.exp(e + 1.96 * sd) - 1), color=cor, alpha=0.12, lw=0)
        ax.plot(Sg, 100 * (np.exp(e) - 1), color=cor, lw=2.2)
        ax.text(Sg[-1] + 0.01, 100 * (np.exp(e[-1]) - 1), rot, color=INK, fontsize=8.5, va="center")
    ax.set_xlim(0, 1.2); ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Valor do jogo  S = P(fica | vence) − P(fica | perde)")
    ax.set_title(titulo, loc="left", fontsize=11, fontweight="bold"); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
axs[0].set_ylabel("Variação vs. o previsto pelo modelo (%)")
d = json.load(open("resultados/historico_desespero.json", encoding="utf-8"))["descr"]
fig.suptitle(f"O que dizem {d['temporadas'][1] - d['temporadas'][0] + 1} temporadas do Brasileirão ({d['temporadas'][0]}–{d['temporadas'][1]}, rodadas 29–38)",
             x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.text(0.01, 0.005, f"{d['n']:,} clube-jogos. Faixas: intervalo de 95% (EP robusto por rodada). Curvas só até onde há dados (percentil 99 de S para cada nível de T).".replace(",", "."),
         fontsize=8, color=SEC)
fig.tight_layout(rect=(0, 0.03, 1, 0.95)); fig.savefig("resultados/05_desespero_dados.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=160); limpar_eixos(ax)
tg, sg = np.meshgrid(np.linspace(0, 1, 201), np.linspace(0, 1, 201))             # x = ameaça T, y = valor do jogo S
liq = 100 * (np.exp(efeito(sg, tg, "pro")[0] - efeito(sg, tg, "sof")[0]) - 1)     
NB = 50; dens, _, _ = np.histogram2d(S_h, T_h, bins=NB, range=[[0, 1], [0, 1]])
for _ in range(3):                                                   
    pad = np.pad(dens, 1, mode="edge"); dens = sum(pad[1 + di:NB + 1 + di, 1 + dj:NB + 1 + dj] for di in (-1, 0, 1) for dj in (-1, 0, 1)) / 9
alfa = 0.6 * np.sqrt(np.clip(dens[np.minimum((sg * NB).astype(int), NB - 1), np.minimum((tg * NB).astype(int), NB - 1)] / 0.35, 0, 1))
cmap = LinearSegmentedColormap.from_list("div", ["#b3302f", "#e34948", "#f0efec", "#3987e5", "#1c5cab"])
lim = 40
im = ax.imshow(np.clip(liq, -lim, lim), origin="lower", extent=(0, 1, 0, 1), aspect="auto", cmap=cmap,
               norm=TwoSlopeNorm(0, -lim, lim), alpha=alfa, interpolation="bilinear")
ax.scatter(T_h, S_h, s=3, color="#7a7a7a", alpha=0.25, lw=0, zorder=2)
ef = R["desespero"]["efeito"]
for t in ALVO:
    e37, e38 = ef[t]["rodada 37"], ef[t]["rodada 38"]
    if max(e37["S"], e38["S"], e37["T"], e38["T"]) < 0.01:
        ax.scatter([0.005], [0.005], s=70, color=COR[t], edgecolor=BORDA.get(t, BG), lw=1.2, zorder=5)
        ax.annotate(f"{t} (salvo)", (0.005, 0.005), xytext=(8, 8), textcoords="offset points", fontsize=9, color=INK)
        continue
    ax.annotate("", xy=(e38["T"], e38["S"]), xytext=(e37["T"], e37["S"]),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.0, shrinkA=6, shrinkB=6), zorder=4)
    ax.scatter([e37["T"]], [e37["S"]], s=90, color=COR[t], edgecolor=BORDA.get(t, BG), lw=1.2, zorder=5)
    ax.scatter([e38["T"]], [e38["S"]], s=90, facecolor=BG, edgecolor=COR[t] if t != "Botafogo" else "#000000", lw=2, zorder=5)
    dx, ha = (-8, "right") if e37["T"] > 0.75 else (8, "left")
    ax.annotate(t, (e37["T"], e37["S"]), xytext=(dx, -14), textcoords="offset points", fontsize=9, color=INK, zorder=6, ha=ha)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Efeito líquido: gols marcados ÷ sofridos (%)", color=SEC); cb.outline.set_visible(False)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("Ameaça  T = P(cair)"); ax.set_ylabel("Valor do jogo  S = P(fica | vence) − P(fica | perde)")
ax.set_title("Desespero que ajuda × medo que paralisa. Analise de onde cada clube está", loc="left", fontsize=12, fontweight="bold")
fig.text(0.01, 0.01, "Ponto cheio: 37ª rodada · círculo vazio: 38ª (média das simulações) · pontos cinza: situações reais de 2006–2024.\n"
         "Azul = rende acima do previsto pelo modelo; vermelho = abaixo; a cor some onde não há dados históricos.", fontsize=7.8, color=SEC)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig("resultados/06_desespero_mapa.png"); plt.close(fig)

var = R["desespero"]["variantes"]
sem, com, apri = var["sem desespero"], var["B: calibrado (principal)"], var["A: a priori (κ·S)"]
ordem = sorted(ALVO, key=lambda t: com[t])
fig, ax = plt.subplots(figsize=(10, 5.4), dpi=160); limpar_eixos(ax)
for i, t in enumerate(ordem):
    x0, x1, xa = 100 * sem[t], 100 * com[t], 100 * apri[t]
    ax.plot([x0, x1], [i, i], color=GRID, lw=6, solid_capstyle="round", zorder=1)
    ax.scatter([x0], [i], s=80, facecolor=BG, edgecolor=COR[t] if t != "Botafogo" else "#000000", lw=2, zorder=3)
    ax.scatter([xa], [i], s=55, marker="D", facecolor=BG, edgecolor=SEC, lw=1.2, zorder=3)
    ax.scatter([x1], [i], s=110, color=COR[t], edgecolor=BORDA.get(t, BG), lw=1.2, zorder=4)
    ax.text(max(x0, x1, xa) + 2.5, i, f"{x0:.1f}% → {x1:.1f}%  (Δ {x1 - x0:+.1f} p.p.)", va="center", fontsize=9.5, color=INK)
iI = ordem.index("Internacional")
for nome, rot, dy in (("B + reação extra do Inter (+0,15)", "+0,15", 0.30), ("B + reação extra do Inter (+0,30)", "+0,30", -0.30)):
    xr = 100 * var[nome]["Internacional"]
    ax.plot([xr, 100 * com["Internacional"]], [iI + dy, iI + dy], color=COR["Internacional"], lw=1, ls=(0, (2, 2)), zorder=2)
    ax.scatter([xr], [iI + dy], s=60, marker="^", color=COR["Internacional"], edgecolor=BG, lw=0.8, zorder=5)
    ax.text(xr - 1.5, iI + dy, f"reação extra {rot}: {xr:.1f}%", fontsize=8.5, color=INK, ha="right", va="center")
ax.set_yticks(range(len(ordem))); ax.set_yticklabels(ordem, fontsize=11); ax.set_xlim(-2, 128); ax.set_ylim(-0.6, len(ordem) - 0.35)
ax.set_xlabel("Chance de queda (%)"); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.set_title("Chance de queda sem e com a variável de desespero", loc="left", fontsize=12, fontweight="bold")
n_txt = f"{R['desespero']['n']:,}".replace(",", ".")
fig.text(0.01, 0.01, "Círculo vazio: sem desespero · losango: versão A (a priori) · círculo cheio: versão B (calibrada, modelo final)\n"
         f"Triângulos: Inter com reação extra hipotética (limite superior). {n_txt} campeonatos por versão, mesma semente.", fontsize=7.8, color=SEC)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig("resultados/07_desespero_impacto.png"); plt.close(fig)
print("Gráficos do desespero salvos na pasta resultados.")
