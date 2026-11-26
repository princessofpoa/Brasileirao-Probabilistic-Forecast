"""Refino sigma, tau, dependência (rho), inflação de empates (kappa) e subdispersão (nu)."""
import numpy as np, json, itertools, time
from carregar import carregar
from dinamico import *
from dados_tabela import TABELA
J = carregar(); times = list(TABELA); B = Base(J, times)
SIG = [0.005, 0.01, 0.02, 0.03]; TAU = [0.12, 0.18, 0.25]
SPECS = [(r, 0.0, nu) for r in (-0.10, -0.15, -0.20) for nu in (1.2, 1.3, 1.4, 1.5)]
ORIG = list(range(130, 343, 10)); rng = np.random.default_rng(3); res = {}; t0 = time.time()
for sg, tau in itertools.product(SIG, TAU):
    acc = {s: [[], [], []] for s in SPECS}
    for n0 in ORIG:
        f = ajustar(B, B.montar(n0), sg, sg, tau, 0.0); alvo = J[n0:n0 + 10]
        pm = prever(B, f, alvo, SPECS, rng=rng)
        for s in SPECS:
            a, b, c = metricas(pm[s], alvo)
            for i, v in enumerate((a, b, c)): acc[s][i].append(v)
    for s in SPECS: res[(sg, tau) + s] = tuple(np.concatenate(x) for x in acc[s])
print(f"{len(res)} configs em {time.time()-t0:.0f}s")
ref = np.load("bt_ref_ls.npy")
rows = sorted(((k, v[0].mean(), v[1].mean(), v[2].mean(), (v[0] - ref).mean(), (v[0] - ref).std(ddof=1) / np.sqrt(len(ref))) for k, v in res.items()), key=lambda r: -r[1])
print(f"{'sigma':>6}{'tau':>6}{'rho':>7}{'kappa':>7}{'nu':>5}{'logscore':>10}{'logloss':>9}{'RPS':>8}{'Δ vs estático':>16}")
for k, a, b, c, dm, se in rows[:12]: print(f"{k[0]:>6}{k[1]:>6}{k[2]:>7}{k[3]:>7}{k[4]:>5}{a:>10.4f}{b:>9.4f}{c:>8.4f}{dm:>+10.4f} ±{se:.4f}")
def perfil(pos, nome, vals):
    print(f"\nPerfil {nome}:")
    for v in vals:
        best = max((r for r in rows if r[0][pos] == v), key=lambda r: r[1]); print(f"  {nome}={v:<6} melhor logscore={best[1]:.4f}  logloss={best[2]:.4f}  RPS={best[3]:.4f}")
perfil(0, "sigma", SIG); perfil(1, "tau", TAU); perfil(2, "rho", (-0.10, -0.15, -0.20)); perfil(4, "nu", (1.2, 1.3, 1.4, 1.5))
tot = np.array([r[1] * len(ref) for r in rows]); w = np.exp(tot - tot.max()); w /= w.sum()
json.dump([dict(sigma=r[0][0], tau=r[0][1], rho=r[0][2], kappa=r[0][3], nu=r[0][4], logscore=float(r[1]), logloss=float(r[2]), rps=float(r[3]), peso=float(ww)) for r, ww in zip(rows, w)], open("backtest3_config.json", "w"), indent=1)
print("\nPesos pseudo-BMA (top 8):")
for r, ww in list(zip(rows, w))[:8]: print(f"  sigma={r[0][0]} tau={r[0][1]} rho={r[0][2]} kappa={r[0][3]} nu={r[0][4]}  peso={ww:.3f}")
