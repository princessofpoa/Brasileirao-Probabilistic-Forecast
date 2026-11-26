"""escada de camadas, execução final, sensibilidade estrutural, impacto dos jogos
e efeito da regra de vagas.
Uso:  python analise_v2.py [--n 1000000] [--nl 200000] [--desespero]
--desespero liga a camada de desespero calibrada (requer resultados/coef_desespero.json, de estudo_desespero.py)."""
import json, time, os, argparse, csv
import numpy as np
import simular_v2 as S
from simular_v2 import Cfg, rodar, TIMES, IDX, ALVO, RESTANTES

os.makedirs("resultados", exist_ok=True)
NBT = 220                       # jogos fora da amostra no backtest (para pesos BMA)
CENTRAL = dict(sigma=0.02, tau=0.18, rho=-0.15, nu=1.4)
t_ini = time.time()
def log(*a): print(f"[{time.time()-t_ini:6.0f}s]", *a, flush=True)

def cfgs_bma(beta=0.5, **ov):
    L = json.load(open("backtest3_config.json")); tot = np.array([c["logscore"] * NBT for c in L])
    w = np.exp(beta * (tot - tot.max())); w /= w.sum()
    return [(Cfg(sigma=c["sigma"], tau=c["tau"], rho=c["rho"], nu=c["nu"], **ov), ww) for c, ww in zip(L, w) if ww > 2e-4]

def p_rel(out):
    z = out["rank"] >= 16; return z.mean(0)

def ic(p, n): return 1.96 * np.sqrt(p * (1 - p) / n)

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1_000_000); ap.add_argument("--nl", type=int, default=200_000)
ap.add_argument("--desespero", action="store_true", help="liga a camada de desespero (valor do jogo e ameaça)")
a = ap.parse_args()
R = {}

USAR_DESESPERO = a.desespero
DES = dict(desespero="B", morto_dados=True) if USAR_DESESPERO else {}
if USAR_DESESPERO:
    log("desespero: construindo as tabelas de valor do jogo e ameaça (ponto fixo de expectativas)")
    _tab, hist_pf = S.equilibrio(cfgs_bma(jump_prob=0.05, **DES), N=300_000, iters=2, log=log)

base = dict(param_incerto=False, incentivos=False, confronto_direto=False, jump_prob=0.0)
LAD = [
    ("0. Estático + Poisson (≈ v1)", [(Cfg(sigma=0.003, tau=0.35, rho=-0.08, nu=1.0, **base), 1)]),
    ("1. + gols calibrados (ν, ρ) e encolhimento", [(Cfg(sigma=0.003, tau=0.18, rho=-0.15, nu=1.4, **base), 1)]),
    ("2. + incerteza dos parâmetros", [(Cfg(sigma=0.003, tau=0.18, rho=-0.15, nu=1.4, **{**base, "param_incerto": True}), 1)]),
    ("3. + forma dinâmica (passeio aleatório)", [(Cfg(**CENTRAL, **{**base, "param_incerto": True}), 1)]),
    ("4. + saltos raros de forma", [(Cfg(**CENTRAL, **{**base, "param_incerto": True, "jump_prob": 0.05}), 1)]),
    ("5. + desempate: confronto direto", [(Cfg(**CENTRAL, **{**base, "param_incerto": True, "jump_prob": 0.05, "confronto_direto": True}), 1)]),
    ("6. + incentivos endógenos e rodízio", [(Cfg(**CENTRAL, param_incerto=True, jump_prob=0.05, confronto_direto=True, incentivos=True), 1)]),
    ("7. + média sobre hiperparâmetros (BMA)", cfgs_bma(jump_prob=0.05)),
]
if USAR_DESESPERO: LAD.append(("8. + desespero (valor do jogo e ameaça, calibrado)", cfgs_bma(jump_prob=0.05, **DES)))
escada = []
for nome, cf in LAD:
    o = rodar(cf, a.nl, seed=11); p = p_rel(o); escada.append((nome, [float(p[IDX[t]]) for t in ALVO]))
    log(nome, " | ".join(f"{t[:5]} {p[IDX[t]]*100:5.1f}" for t in ALVO))
R["escada"] = escada

log(f"execução final: {a.n:,} campeonatos (BMA, todas as camadas{', desespero calibrado' if USAR_DESESPERO else ''})")
F = rodar(cfgs_bma(jump_prob=0.05, **DES), a.n, seed=2026)
rank, pts, res, x_lib, n_t = F["rank"], F["pts"], F["res"], F["x_lib"], F["n_t"]
z = rank >= 16; N = a.n
pr = z.mean(0)
R["final"] = {t: dict(p_queda=float(pr[IDX[t]]), ic_mc=float(ic(pr[IDX[t]], N)), pts_atual=int(S.PTS0[IDX[t]]),
              pts_med=float(pts[:, IDX[t]].mean()), pts_p5=float(np.percentile(pts[:, IDX[t]], 5)), pts_p95=float(np.percentile(pts[:, IDX[t]], 95))) for t in TIMES}
for t in ALVO: log(f"FINAL {t:<14} queda {pr[IDX[t]]*100:5.1f}% ±{ic(pr[IDX[t]], N)*100:.2f}  pts {R['final'][t]['pts_med']:.1f} [{R['final'][t]['pts_p5']:.0f}-{R['final'][t]['pts_p95']:.0f}]")
R["empate_fronteira"] = dict(n_tie=int(F["n_tie_fronteira"]), n_cd=int(F["n_cd"]), N=N)
log(f"empates 16º/17º nos 3 primeiros critérios: {F['n_tie_fronteira']/N*100:.3f}% ; decididos por confronto direto: {F['n_cd']/N*100:.3f}%")


from collections import Counter
cod = np.sort(np.argsort(-rank, axis=1)[:, :4], axis=1)   
chave = cod[:, 0] * 8000 + cod[:, 1] * 400 + cod[:, 2] * 20 + cod[:, 3]
cnt = Counter(chave.tolist()); comb = []
for k, v in cnt.most_common(8):
    ids = [k // 8000, (k // 400) % 20, (k // 20) % 20, k % 20]; comb.append(([TIMES[i] for i in ids if TIMES[i] not in ("Chapecoense", "Remo")], v / N))
R["combinacoes_z4"] = comb
k_alvo = z[:, [IDX[t] for t in ALVO]].sum(1); R["quantos_dos_5_caem"] = {int(k): float((k_alvo == k).mean()) for k in range(len(ALVO) + 1)}
R["gre_inter_juntos"] = float((z[:, IDX["Grêmio"]] & z[:, IDX["Internacional"]]).mean())
R["nenhum_gre_inter"] = float((~z[:, IDX["Grêmio"]] & ~z[:, IDX["Internacional"]]).mean())
for k in ("rank",): pass

ptsord = np.take_along_axis(pts, np.argsort(rank, axis=1), 1)
R["pts_16"] = [float(np.percentile(ptsord[:, 15], q)) for q in (10, 50, 90)]; R["pts_17"] = [float(np.percentile(ptsord[:, 16], q)) for q in (10, 50, 90)]
# distribuição de posição final
R["posicao"] = {t: [float((rank[:, IDX[t]] == p).mean()) for p in range(20)] for t in ALVO}

R["risco_por_pontos"] = {}
for t in ALVO:
    pt_t = pts[:, IDX[t]]; R["risco_por_pontos"][t] = {int(v): float(z[pt_t == v, IDX[t]].mean()) for v in range(38, 52) if (pt_t == v).sum() > 500}
R["n_t"] = {int(k): float((n_t == k).mean()) for k in np.unique(n_t)}
R["p_lib_campea"] = float(x_lib.mean())

if USAR_DESESPERO:
    ND = max(a.n // 2, 50_000)
    VERS = [("sem desespero", dict()),
            ("A: a priori (κ·S)", dict(desespero="A")),
            ("B: calibrado (principal)", DES),
            ("B sem interação S×T", dict(desespero="B_sem_interacao", morto_dados=True)),
            ("B + pós-goleada (dados)", dict(desespero="B_goleada", morto_dados=True)),
            ("B + reação extra do Inter (+0,15)", dict(**DES, reacao_time="Internacional", reacao=0.15)),
            ("B + reação extra do Inter (+0,30)", dict(**DES, reacao_time="Internacional", reacao=0.30))]
    var = {}
    for nome, kw in VERS:
        o = rodar(cfgs_bma(jump_prob=0.05, **kw), ND, seed=4242); p = p_rel(o)
        var[nome] = {t: float(p[IDX[t]]) for t in ALVO}
        log("desespero", nome, " | ".join(f"{t[:5]} {p[IDX[t]]*100:5.1f}" for t in ALVO))
    nomes_c, b_c, V_c = S.coeficientes("principal"); bc = dict(zip(nomes_c, b_c))
    efeito = {}
    for t in ALVO:
        i = IDX[t]; e = {}
        for st in (2, 3):
            s_, t_ = float(F["S_med"][st - 1, i]), float(F["T_med"][st - 1, i])
            pro = bc["S"] * s_ + bc["T"] * t_ + bc["SxT"] * s_ * t_; sof = bc["S_adv"] * s_ + bc["T_adv"] * t_ + bc["SxT_adv"] * s_ * t_
            e[f"rodada {35 + st}"] = dict(S=s_, T=t_, gols_pro=float(np.exp(pro) - 1), gols_sofridos=float(np.exp(sof) - 1))
        efeito[t] = e
    R["desespero"] = dict(variantes=var, n=ND, efeito=efeito, ponto_fixo=hist_pf,
                          coef={k: float(v) for k, v in bc.items()}, ep={k: float(np.sqrt(V_c[j, j])) for j, k in enumerate(nomes_c)})

idj = {f"{h} x {v}": k for k, (h, v, _) in enumerate(RESTANTES)}
impacto = []; cenarios = {}
for k, (h, v, st) in enumerate(RESTANTES):
    deslocamento = 0.0; linha = {}
    for r in (2, 1, 0):
        m = res[:, k] == r; pr_r = m.mean()
        cond = z[m].mean(0); deslocamento += pr_r * np.abs(cond - pr).sum()
        linha[r] = dict(p=float(pr_r), **{t: float(cond[IDX[t]]) for t in ALVO})
    impacto.append((f"{h} x {v}", st, float(deslocamento) * 100, linha))
impacto.sort(key=lambda x: -x[2]); R["impacto"] = impacto[:12]
R["cenarios_proximo"] = {t: {} for t in ALVO}
prox = {"Internacional": "Internacional x Bragantino", "Grêmio": "Corinthians x Grêmio", "Mirassol": "Mirassol x Athletico-PR",
        "Corinthians": "Corinthians x Grêmio", "Vitória": "Vitória x Santos", "Botafogo": "Botafogo x Bahia"}
prox = {t: j for t, j in prox.items() if t in ALVO}
for t, jogo in prox.items():
    k = idj[jogo]; casa = RESTANTES[k][0] == t
    for nome, r in (("vitória", 2 if casa else 0), ("empate", 1), ("derrota", 0 if casa else 2)):
        m = res[:, k] == r; R["cenarios_proximo"][t][nome] = dict(jogo=jogo, p_resultado=float(m.mean()), p_queda=float(z[m, IDX[t]].mean()))
log("impacto top:", [(i[0], round(i[2], 1)) for i in impacto[:5]])

gre_cai = rank[:, IDX["Grêmio"]] >= 16
n_com = 6 + x_lib.astype(int) + gre_cai.astype(int); n_sem = 6 + x_lib.astype(int)
cand = ["Athletico-PR", "Bahia", "Atlético-MG", "Santos", "Bragantino", "Vasco"]
R["lib"] = {t: dict(com_regra=float((rank[:, IDX[t]] < n_com).mean()), sem_regra=float((rank[:, IDX[t]] < n_sem).mean())) for t in cand}
ks = idj["Vitória x Santos"]
R["santos"] = {nome: dict(p_resultado=float((res[:, ks] == r).mean()), p_lib=float((rank[res[:, ks] == r, IDX["Santos"]] < n_com[res[:, ks] == r]).mean()),
                           p_gremio_cai=float(gre_cai[res[:, ks] == r].mean())) for nome, r in (("Santos vence (Vitória x Santos)", 0), ("empate", 1), ("Vitória vence", 2))}
log("Libertadores via tabela (com regra vs sem):", {t: (round(R['lib'][t]['com_regra'] * 100, 1), round(R['lib'][t]['sem_regra'] * 100, 1)) for t in cand})

C = dict(param_incerto=True, jump_prob=0.05, confronto_direto=True, incentivos=True, **DES)
def cen(**kw): return [(Cfg(**{**CENTRAL, **C, **kw}), 1)]
SENS_DES = [("desespero desligado", dict(desespero="nao", morto_dados=False)), ("desespero A (a priori: κ·S)", dict(desespero="A", morto_dados=False)),
            ("desespero sem interação S×T", dict(desespero="B_sem_interacao")), ("desespero + pós-goleada (dados)", dict(desespero="B_goleada")),
            ("reação extra do Inter (+0,15)", dict(reacao_time="Internacional", reacao=0.15)),
            ("'jogo morto' desligado", dict(incentivos=False)), ("'jogo morto' a priori (δ=0,06)", dict(morto_dados=False)),
            ("'jogo morto' a priori dobrado (δ=0,12)", dict(morto_dados=False, delta_a=0.12, delta_d=0.12, delta_sd=0.06))]
SENS_SEM = [("incentivos: δ = 0 (sem 'jogo morto')", dict(incentivos=False)),
            ("incentivos: δ dobrado (0,12)", dict(delta_a=0.12, delta_d=0.12, delta_sd=0.06))]
SENS = [("central", {})] + (SENS_DES if USAR_DESESPERO else SENS_SEM) + [
        ("rodízio do Palmeiras = 0", dict(rodizio=0.0, rodizio_sd=0.0)), ("rodízio do Palmeiras = 0,25", dict(rodizio=0.25, rodizio_sd=0.08)),
        ("regra Conmebol vale p/ Grêmio: 0%", dict(p_regra=0.0)), ("Palmeiras campeão Libertadores: 25%", dict(p_lib=0.25)), ("Palmeiras campeão Libertadores: 75%", dict(p_lib=0.75)),
        ("gols Poisson (ν=1,0)", dict(nu=1.0)), ("subdispersão forte (ν=1,5)", dict(nu=1.5)),
        ("encolhimento fraco (τ=0,35)", dict(tau=0.35)), ("encolhimento forte (τ=0,12)", dict(tau=0.12)),
        ("forma quase estática (σ=0,005)", dict(sigma=0.005)), ("forma muito volátil (σ=0,06)", dict(sigma=0.06)),
        ("sem saltos de forma", dict(jump_prob=0.0)), ("saltos 15%/jogo (sd 0,15)", dict(jump_prob=0.15, jump_sd=0.15))]
sens = []
for nome, kw in SENS:
    o = rodar(cen(**kw), a.nl, seed=77); p = p_rel(o); sens.append((nome, [float(p[IDX[t]]) for t in ALVO]))
    log("sens", nome, " | ".join(f"{p[IDX[t]]*100:5.1f}" for t in ALVO))
R["sens"] = sens
CENARIOS = {"reação extra do Inter (+0,15)"}          # cenário hipotético: fica fora da faixa estrutural
cent = np.array(sens[0][1]); M_ = np.array([s[1] for s in sens if s[0] not in CENARIOS])
R["n_hipoteses"] = int(len(M_))
R["faixa_estrutural"] = {t: [float(M_[:, i].min()), float(M_[:, i].max())] for i, t in enumerate(ALVO)}

R["tempo_total_s"] = time.time() - t_ini
json.dump(R, open("resultados/resultados.json", "w"), indent=1, ensure_ascii=False)
np.savez_compressed("resultados/amostra_final.npz", rank=rank[:200000], pts=pts[:200000], res=res[:200000], x_lib=x_lib[:200000], n_t=n_t[:200000])
log("concluído")
