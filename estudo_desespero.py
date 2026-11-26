"""
Estudo histórico do "desespero" Brasileirão Série A, 2006 a 2024.

Pergunta: times com muito em jogo rendem MAIS do que o modelo de força previa?

Pra cada temporada e cada rodada alvo t (29ª a 38ª):
  1. ajustei o modelo dinâmico só com os jogos disputados ANTES da rodada t;
  2. ele simula o restante do campeonato M vezes, SEM nenhum efeito de motivação;
  3. para cada clube de cada jogo da rodada t calcula a partir dessas simulações:
       S   = P(fica | vence) − P(fica | perde)                 valor do jogo para o rebaixamento
       T   = P(cai)                                            ameaça (o "medo")
       Sto = Σ_{k∈{1,4,6}} [P(top-k | vence) − P(top-k | perde)]  valor do jogo no topo (controle)
       M   = 1 se nenhum objetivo (título, G4…G8, G12, Z4) ainda está em jogo (prob. < 1%)
       G   = 1 se perdeu o jogo anterior por 3+ gols ;  Gc = 1 se essa goleada foi em casa
       N   = 1 nos 3 primeiros jogos de um técnico novo (2014+, quando há dado de técnico)
     e guarda os gols esperados pelo modelo (offset) e o placar real.
Depois estima, por máxima verossimilhança de Poisson com offset,
    log E[gols do clube] = log(gols esperados) + α + γ·mando + β_próprio·X_clube + β_adv·X_adversário
com erros-padrão robustos agrupados por rodada (estimador sanduíche).

"""
import csv, datetime as dt, json, os, sys, time, urllib.request
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.special import gammaln
from dinamico import Base, ajustar, indices_finais

URL = "https://raw.githubusercontent.com/adaoduque/Brasileirao_Dataset/master/campeonato-brasileiro-full.csv"
_ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
CSV = _ARGS[0] if _ARGS else "dados_historicos/campeonato-brasileiro-full.csv"
RODADAS = range(29, 39)
M_SIM = 20000
LIMIARES_TOPO = (1, 4, 6)
LIMIARES_VIVO = (1, 4, 5, 6, 7, 8, 12, 16)   # título, G4–G8 (vagas variaram entre temporadas), G12, Z4


def ler(caminho):
    if not os.path.exists(caminho):
        os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
        urllib.request.urlretrieve(URL, caminho)
    jogos = []
    for r in csv.DictReader(open(caminho, encoding="utf-8")):
        d, m, y = map(int, r["data"].split("/"))
        temp = 2020 if (y == 2021 and m <= 3) else y          # temporada 2020 terminou em fev/2021
        if temp < 2006: continue
        jogos.append(dict(id=int(r["ID"]), temp=temp, rodada=int(r["rodata"]), data=dt.date(y, m, d),
                          casa=r["mandante"].strip(), fora=r["visitante"].strip(),
                          gc=int(r["mandante_Placar"]), gv=int(r["visitante_Placar"]),
                          tc=r["tecnico_mandante"].strip(), tf=r["tecnico_visitante"].strip()))
    jogos.sort(key=lambda g: (g["data"], g["id"]))
    return jogos


def tecnico_novo(J):
    """1 nos 3 primeiros jogos de um técnico que (i) não tinha aparecido antes no time na temporada,
    (ii) substitui outro e (iii) fica ao menos 3 jogos seguidos (exclui interinos de suspensão)."""
    seq = defaultdict(list)
    for g in J:
        seq[g["casa"]].append((g["id"], g["tc"])); seq[g["fora"]].append((g["id"], g["tf"]))
    novo = {}
    for t, s in seq.items():
        for gid, _ in s: novo[(gid, t)] = 0
        vistos, i = set(), 0
        while i < len(s):
            c, j = s[i][1], i
            while j < len(s) and s[j][1] == c: j += 1
            if c and i > 0 and c not in vistos and j - i >= 3:
                for k in range(i, min(i + 3, j)): novo[(s[k][0], t)] = 1
            vistos.add(c); i = j
    return novo


def _cond(mask, val, base, a=2.0):
    """média de val dado mask, suavizada para a média incondicional (evita divisões por quase-zero)."""
    return (val[mask].sum() + a * base) / (mask.sum() + a)


def analisar_temporada(args):
    temp, J = args
    rng = np.random.default_rng(temp)
    times = sorted({g["casa"] for g in J} | {g["fora"] for g in J}); idx = {x: i for i, x in enumerate(times)}; n = len(times)
    novo = tecnico_novo(J) if temp >= 2014 else {}
    saida = []
    for t in RODADAS:
        alvo_r = [g for g in J if g["rodada"] == t]
        if not alvo_r: continue
        D = min(g["data"] for g in alvo_r)
        treino = [g for g in J if g["data"] < D]; resto = [g for g in J if g["data"] >= D]
        alvo = [g for g in alvo_r if (g["data"] - D).days <= 6]
        pts = np.zeros(n); w = np.zeros(n); gd = np.zeros(n); gf = np.zeros(n); ult = {}
        for g in treino:
            c, f, x, y = idx[g["casa"]], idx[g["fora"]], g["gc"], g["gv"]
            pts[c] += 3 * (x > y) + (x == y); pts[f] += 3 * (y > x) + (x == y)
            w[c] += x > y; w[f] += y > x; gd[c] += x - y; gd[f] += y - x; gf[c] += x; gf[f] += y
            ult[g["casa"]] = (x - y, True); ult[g["fora"]] = (y - x, False)
        B = Base(treino, times); d = B.montar(len(treino)); fit = ajustar(B, d, 0.02, 0.02, 0.18, 0.0)
        S, th, Sg = indices_finais(B, fit)
        L = np.linalg.cholesky(Sg + 1e-9 * np.eye(len(S)))
        Z = th + rng.standard_normal((M_SIM, len(S))) @ L.T
        mu, h = Z[:, 0], Z[:, 1]; eta = Z[:, 2:2 + n]; A = Z[:, 2 + n:2 + 2 * n]; Dd = Z[:, 2 + 2 * n:2 + 3 * n]
        hc = np.array([idx[g["casa"]] for g in resto]); fc = np.array([idx[g["fora"]] for g in resto])
        lh = np.exp(mu[:, None] + h[:, None] + eta[:, hc] + A[:, hc] - Dd[:, fc])
        la = np.exp(mu[:, None] + A[:, fc] - Dd[:, hc])
        x = rng.poisson(lh); y = rng.poisson(la)
        P = np.tile(pts, (M_SIM, 1)); W = np.tile(w, (M_SIM, 1)); GD = np.tile(gd, (M_SIM, 1)); GF = np.tile(gf, (M_SIM, 1))
        for k in range(len(resto)):
            c, f, xv, yv = hc[k], fc[k], x[:, k], y[:, k]
            P[:, c] += 3 * (xv > yv) + (xv == yv); P[:, f] += 3 * (yv > xv) + (xv == yv)
            W[:, c] += xv > yv; W[:, f] += yv > xv; GD[:, c] += xv - yv; GD[:, f] += yv - xv; GF[:, c] += xv; GF[:, f] += yv
        chave = P * 1e8 + W * 1e6 + (GD + 500) * 1e2 + GF + rng.random((M_SIM, n))
        ordem = np.argsort(-chave, axis=1); rank = np.empty_like(ordem); rank[np.arange(M_SIM)[:, None], ordem] = np.arange(n)
        fica = (rank < n - 4).astype(float)
        pos = {g["id"]: k for k, g in enumerate(resto)}
        for g in alvo:
            k = pos[g["id"]]; c, f = idx[g["casa"]], idx[g["fora"]]
            r = np.sign(x[:, k] - y[:, k])
            for lado, eu, adv, rv, gols, esp in (("casa", c, f, r, g["gc"], lh[:, k].mean()), ("fora", f, c, -r, g["gv"], la[:, k].mean())):
                nome = times[eu]; vW, vD, vL = rv == 1, rv == 0, rv == -1
                base = fica[:, eu].mean()
                fW, fD, fL = _cond(vW, fica[:, eu], base), _cond(vD, fica[:, eu], base), _cond(vL, fica[:, eu], base)
                sto = 0.0
                for kk in LIMIARES_TOPO:
                    top = (rank[:, eu] < kk).astype(float); bt = top.mean()
                    sto += _cond(vW, top, bt) - _cond(vL, top, bt)
                vivo = any(min(p, 1 - p) > 0.01 for p in ((rank[:, eu] < kk).mean() for kk in LIMIARES_VIVO))
                u = ult.get(nome, (0, False))
                saida.append(dict(temp=temp, rodada=t, jogo=g["id"], lado=lado, time=nome, adv=times[adv],
                                  gols=gols, esperado=float(esp), S=fW - fL, S_emp=fW - fD, T=1 - base, Sto=sto,
                                  M=0 if vivo else 1, G=int(u[0] <= -3), Gc=int(u[0] <= -3 and u[1]),
                                  N=novo.get((g["id"], nome), np.nan) if temp >= 2014 else np.nan,
                                  pts=int(pts[eu])))
    return saida


def poisson_glm(y, X, off):
    b = np.zeros(X.shape[1])
    for _ in range(60):
        mu = np.exp(off + X @ b); g = X.T @ (y - mu); H = (X * mu[:, None]).T @ X
        dl = np.linalg.solve(H, g); b += dl
        if np.abs(dl).max() < 1e-10: break
    mu = np.exp(off + X @ b); H = (X * mu[:, None]).T @ X; Hi = np.linalg.inv(H)
    ll = float(np.sum(y * np.log(mu) - mu - gammaln(y + 1)))
    return b, Hi, mu, ll


def sanduiche(X, y, mu, Hi, grupos):
    sc = X * (y - mu)[:, None]; acc = {}
    for gi, s in zip(grupos, sc): acc[gi] = acc.get(gi, 0) + s
    meat = sum(np.outer(v, v) for v in acc.values()); ng = len(acc)
    return Hi @ meat @ Hi * ng / (ng - 1)


def montar_observacoes():
    J = ler(CSV); por_temp = defaultdict(list)
    for g in J: por_temp[g["temp"]].append(g)
    with ProcessPoolExecutor(max_workers=2) as ex:
        obs = [o for lista in ex.map(analisar_temporada, sorted(por_temp.items())) for o in lista]
    chave = {(o["jogo"], o["lado"]): o for o in obs}          # pareia com as variáveis do adversário
    for o in obs:
        a = chave[(o["jogo"], "fora" if o["lado"] == "casa" else "casa")]
        for v in ("S", "S_emp", "T", "Sto", "M", "G", "Gc", "N", "esperado"): o[v + "_adv"] = a[v]
    with open("resultados/historico_covariaveis.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(obs[0].keys())); wr.writeheader(); wr.writerows(obs)
    return obs


def ler_observacoes():
    obs = list(csv.DictReader(open("resultados/historico_covariaveis.csv", encoding="utf-8")))
    for o in obs:
        for k, v in o.items():
            if k not in ("lado", "time", "adv"):
                o[k] = float(v) if v not in ("", "nan") else np.nan
    return obs


ESPEC = {
    "E0: só calibração": [],
    "E1: valor do jogo (S)": ["S", "S_adv"],
    "E2: ameaça (T)": ["T", "T_adv"],
    "E3: S + T": ["S", "S_adv", "T", "T_adv"],
    "E4: S + T + jogo morto + topo": ["S", "S_adv", "T", "T_adv", "M", "M_adv", "Sto", "Sto_adv"],
    "E5: E4 + pós-goleada": ["S", "S_adv", "T", "T_adv", "M", "M_adv", "Sto", "Sto_adv", "G", "G_adv", "Gc"],
    "E7a: S + T + morto + calibração": ["S", "S_adv", "T", "T_adv", "M", "M_adv", "cal", "cal_adv"],
    "E7: E7a + interação S×T (PRINCIPAL)": ["S", "S_adv", "T", "T_adv", "SxT", "SxT_adv", "M", "M_adv", "cal", "cal_adv"],
    "E8: E7 + pós-goleada": ["S", "S_adv", "T", "T_adv", "SxT", "SxT_adv", "M", "M_adv", "cal", "cal_adv", "G", "G_adv", "Gc"],
    "E9: E8 + técnico novo (2014+)": ["S", "S_adv", "T", "T_adv", "SxT", "SxT_adv", "M", "M_adv", "cal", "cal_adv", "G", "G_adv", "Gc", "N", "N_adv"],
}


if __name__ == "__main__":
    t0 = time.time()
    obs = ler_observacoes() if ("--reusar" in sys.argv and os.path.exists("resultados/historico_covariaveis.csv")) else montar_observacoes()
    print(f"{len(obs)} observações (clube-jogo) em {time.time() - t0:.0f}s")
    col = lambda v: np.array([o[v] for o in obs], float)
    y = col("gols"); off = np.log(col("esperado")); casa = np.array([o["lado"] == "casa" for o in obs], float)
    grp = [(o["temp"], o["rodada"]) for o in obs]
    D = {v: col(v) for v in ("S", "S_adv", "T", "T_adv", "M", "M_adv", "Sto", "Sto_adv", "G", "G_adv", "Gc", "N", "N_adv")}
    D["SxT"] = D["S"] * D["T"]; D["SxT_adv"] = D["S_adv"] * D["T_adv"]
    D["cal"] = off - off.mean(); la = np.log(col("esperado_adv")); D["cal_adv"] = la - la.mean()
    res = {}
    for nome, vs in ESPEC.items():
        m = np.ones(len(y), bool) if "N" not in vs else (col("temp") >= 2014)
        X = np.column_stack([np.ones(len(y)), casa] + [D[v] for v in vs])[m]
        b, Hi, mu, ll = poisson_glm(y[m], X, off[m]); V = sanduiche(X, y[m], mu, Hi, [g for g, mm in zip(grp, m) if mm])
        nomes = ["intercepto", "mando"] + vs
        res[nome] = dict(coef={k: [float(b[i]), float(np.sqrt(V[i, i]))] for i, k in enumerate(nomes)}, ll=ll,
                         aic=float(2 * len(b) - 2 * ll), n=int(m.sum()), V=V.tolist(), b=b.tolist(), nomes=nomes)
        print(f"\n{nome}   n={int(m.sum())}  logL={ll:.1f}  AIC={2 * len(b) - 2 * ll:.1f}")
        for i, k in enumerate(nomes): print(f"   {k:<11} {b[i]:+.3f}  (EP {np.sqrt(V[i, i]):.3f})  z={b[i] / np.sqrt(V[i, i]):+.2f}")
    # descritivos
    S = D["S"]
    res["descr"] = dict(n=len(obs), S_q=[float(np.quantile(S, q)) for q in (0.5, 0.9, 0.95, 0.99)], S_max=float(S.max()),
                        frac_S_maior_0_2=float((S > 0.2).mean()), frac_M=float(D["M"].mean()), n_G=int(D["G"].sum()), n_Gc=int(D["Gc"].sum()),
                        n_N=int(np.nansum(D["N"])), temporadas=[int(col("temp").min()), int(col("temp").max())],
                        dispersao=float(np.sum((y - np.exp(off)) ** 2 / np.exp(off)) / len(y)))
    print("\ndescritivos:", res["descr"])
    json.dump(res, open("resultados/historico_desespero.json", "w"), indent=1, ensure_ascii=False)

    def bloco(nome, vs):
        e = res[nome]; ii = [e["nomes"].index(v) for v in vs]
        return dict(espec=nome, nomes=vs, b=[e["b"][i] for i in ii], V=[[e["V"][i][j] for j in ii] for i in ii])
    CORE = ["S", "S_adv", "T", "T_adv", "SxT", "SxT_adv"]
    coef = dict(principal=bloco("E7: E7a + interação S×T (PRINCIPAL)", CORE + ["M", "M_adv"]),
                sem_interacao=bloco("E7a: S + T + morto + calibração", ["S", "S_adv", "T", "T_adv", "M", "M_adv"]),
                com_goleada=bloco("E8: E7 + pós-goleada", CORE + ["M", "M_adv", "G", "G_adv", "Gc"]),
                tecnico=bloco("E9: E8 + técnico novo (2014+)", ["N", "N_adv"]),
                aic={k: v["aic"] for k, v in res.items() if isinstance(v, dict) and "aic" in v and "N" not in v["nomes"]})
    json.dump(coef, open("resultados/coef_desespero.json", "w"), indent=1, ensure_ascii=False)
    print(f"concluído em {time.time() - t0:.0f}s")
