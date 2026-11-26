"""
Modelo de força dinâmica (espaço de estados) para gols no Brasileirão.

  gols_casa ~ Poisson(exp(mu + h + eta_H[casa] + A[casa,k] - D[fora,l]))
  gols_fora ~ Poisson(exp(mu +                  A[fora,l] - D[casa,k]))

  A[i,k] = ataque do time i no seu k-ésimo jogo ;  D[i,k] = qualidade defensiva
  A[i,0] ~ N(0,tau^2),  A[i,k]-A[i,k-1] ~ N(0,sigma^2)     (passeio aleatório por jogo)
  idem para D; eta_H[i] ~ N(0,s_h^2) (mando específico, encolhido); mu, h ~ priors vagos.

Inferência: moda a posteriori (Newton-Raphson esparso) + aproximação de Laplace
(covariância = inverso da Hessiana). Hiperparâmetros escolhidos por validação walk-forward.
"""
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.special import gammaln
from dist import pmf_conjunta

NG = 13  # suporte 0..12 gols nas tabelas de placar


class Base:
    """Organiza partidas em ordem cronológica e indexa parâmetros."""
    def __init__(self, jogos, times):
        self.times = list(times); self.n = len(times)
        self.idx = {t: i for i, t in enumerate(self.times)}
        self.jogos = jogos

    def montar(self, m):
        """Índices e matriz de desenho para as m primeiras partidas."""
        J = self.jogos[:m]; n = self.n
        cont = np.zeros(n, int); kc = np.zeros(m, int); kf = np.zeros(m, int)
        ic = np.array([self.idx[g["casa"]] for g in J]); iff = np.array([self.idx[g["fora"]] for g in J])
        for r in range(m):
            kc[r] = cont[ic[r]]; cont[ic[r]] += 1
            kf[r] = cont[iff[r]]; cont[iff[r]] += 1
        offA = 2 + n + np.concatenate([[0], np.cumsum(cont)[:-1]])
        offD = 2 + n + cont.sum() + np.concatenate([[0], np.cumsum(cont)[:-1]])
        P = 2 + n + 2 * cont.sum()
        gc = np.array([g["gc"] for g in J]); gv = np.array([g["gv"] for g in J])
        rows, cols, vals = [], [], []
        def add(r, c, v): rows.append(r); cols.append(c); vals.append(v)
        for r in range(m):
            h = 2 * r; a = 2 * r + 1
            add(h, 0, 1); add(h, 1, 1); add(h, 2 + ic[r], 1)
            add(h, offA[ic[r]] + kc[r], 1); add(h, offD[iff[r]] + kf[r], -1)
            add(a, 0, 1)
            add(a, offA[iff[r]] + kf[r], 1); add(a, offD[ic[r]] + kc[r], -1)
        X = sp.csr_matrix((vals, (rows, cols)), shape=(2 * m, P))
        y = np.empty(2 * m); y[0::2] = gc; y[1::2] = gv
        return dict(X=X, y=y, P=P, cont=cont, offA=offA, offD=offD, ic=ic, iff=iff, kc=kc, kf=kf, m=m)

    def precisao(self, d, sig_a, sig_d, tau, s_h):
        n, cont, offA, offD, P = self.n, d["cont"], d["offA"], d["offD"], d["P"]
        rows, cols, vals = [], [], []; r = 0
        def row(cs, vs):
            nonlocal r
            for c, v in zip(cs, vs): rows.append(r); cols.append(c); vals.append(v)
            r += 1
        row([0], [0.2]); row([1], [0.2])                                       # priors vagos (sd=5)
        for i in range(n): row([2 + i], [1 / max(s_h, 1e-3)])                    # mando por time
        for i in range(n):
            for off, sg in ((offA[i], sig_a), (offD[i], sig_d)):
                row([off], [1 / tau])
                for k in range(1, cont[i]): row([off + k - 1, off + k], [-1 / sg, 1 / sg])
        L = sp.csr_matrix((vals, (rows, cols)), shape=(r, P))
        return (L.T @ L).tocsc()


def ajustar(base, d, sig_a, sig_d, tau=0.35, s_h=0.0, iters=40):
    X, y, P = d["X"], d["y"], d["P"]
    Q = base.precisao(d, sig_a, sig_d, tau, s_h)
    th = np.zeros(P); th[0] = np.log(1.3)
    for _ in range(iters):
        lam = np.exp(X @ th)
        g = X.T @ (y - lam) - Q @ th
        H = (X.T @ sp.diags(lam) @ X + Q).tocsc()
        lu = splu(H); dl = lu.solve(g); th += dl
        if np.abs(dl).max() < 1e-8: break
    lam = np.exp(X @ th)
    H = (X.T @ sp.diags(lam) @ X + Q).tocsc(); lu = splu(H)
    return dict(th=th, lu=lu, H=H, Q=Q, d=d, sig_a=sig_a, sig_d=sig_d, tau=tau, s_h=s_h,
                loglik=float(np.sum(y * np.log(lam) - lam - gammaln(y + 1))))


def indices_finais(base, fit):
    """Índices dos parâmetros 'vivos' (estado atual) e seu vetor/covariância posterior."""
    d = fit["d"]; n = base.n
    S = [0, 1] + [2 + i for i in range(n)]
    S += [int(d["offA"][i] + d["cont"][i] - 1) for i in range(n)]
    S += [int(d["offD"][i] + d["cont"][i] - 1) for i in range(n)]
    S = np.array(S)
    E = np.zeros((d["P"], len(S))); E[S, np.arange(len(S))] = 1
    Z = fit["lu"].solve(E); Sigma = Z[S, :]
    Sigma = (Sigma + Sigma.T) / 2
    return S, fit["th"][S], Sigma


def pmf_tabela_old(lh, la, rho):
    """P(x,y) bivariada com correção de Dixon-Coles. lh, la: (...,) -> (..., NG, NG)."""
    k = np.arange(NG)
    lg = lambda lam: (-lam[..., None] + k * np.log(lam[..., None]) - gammaln(k + 1))
    px = np.exp(lg(lh)); py = np.exp(lg(la))
    P = px[..., :, None] * py[..., None, :]
    if rho != 0:
        P[..., 0, 0] *= 1 - lh * la * rho
        P[..., 0, 1] *= 1 + lh * rho
        P[..., 1, 0] *= 1 + la * rho
        P[..., 1, 1] *= 1 - rho
    return P


def prever(base, fit, jogos_alvo, specs, ndraw=300, rng=None, passos=1):
    """pmf preditiva (placar) por partida, integrando incerteza de parâmetros e passo do passeio."""
    rng = rng or np.random.default_rng(0)
    S, th_S, Sigma = indices_finais(base, fit); n = base.n; pos = {int(s): j for j, s in enumerate(S)}
    iA = lambda i: 2 + n + i; iD = lambda i: 2 + 2 * n + i
    out = {sp_: [] for sp_ in specs}
    ev = (fit["sig_a"] ** 2 + fit["sig_d"] ** 2) * passos
    for g in jogos_alvo:
        c, f = base.idx[g["casa"]], base.idx[g["fora"]]
        U = np.zeros((2, len(S)))
        U[0, 0] = U[0, 1] = 1; U[0, 2 + c] = 1; U[0, iA(c)] = 1; U[0, iD(f)] = -1
        U[1, 0] = 1; U[1, iA(f)] = 1; U[1, iD(c)] = -1
        mu = U @ th_S; C = U @ Sigma @ U.T + np.eye(2) * ev
        Lc = np.linalg.cholesky(C + 1e-10 * np.eye(2))
        z = rng.standard_normal((ndraw, 2)) @ Lc.T + mu
        lam = np.exp(z)
        for sp_ in specs: out[sp_].append(pmf_conjunta(lam[:, 0], lam[:, 1], *sp_).mean(0))
    return {k: np.array(v) for k, v in out.items()}


def metricas(pmf, jogos_alvo):
    """log-score do placar, log-loss e RPS do resultado."""
    ls, ll, rps = [], [], []
    for P, g in zip(pmf, jogos_alvo):
        x, y = min(g["gc"], NG - 1), min(g["gv"], NG - 1)
        ls.append(np.log(max(P[x, y], 1e-12)))
        ph = np.tril(P, -1).sum(); pa = np.triu(P, 1).sum(); pd = np.trace(P)
        ob = np.array([x > y, x == y, x < y], float); pr = np.array([ph, pd, pa]); pr = pr / pr.sum()
        ll.append(-np.log(max(pr[ob.argmax()], 1e-12)))
        rps.append(0.5 * (((pr[0] - ob[0]) ** 2) + ((pr[0] + pr[1] - ob[0] - ob[1]) ** 2)))
    return np.array(ls), np.array(ll), np.array(rps)
