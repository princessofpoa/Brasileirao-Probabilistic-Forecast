"""Distribuições de gols: Poisson / Conway-Maxwell-Poisson (subdispersão) com média fixa,
dependência de Dixon-Coles e inflação da diagonal (empates)."""
import numpy as np
from scipy.special import gammaln
NG = 13
_XS = np.arange(NG); _XL = np.arange(60); _C = {}

def _mean_com(loglam, nu):
    lw = _XL * loglam[..., None] - nu * gammaln(_XL + 1)
    lw -= lw.max(-1, keepdims=True); w = np.exp(lw)
    return (w * _XL).sum(-1) / w.sum(-1)

def _tabela_lam(nu):
    if nu not in _C:
        gm = np.linspace(0.03, 6.0, 600); lo = np.full_like(gm, -9.0); hi = np.full_like(gm, 9.0)
        for _ in range(60):
            mid = (lo + hi) / 2; mm = _mean_com(mid, nu)
            hi = np.where(mm > gm, mid, hi); lo = np.where(mm <= gm, mid, lo)
        _C[nu] = (gm, (lo + hi) / 2)
    return _C[nu]

def pmf_marg(m, nu=1.0):
    """pmf (..., NG) com média m."""
    m = np.asarray(m, float)
    if nu == 1.0:
        return np.exp(-m[..., None] + _XS * np.log(m[..., None]) - gammaln(_XS + 1))
    gm, ll = _tabela_lam(nu); l = np.interp(m, gm, ll)
    lz = np.log(np.exp(_XL * l[..., None] - nu * gammaln(_XL + 1)).sum(-1))
    return np.exp(_XS * l[..., None] - nu * gammaln(_XS + 1) - lz[..., None])

def peso_conjunto(lh, la, x, y, rho, kappa):
    """Fator multiplicativo de dependência para o placar (x,y): Dixon-Coles x inflação da diagonal."""
    w = np.ones(np.broadcast(lh, x).shape)
    w = np.where((x == 0) & (y == 0), 1 - lh * la * rho, w)
    w = np.where((x == 0) & (y == 1), 1 + lh * rho, w)
    w = np.where((x == 1) & (y == 0), 1 + la * rho, w)
    w = np.where((x == 1) & (y == 1), 1 - rho, w)
    return w * np.where(x == y, 1 + kappa, 1.0)

def pmf_conjunta(lh, la, rho=0.0, kappa=0.0, nu=1.0):
    px = pmf_marg(lh, nu); py = pmf_marg(la, nu)
    P = px[..., :, None] * py[..., None, :]
    if rho != 0:
        P[..., 0, 0] *= 1 - lh * la * rho; P[..., 0, 1] *= 1 + lh * rho
        P[..., 1, 0] *= 1 + la * rho;      P[..., 1, 1] *= 1 - rho
    if kappa != 0:
        d = np.arange(NG); P[..., d, d] *= 1 + kappa
    return P / P.sum((-1, -2), keepdims=True)
