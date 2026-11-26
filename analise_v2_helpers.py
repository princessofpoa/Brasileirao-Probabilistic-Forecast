import json, numpy as np
from simular_v2 import Cfg
NBT=220
def cfgs_bma(beta=0.5, **ov):
    L = json.load(open("backtest3_config.json")); tot = np.array([c["logscore"] * NBT for c in L])
    w = np.exp(beta * (tot - tot.max())); w /= w.sum()
    return [(Cfg(sigma=c["sigma"], tau=c["tau"], rho=c["rho"], nu=c["nu"], **ov), ww) for c, ww in zip(L, w) if ww > 2e-4]

