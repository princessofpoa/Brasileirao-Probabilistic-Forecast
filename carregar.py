"""Le e valida base de partidas contra a classificação oficial."""
import re, datetime as dt
from collections import defaultdict

PADRAO = re.compile(r"^(\d+)ª \| (\d\d)/(\d\d) \| (.+?) (\d+) x (\d+) (.+)$")

def carregar(caminho="jogos_2026.txt"):
    jogos = []
    for linha in open(caminho, encoding="utf-8"):
        linha = linha.strip()
        if not linha: continue
        m = PADRAO.match(linha)
        assert m, f"linha inválida: {linha}"
        rod, d, mes, casa, gc, gv, fora = m.groups()
        data = dt.date(2026, int(mes), int(d))
        jogos.append(dict(rodada=int(rod), data=data, casa=casa, fora=fora, gc=int(gc), gv=int(gv)))
    jogos.sort(key=lambda j: (j["data"], j["rodada"]))
    return jogos

if __name__ == "__main__":
    from dados_tabela import TABELA
    J = carregar()
    print("jogos:", len(J))
    agg = defaultdict(lambda: dict(pts=0, j=0, gp=0, gc=0, v=0))
    for g in J:
        for t, gp, gs in ((g["casa"], g["gc"], g["gv"]), (g["fora"], g["gv"], g["gc"])):
            a = agg[t]; a["j"] += 1; a["gp"] += gp; a["gc"] += gs
            a["pts"] += 3 if gp > gs else 1 if gp == gs else 0; a["v"] += gp > gs
    bad = 0
    for t, ref in TABELA.items():
        a = agg[t]
        ok = (a["pts"], a["j"], a["gp"], a["gc"]) == (ref["pts"], ref["j"], ref["gp"], ref["gc"])
        if not ok:
            bad += 1; print("DIVERGE", t, dict(a), ref)
    print("clubes divergentes:", bad)
    tot_g = sum(g["gc"] + g["gv"] for g in J); print("gols/jogo:", round(tot_g / len(J), 3),
          "| mandante:", round(sum(g["gc"] for g in J) / len(J), 3), "| visitante:", round(sum(g["gv"] for g in J) / len(J), 3))
    # dispersão de gols por time-jogo
    import numpy as np
    x = np.array([g["gc"] for g in J] + [g["gv"] for g in J]); print("média, var, var/média:", x.mean().round(3), x.var().round(3), (x.var() / x.mean()).round(3))
    emp = np.mean([g["gc"] == g["gv"] for g in J]); print("empates:", round(emp, 3))
