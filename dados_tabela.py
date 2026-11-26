
"""
Dados do Brasileirão
Atualizado em 26/11/2026.
"""
TABELA = {
    "Flamengo":      dict(pts=80, j=36, gp=74, gc=31),
    "Palmeiras":     dict(pts=74, j=37, gp=64, gc=29),
    "Cruzeiro":      dict(pts=66, j=36, gp=59, gc=50),
    "Fluminense":    dict(pts=63, j=36, gp=57, gc=42),
    "Athletico-PR":  dict(pts=57, j=36, gp=56, gc=47),
    "Bahia":         dict(pts=54, j=36, gp=50, gc=46),
    "Atlético-MG":   dict(pts=52, j=36, gp=45, gc=41),
    "Bragantino":    dict(pts=50, j=36, gp=46, gc=41),
    "Santos":        dict(pts=50, j=36, gp=51, gc=51),
    "Vasco":         dict(pts=47, j=36, gp=47, gc=52),
    "São Paulo":     dict(pts=45, j=36, gp=40, gc=41),
    "Coritiba":      dict(pts=45, j=36, gp=41, gc=52),
    "Vitória":       dict(pts=44, j=36, gp=42, gc=54),
    "Botafogo":      dict(pts=43, j=36, gp=47, gc=58),
    "Corinthians":   dict(pts=41, j=36, gp=41, gc=43),
    "Grêmio":        dict(pts=41, j=36, gp=40, gc=47),
    "Mirassol":      dict(pts=41, j=36, gp=39, gc=49),
    "Internacional": dict(pts=38, j=36, gp=41, gc=49),
    "Remo":          dict(pts=32, j=36, gp=41, gc=61),
    "Chapecoense":   dict(pts=24, j=37, gp=37, gc=74),
}

CASA = {
    "Flamengo": (45, 37, 14),
    "Palmeiras": (43, 37, 12),
    "Fluminense": (42, 32, 17),
    "Cruzeiro": (37, 32, 19),
    "Atlético-MG": (36, 28, 19),
    "Vitória": (35, 25, 17),
    "Athletico-PR": (35, 32, 20),
    "São Paulo": (34, 25, 14),
    "Bragantino": (33, 30, 18),
    "Vasco": (30, 27, 25),
    "Grêmio": (30, 27, 21),
    "Bahia": (30, 26, 20),
    "Botafogo": (28, 28, 23),
    "Mirassol": (28, 24, 22),
    "Corinthians": (25, 22, 17),
    "Santos": (25, 23, 23),
    "Internacional": (23, 24, 22),
    "Coritiba": (23, 22, 22),
    "Remo": (22, 24, 22),
    "Chapecoense": (17, 26, 40),
}

FORA = {
    "Flamengo": (35, 37, 17),
    "Palmeiras": (31, 27, 17),
    "Cruzeiro": (29, 27, 31),
    "Santos": (25, 28, 28),
    "Bahia": (24, 24, 26),
    "Athletico-PR": (22, 24, 27),
    "Coritiba": (22, 19, 30),
    "Fluminense": (21, 25, 25),
    "Vasco": (17, 20, 27),
    "Bragantino": (17, 16, 23),
    "Atlético-MG": (16, 17, 22),
    "Corinthians": (16, 19, 26),
    "Botafogo": (15, 19, 35),
    "Internacional": (15, 17, 27),
    "Mirassol": (13, 15, 27),
    "São Paulo": (11, 15, 27),
    "Grêmio": (11, 13, 26),
    "Remo": (10, 17, 39),
    "Vitória": (9, 17, 37),
    "Chapecoense": (7, 11, 34),
}

ULT10 = {
    "Flamengo": (26, 23, 10),
    "Cruzeiro": (24, 21, 13),
    "Vasco": (19, 18, 11),
    "Fluminense": (18, 17, 10),
    "Palmeiras": (18, 17, 8),
    "Santos": (15, 12, 12),
    "Bragantino": (14, 14, 12),
    "Internacional": (13, 13, 15),
    "Mirassol": (13, 10, 9),
    "Grêmio": (13, 12, 12),
    "Atlético-MG": (13, 10, 10),
    "Vitória": (12, 17, 17),
    "Athletico-PR": (12, 18, 19),
    "São Paulo": (12, 9, 11),
    "Bahia": (11, 10, 14),
    "Botafogo": (11, 9, 17),
    "Corinthians": (9, 14, 16),
    "Remo": (9, 11, 18),
    "Coritiba": (8, 7, 17),
    "Chapecoense": (6, 8, 21),
}

# Jogos restantes a partir de 26/11/2026
JOGOS = [
    # 37ª rodada
    ("São Paulo", "Remo"),               # 26/11
    ("Fluminense", "Cruzeiro"),          # 27/11
    ("Botafogo", "Bahia"),               # 28/11
    ("Coritiba", "Flamengo"),            # 29/11
    ("Vitória", "Santos"),               # 29/11
    ("Corinthians", "Grêmio"),           # 29/11
    ("Mirassol", "Athletico-PR"),        # 29/11
    ("Internacional", "Bragantino"),     # 29/11
    ("Atlético-MG", "Vasco"),            # 29/11

    # 38ª rodada - 02/12, todos às 21h30
    ("Flamengo", "Chapecoense"),
    ("Vasco", "Vitória"),
    ("Santos", "Botafogo"),
    ("Palmeiras", "Coritiba"),
    ("Bragantino", "Fluminense"),
    ("Cruzeiro", "Internacional"),
    ("Grêmio", "Mirassol"),
    ("Athletico-PR", "São Paulo"),
    ("Bahia", "Atlético-MG"),
    ("Remo", "Corinthians"),
]

ALVO = [
    "Internacional",
    "Corinthians",
    "Grêmio",
    "Botafogo",
    "Mirassol",
]

VITORIAS = {
    "Flamengo": 24,
    "Palmeiras": 21,
    "Cruzeiro": 20,
    "Fluminense": 18,
    "Athletico-PR": 16,
    "Bahia": 14,
    "Atlético-MG": 14,
    "Bragantino": 14,
    "Santos": 13,
    "Vasco": 13,
    "São Paulo": 12,
    "Coritiba": 12,
    "Vitória": 12,
    "Botafogo": 11,
    "Corinthians": 10,
    "Grêmio": 10,
    "Mirassol": 10,
    "Internacional": 9,
    "Remo": 7,
    "Chapecoense": 5,
}