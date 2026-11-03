"""
═══════════════════════════════════════════════════════════════════════════
  ELO INTERNO — rating calcolato dai risultati, per TUTTI i 38 campionati
═══════════════════════════════════════════════════════════════════════════

clubelo copre solo l'Europa. Qui calcoliamo un ELO da zero, uniforme ovunque
(anche Argentina, Brasile, USA, Giappone…), processando ogni partita del DB in
ordine cronologico:

  E_casa = 1 / (1 + 10^((elo_osp − elo_casa − HFA)/400))      # attesa
  agg.   = K · G · (risultato − E_casa)                        # G = peso margine gol
  cross-stagione: regressione verso 1500 (mercato, reset parziale)

Identità squadra = (paese, nome) → niente collisioni tra campionati.
Salva elo_home_calc / elo_away_calc nel DB (pre-match), e valida vs clubelo.

Uso:  python build_elo.py            # dry-run + validazione
      python build_elo.py --commit   # scrive le colonne nel DB
"""

import os
import warnings; warnings.filterwarnings("ignore")
import sys
import sqlite3
import numpy as np
import pandas as pd

DB = os.environ.get("EURO_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "core", "euro_core.db"))

K = 20.0
HFA = 65.0          # vantaggio campo (punti ELO)
BASE = 1500.0
SEASON_REG = 0.15   # regressione verso la media a inizio stagione

COUNTRY = {
    "E0": "ENG", "E1": "ENG", "E2": "ENG", "E3": "ENG", "EC": "ENG",
    "SC0": "SCO", "SC1": "SCO", "SC2": "SCO", "SC3": "SCO",
    "I1": "ITA", "I2": "ITA", "SP1": "ESP", "SP2": "ESP",
    "F1": "FRA", "F2": "FRA", "D1": "GER", "D2": "GER",
    "N1": "NED", "B1": "BEL", "P1": "POR", "T1": "TUR", "G1": "GRE",
    "ARG1": "ARG", "USA1": "USA", "BRA1": "BRA", "MEX1": "MEX", "JPN1": "JPN",
    "ROU1": "ROU", "POL1": "POL", "NOR1": "NOR", "SWE1": "SWE", "RUS1": "RUS",
    "DNK1": "DNK", "CHN1": "CHN", "SWZ1": "SWZ", "IRL1": "IRL", "AUT1": "AUT", "FIN1": "FIN",
}

def gd_mult(gd):
    """Peso del margine di gol (stile FiveThirtyEight)."""
    if gd <= 1: return 1.0
    if gd == 2: return 1.5
    return (11 + gd) / 8.0

# ─────────────────────────────────────────────────────────────────────────
print("Carico le partite…")
conn = sqlite3.connect(DB)
df = pd.read_sql(
    "SELECT id, season, div, date, home_team, away_team, fthg, ftag, ftr, home_elo "
    "FROM matches WHERE fthg IS NOT NULL AND ftag IS NOT NULL "
    "ORDER BY date, id", conn)
conn.close()
df["country"] = df["div"].map(COUNTRY).fillna(df["div"])
print(f"Partite: {len(df):,}  |  paesi: {df['country'].nunique()}")

# ─────────────────────────────────────────────────────────────────────────
print("Calcolo ELO cronologico…")
elo = {}            # (country, team) -> rating
last_season = {}    # (country, team) -> ultima stagione vista
eh = np.empty(len(df)); ea = np.empty(len(df))

for i, (cty, seas, h, a, fthg, ftag) in enumerate(zip(
        df["country"].values, df["season"].values,
        df["home_team"].values, df["away_team"].values,
        df["fthg"].values, df["ftag"].values)):
    kh, ka = (cty, h), (cty, a)
    rh = elo.get(kh, BASE); ra = elo.get(ka, BASE)
    # regressione a inizio nuova stagione
    if last_season.get(kh) != seas:
        rh = BASE + (1 - SEASON_REG) * (rh - BASE); last_season[kh] = seas
    if last_season.get(ka) != seas:
        ra = BASE + (1 - SEASON_REG) * (ra - BASE); last_season[ka] = seas
    # pre-match (quello che serve al modello)
    eh[i] = rh; ea[i] = ra
    # aggiornamento
    exp_h = 1.0 / (1 + 10 ** ((ra - rh - HFA) / 400.0))
    res_h = 1.0 if fthg > ftag else (0.5 if fthg == ftag else 0.0)
    g = gd_mult(abs(int(fthg) - int(ftag)))
    delta = K * g * (res_h - exp_h)
    elo[kh] = rh + delta; elo[ka] = ra - delta

df["elo_home_calc"] = eh.round(2)
df["elo_away_calc"] = ea.round(2)

# ─────────────────────────────────────────────────────────────────────────
# Validazione vs clubelo (dove disponibile)
# ─────────────────────────────────────────────────────────────────────────
val = df[df["home_elo"].notna()]
corr = np.corrcoef(val["elo_home_calc"], val["home_elo"])[0, 1]
print(f"\n── Validazione vs clubelo ({len(val):,} partite europee) ──")
print(f"  Correlazione ELO interno vs clubelo: {corr:.3f}")
# top squadre attuali per ELO interno (sanity check Serie A)
print("\n  Top 6 ELO interno per Serie A (ultima partita di ogni squadra):")
ita = df[df["div"] == "I1"].copy()
last = pd.concat([
    ita[["home_team", "elo_home_calc", "date"]].rename(columns={"home_team": "team", "elo_home_calc": "elo"}),
    ita[["away_team", "elo_away_calc", "date"]].rename(columns={"away_team": "team", "elo_away_calc": "elo"}),
]).sort_values("date").groupby("team").tail(1).sort_values("elo", ascending=False)
print(last.head(6)[["team", "elo"]].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────
if "--commit" in sys.argv:
    print("\nScrivo elo_home_calc / elo_away_calc nel DB…")
    conn = sqlite3.connect(DB)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(matches)").fetchall()}
    for c in ("elo_home_calc", "elo_away_calc"):
        if c not in cols:
            conn.execute(f"ALTER TABLE matches ADD COLUMN {c} REAL")
    conn.executemany("UPDATE matches SET elo_home_calc=?, elo_away_calc=? WHERE id=?",
                     list(zip(df["elo_home_calc"], df["elo_away_calc"], df["id"])))
    conn.commit(); conn.close()
    print(f"✓ Scritte {len(df):,} righe.")
else:
    print("\n(dry-run: rilancia con --commit per scrivere nel DB)")
