"""
═══════════════════════════════════════════════════════════════════════════
  DB DINAMICO — aggiorna euro_data.db dai feed live di football-data.co.uk
═══════════════════════════════════════════════════════════════════════════

Rende il database vivo: scarica i risultati/quote aggiornati e li scrive nel DB,
in modo IDEMPOTENTE (chiave naturale season+div+home+away):
  - partita non presente            → INSERT
  - partita presente (ora giocata)  → UPDATE dei campi risultato/quote

Per ora: campionati MAIN (formato mmz4281) — tutti i principali europei.
Le "new leagues" (extra-EU, formato diverso) in uno step successivo.

Uso:
  python update_database.py 2025-2026            # dry-run, tutti i main
  python update_database.py 2025-2026 --commit   # scrive davvero
  python update_database.py 2025-2026 I1 --commit # solo Serie A
"""

import os
import warnings; warnings.filterwarnings("ignore")
import sys, io
import sqlite3
import urllib.request
import pandas as pd

DB = os.environ.get("EURO_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "core", "euro_core.db"))

MAIN_LEAGUES = ["E0", "E1", "E2", "E3", "EC", "SP1", "SP2", "I1", "I2",
                "F1", "F2", "D1", "D2", "N1", "B1", "P1", "T1", "G1",
                "SC0", "SC1", "SC2", "SC3"]

# CSV football-data → colonna DB (campi core; le quote via normalizzazione)
CORE_MAP = {
    "HomeTeam": "home_team", "AwayTeam": "away_team",
    "FTHG": "fthg", "FTAG": "ftag", "FTR": "ftr",
    "HTHG": "hthg", "HTAG": "htag", "HTR": "htr", "Referee": "referee",
    "HS": "home_shots", "AS": "away_shots", "HST": "hst", "AST": "ast",
    "HF": "hf", "AF": "af", "HC": "hc", "AC": "ac",
    "HY": "hy", "AY": "ay", "HR": "hr", "AR": "ar",
}

def norm_col(c):
    """Normalizza un nome colonna CSV verso lo stile DB (per le quote)."""
    if c in CORE_MAP:
        return CORE_MAP[c]
    return (c.lower().replace(">2.5", "_gt_2_5").replace("<2.5", "_lt_2_5"))

def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="replace")

def db_columns():
    conn = sqlite3.connect(DB)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(matches)").fetchall()}
    conn.close()
    return cols

def update_league(season, div, db_cols, commit):
    yy = season[2:4] + season[-2:]
    try:
        csv = fetch(f"https://www.football-data.co.uk/mmz4281/{yy}/{div}.csv")
        df = pd.read_csv(io.StringIO(csv))
    except Exception as e:
        print(f"  {div}: ⚠ scaricamento fallito ({e})"); return (0, 0, 0)

    df = df.dropna(subset=["HomeTeam", "AwayTeam"]).copy()
    # data ISO
    df["date_iso"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["date_iso"])
    df["date_iso"] = df["date_iso"].dt.strftime("%Y-%m-%d")

    # colonne CSV mappabili che esistono nel DB
    colmap = {c: norm_col(c) for c in df.columns if c not in ("Div", "Date", "date_iso")}
    colmap = {c: t for c, t in colmap.items() if t in db_cols}

    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    # partite già presenti per (season, div)
    # chiave = (casa, trasferta) MA la stessa coppia puo' incontrarsi piu' volte nella stagione
    # (Scozia: 3 sfide prima dello split). existing[(h,a)] = lista di (data, ha_risultato).
    existing = {}
    for h, a, d, g in cur.execute(
            "SELECT home_team, away_team, date, fthg FROM matches WHERE season=? AND div=?", (season, div)):
        existing.setdefault((h, a), []).append((d, g is not None))

    n_ins = n_upd = 0
    for _, r in df.iterrows():
        key = (r["HomeTeam"], r["AwayTeam"])
        payload = {t: (None if pd.isna(r[c]) else r[c]) for c, t in colmap.items()}
        payload["season"] = season; payload["div"] = div; payload["date"] = r["date_iso"]

        same = [e for e in existing.get(key, []) if e[0] == r["date_iso"]]          # stessa data: e' lei
        open_ = [e for e in existing.get(key, []) if not e[1]]                       # fixture senza risultato (data spostata)
        rematch = bool(existing.get(key)) and not same and not open_ and (r.get("FTHG") is not None and not pd.isna(r.get("FTHG")))
        if (same or open_) and not rematch:
            # aggiorna solo i campi valorizzati dal CSV (mai sovrascrivere con NULL:
            # protegge le colonne calcolate del DB — elo, prob, xg, rating, ecc.)
            upd = {k: v for k, v in payload.items()
                   if k not in ("season", "div") and v is not None}
            if upd:
                sets = ", ".join(f"{k}=?" for k in upd)
                if commit:
                    tgt = (same or open_)[0][0]
                    cur.execute(f"UPDATE matches SET {sets} WHERE season=? AND div=? AND home_team=? AND away_team=? AND date=?",
                                list(upd.values()) + [season, div, key[0], key[1], tgt])
            n_upd += 1
        else:
            keys = list(payload.keys())
            if commit:
                cur.execute(f"INSERT INTO matches ({','.join(keys)}) VALUES ({','.join('?'*len(keys))})",
                            [payload[k] for k in keys])
            existing.setdefault(key, []).append((r["date_iso"], payload.get("fthg") is not None))
            n_ins += 1
    if commit:
        conn.commit()
    conn.close()
    return (len(df), n_ins, n_upd)

if __name__ == "__main__":
    args = sys.argv[1:]
    commit = "--commit" in args
    args = [a for a in args if a != "--commit"]
    season = args[0] if args else "2025-2026"
    leagues = [args[1]] if len(args) > 1 else MAIN_LEAGUES

    mode = "SCRITTURA" if commit else "DRY-RUN (nessuna modifica)"
    print(f"▶ Aggiornamento DB — stagione {season} — {mode}\n")
    db_cols = db_columns()
    tot_csv = tot_ins = tot_upd = 0
    for div in leagues:
        n_csv, n_ins, n_upd = update_league(season, div, db_cols, commit)
        if n_csv:
            print(f"  {div:4}: {n_csv:3} partite CSV  →  {n_ins:3} nuove, {n_upd:3} aggiornate")
        tot_csv += n_csv; tot_ins += n_ins; tot_upd += n_upd
    print(f"\nTotale: {tot_csv} partite — {tot_ins} nuove, {tot_upd} aggiornate")
    if not commit:
        print("(dry-run: rilancia con --commit per scrivere)")
