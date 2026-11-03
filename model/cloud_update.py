"""
cloud_update.py | aggiornamento completo Antepost, pensato per GitHub Actions (ma gira anche in locale).

Il database da 411 MB non va su Git: qui si ricostruisce ogni volta un DB compatto
  data/history_core.parquet (storico fisso, 241k partite)  +  CSV freschi di football-data (stagione corrente)
poi, in ordine:
  1) update_database  : risultati nuovi (football-data, ritardo ~1 giorno)
  2) build_elo        : ELO interno ricalcolato su tutta la storia (stesso codice del locale)
  3) update_values    : valori rosa Transfermarkt, 1 volta a settimana (o --valori). Se il sito blocca
                        o i numeri sono sospetti, si TIENE il file vecchio (mai valori rotti in produzione)
  4) gen_season_data  : simulazione 16 campionati -> site/season_data.js   (si ferma se i controlli falliscono)
  5) gen_matches_data : previsioni partita per partita -> site/matches_data.js

Uso:  python cloud_update.py [--valori] [--no-matches]
"""
import os, sys, sqlite3, shutil, subprocess, datetime as dt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
CORE = os.path.join(HERE, "core"); DBF = os.path.join(CORE, "euro_core.db")
os.environ["EURO_DB"] = DBF
SEASON = "2026-2027"
PY = sys.executable

SCHEMA = """CREATE TABLE matches (id INTEGER PRIMARY KEY AUTOINCREMENT, season TEXT, div TEXT, date TEXT,
  time TEXT, home_team TEXT, away_team TEXT, fthg INTEGER, ftag INTEGER, ftr TEXT, home_elo REAL,
  psh REAL, psd REAL, psa REAL, elo_home_calc REAL, elo_away_calc REAL);
CREATE INDEX ix_m ON matches(season, div); CREATE INDEX ix_d ON matches(div, date);"""

def run(*args, must=True):
    print("▶", " ".join(args), flush=True)
    r = subprocess.run([PY, *args])
    if r.returncode != 0 and must:
        sys.exit(f"✗ {' '.join(args)} fallito (codice {r.returncode}): NIENTE viene pubblicato")
    return r.returncode == 0

def bootstrap():
    shutil.rmtree(CORE, ignore_errors=True); os.makedirs(CORE)
    conn = sqlite3.connect(DBF); conn.executescript(SCHEMA)
    h = pd.read_parquet(f"{HERE}/data/history_core.parquet")
    h.to_sql("matches", conn, if_exists="append", index=False, chunksize=50000)
    n = conn.execute("select count(*) from matches").fetchone()[0]
    conn.commit(); conn.close()
    print(f"✓ DB compatto ricostruito: {n:,} partite storiche", flush=True)

def values_ok(path_new, path_old):
    """Accetta i nuovi valori solo se completi e plausibili rispetto ai vecchi."""
    new = pd.read_parquet(path_new); old = pd.read_parquet(path_old)
    n = new[new.season == SEASON]; o = old[old.season == SEASON]
    if len(n) < 0.9 * len(o): return f"solo {len(n)} valori contro {len(o)}"
    if (n.market_value_mln < 1).any(): return "valori < 1m (parsing sospetto)"
    m = n.merge(o, on=["div", "team"], suffixes=("_n", "_o"))
    r = (m.market_value_mln_n / m.market_value_mln_o)
    if len(m) and ((r > 3) | (r < 0.33)).mean() > 0.05: return "variazioni > 3x su troppe squadre"
    return None

def main():
    force_values = "--valori" in sys.argv
    bootstrap()
    run("update_database.py", SEASON, "--commit")
    run("build_elo.py", "--commit")

    mv = f"{HERE}/data/market_values_all.parquet"; backup = mv + ".bak"
    if force_values or dt.date.today().weekday() == 0:      # lunedi'
        shutil.copy(mv, backup)
        ok = run("update_values.py", must=False)
        why = None if ok else "scraping fallito"
        if ok:
            why = values_ok(mv, backup)
        if why:
            print(f"⚠ valori rosa NON aggiornati ({why}): tengo quelli precedenti", flush=True)
            shutil.copy(backup, mv)
        os.path.exists(backup) and os.remove(backup)

    run("gen_season_data.py")
    if "--no-matches" not in sys.argv:
        run("gen_matches_data.py", must=False)    # pagina partite: se fallisce resta l'ultima versione
    shutil.rmtree(CORE, ignore_errors=True)
    print("✓ aggiornamento completato", dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))

if __name__ == "__main__":
    main()
