"""
gen_matches_data.py — gira match_engine.py su tutti i campionati e compatta
tutto in matches_data.js per la pagina "Partite" (uso locale, stesso pattern
del WorldCup Model: partite giocate di recente + programmate, 1X2/O-U/BTTS).
"""
import warnings; warnings.filterwarnings("ignore")
import json, time
from match_engine import run_league, LEAGUE_ESPN

NAMES = {"I1":"Serie A","E0":"Premier League","SP1":"LaLiga","D1":"Bundesliga","F1":"Ligue 1",
         "N1":"Eredivisie","P1":"Liga Portugal","B1":"Pro League","T1":"Süper Lig",
         "G1":"Super League GR","SC0":"Scottish Prem","I2":"Serie B","E1":"Championship",
         "SP2":"LaLiga 2","F2":"Ligue 2","D2":"2.Bundesliga"}

def main():
    import datetime as dt
    out = {"generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "leagues": {}}
    for div in NAMES:
        print(f"\n=== {div} ({NAMES[div]}) ===")
        try:
            r = run_league(div)
        except Exception as e:
            print(f"  ⚠ errore: {e}"); continue
        if not r: continue
        out["leagues"][div] = {
            "name": NAMES[div], "brier": r["brier"], "mbrier": r["mbrier"],
            "matches": r["matches"],
        }
        time.sleep(0.3)
    import season_engine as _se
    with open(f"{_se.SITE}/matches_data.js", "w") as f:
        f.write("window.MATCHES = " + json.dumps(out, ensure_ascii=False) + ";")
    n_tot = sum(len(v["matches"]) for v in out["leagues"].values())
    print(f"\n✓ matches_data.js — {len(out['leagues'])} campionati, {n_tot} partite totali")

if __name__=="__main__":
    main()
