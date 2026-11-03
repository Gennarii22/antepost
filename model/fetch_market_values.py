"""
═══════════════════════════════════════════════════════════════════════════
  VALORE ROSA — scraping valori di mercato Transfermarkt (storico per stagione)
═══════════════════════════════════════════════════════════════════════════

Feature economica universale (sostituisce/affianca gli stipendi). Per ogni
campionato e stagione scarica il valore totale rosa di ogni squadra da
Transfermarkt, fa il match dei nomi con le squadre del DB e salva.

Robustezza: User-Agent browser, pausa tra richieste, match nomi fuzzy.

Uso:  python fetch_market_values.py I1                 # Serie A, stagioni del modello
      python fetch_market_values.py I1 2015 2025       # range stagioni
"""

import warnings; warnings.filterwarnings("ignore")
import sys, io, re, time, sqlite3, unicodedata
import urllib.request
import difflib
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from scipy.optimize import linear_sum_assignment

def _na(s):
    """Rimuove gli accenti (Alavés → Alaves)."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

DB = "/Users/gennaropancia/Desktop/1 PROJECTS/Football Betting Model/Antepost"  # output
DBFILE = "/Users/gennaropancia/Desktop/1 PROJECTS/Football Betting Model/Database/euro_data.db"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
PAUSE = 2.5   # secondi tra richieste (cortesia)

# div DB → codice competizione Transfermarkt
DIV_TO_TM = {
    "I1": "IT1", "I2": "IT2", "E0": "GB1", "E1": "GB2", "SP1": "ES1", "SP2": "ES2",
    "F1": "FR1", "F2": "FR2", "D1": "L1", "D2": "L2", "N1": "NL1", "B1": "BE1",
    "P1": "PO1", "T1": "TR1", "G1": "GR1", "SC0": "SC1",
}

# alias espliciti per nomi Transfermarkt che contengono altri nomi squadra
# (vanno risolti PRIMA del match fuzzy, altrimenti la sottostringa sbaglia)
TM_ALIAS = {
    # Italia
    "Inter Milan": "Inter", "AC Milan": "Milan",
    "Hellas Verona": "Verona", "Chievo Verona": "Chievo", "AC ChievoVerona": "Chievo",
    # Inghilterra
    "Wolverhampton Wanderers": "Wolves", "Tottenham Hotspur": "Tottenham",
    "Manchester City": "Man City", "Manchester United": "Man United",
    "Newcastle United": "Newcastle", "Nottingham Forest": "Nott'm Forest",
    "Sheffield United": "Sheffield United", "West Bromwich Albion": "West Brom",
    "Queens Park Rangers": "QPR", "Brighton & Hove Albion": "Brighton",
    # Spagna
    "Espanyol": "Espanol", "Deportivo Alavés": "Alaves", "Deportivo Alaves": "Alaves",
    "CA Osasuna": "Osasuna", "Real Betis Balompié": "Betis", "Real Betis": "Betis",
    "Athletic Bilbao": "Ath Bilbao", "Atlético de Madrid": "Ath Madrid", "Atletico Madrid": "Ath Madrid",
    "Cádiz CF": "Cadiz", "Sporting Gijón": "Sp Gijon", "Real Sociedad": "Sociedad",
    # Germania
    "Borussia Mönchengladbach": "M'gladbach", "Bayern Munich": "Bayern Munich",
    "Bayer 04 Leverkusen": "Leverkusen", "Borussia Dortmund": "Dortmund",
    "Eintracht Frankfurt": "Ein Frankfurt", "1.FC Köln": "FC Koln", "1.FC Koln": "FC Koln",
    "TSG 1899 Hoffenheim": "Hoffenheim", "VfL Wolfsburg": "Wolfsburg",
    "FC Schalke 04": "Schalke 04", "Bayer Leverkusen": "Leverkusen",
    # Francia
    "Paris Saint-Germain": "Paris SG", "Olympique Marseille": "Marseille",
    "Olympique Lyon": "Lyon", "AS Saint-Étienne": "St Etienne", "AS Monaco": "Monaco",
    "Stade Rennais FC": "Rennes", "OGC Nice": "Nice", "LOSC Lille": "Lille",
    # Portogallo
    "Sporting CP": "Sp Lisbon", "Estrela Amadora": "Estrela",
    "Belenenses SAD": "Belenenses", "CF Os Belenenses": "Belenenses",
    "Vitória Guimarães SC": "Guimaraes", "Vitória Setúbal": "Setubal",
    # Turchia
    "Basaksehir FK": "Buyuksehyr", "Istanbul Basaksehir": "Buyuksehyr",
    "Gaziantep FK": "Gaziantepspor", "Erzurumspor FK": "Erzurum BB", "BB Erzurumspor": "Erzurum BB",
    # Scozia
    "Heart of Midlothian FC": "Hearts", "Heart of Midlothian": "Hearts",
    # Spagna (extra)
    "RCD Espanyol": "Espanol", "RCD Espanyol Barcelona": "Espanol",
}

def parse_eur(s):
    s = s.replace("€", "").strip()
    mult = 1e9 if s.endswith("bn") else (1e6 if s.endswith("m") else (1e3 if s.endswith("k") else 1))
    num = re.sub(r"[a-zA-Z]", "", s)
    try: return float(num) * mult / 1e6   # milioni
    except: return None

def scrape(tm_code, saison):
    url = f"https://www.transfermarkt.com/x/startseite/wettbewerb/{tm_code}/plus/?saison_id={saison}"
    html = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25).read().decode("utf-8", "replace")
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="items")
    if not table or not table.find("tbody"):
        return []
    out = []
    for tr in table.find("tbody").find_all("tr", recursive=False):
        tds = tr.find_all("td", recursive=False)
        if len(tds) < 3: continue
        nlink = tr.find("td", class_="hauptlink")
        name = nlink.get_text(strip=True) if nlink else None
        val = next((td.get_text(strip=True) for td in reversed(tds) if "€" in td.get_text()), None)
        if name and val:
            out.append((name, parse_eur(val)))
    return out

def db_teams(div, season):
    conn = sqlite3.connect(DBFILE)
    teams = [r[0] for r in conn.execute(
        "SELECT DISTINCT home_team FROM matches WHERE div=? AND season=?", (div, season))]
    conn.close()
    return teams

def _sim(a, b):
    """Similarità tra due nomi: alta se uno è sottostringa dell'altro."""
    al, bl = _na(a.lower()), _na(b.lower())
    if al == bl: return 1.0
    if al in bl or bl in al:
        return 0.90 + 0.10 * min(len(al), len(bl)) / max(len(al), len(bl))
    # bonus se condividono una parola significativa (>3 lettere)
    wa = {w for w in re.split(r"\W+", al) if len(w) > 3}
    wb = {w for w in re.split(r"\W+", bl) if len(w) > 3}
    base = difflib.SequenceMatcher(None, al, bl).ratio()
    return max(base, 0.85) if (wa & wb) else base

def match_names(tm_list, db_teams_):
    """Accoppiamento OTTIMALE globale (Hungarian): risolve da solo le ambiguità
    tipo 'Inter Milan'/'AC Milan' massimizzando la corrispondenza complessiva."""
    if not tm_list or not db_teams_:
        return {}
    # applica gli alias noti (rinomina i nomi TM prima del match)
    tm_norm = [(TM_ALIAS.get(n, n), v) for n, v in tm_list]
    tm_names = [n for n, _ in tm_norm]
    # matrice di costo = 1 − similarità
    cost = np.array([[1 - _sim(t, d) for d in db_teams_] for t in tm_names])
    ri, ci = linear_sum_assignment(cost)
    out = {}
    for r, c in zip(ri, ci):
        if (1 - cost[r, c]) >= 0.55:
            out[db_teams_[c]] = tm_norm[r][1]
    return out

if __name__ == "__main__":
    div = sys.argv[1] if len(sys.argv) > 1 else "I1"
    s0 = int(sys.argv[2]) if len(sys.argv) > 2 else 2015
    s1 = int(sys.argv[3]) if len(sys.argv) > 3 else 2025
    tm = DIV_TO_TM.get(div)
    if not tm:
        print(f"Codice Transfermarkt non mappato per {div}"); sys.exit(1)

    rows = []
    for yr in range(s0, s1 + 1):
        season = f"{yr}-{yr+1}"
        try:
            tm_list = scrape(tm, yr)
        except Exception as e:
            print(f"  {season}: ⚠ {e}"); continue
        dbt = db_teams(div, season)
        matched = match_names(tm_list, dbt)
        unmatched_db = [t for t in dbt if t not in matched]
        print(f"  {season}: {len(tm_list)} su TM, {len(matched)}/{len(dbt)} agganciate"
              + (f"  ⚠ non agganciate: {unmatched_db}" if unmatched_db else ""))
        for team, val in matched.items():
            rows.append({"season": season, "div": div, "team": team, "market_value_mln": val})
        time.sleep(PAUSE)

    mv = pd.DataFrame(rows)
    out_path = f"{DB}/data/market_values_{div}.parquet"
    if len(mv):
        mv.to_parquet(out_path, index=False)
        print(f"\n✓ Salvati {len(mv)} valori in data/market_values_{div}.parquet")
        # validazione: confronto con stipendi (Serie A) sull'ultima stagione
        print(f"\nEsempio {div} 2024-2025 (valore rosa):")
        ex = mv[mv.season == "2024-2025"].sort_values("market_value_mln", ascending=False).head(6)
        print(ex[["team", "market_value_mln"]].to_string(index=False))
