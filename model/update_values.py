"""
═══════════════════════════════════════════════════════════════════════════
  AGGIORNA VALORI DI MERCATO — valori CORRENTI da Transfermarkt
═══════════════════════════════════════════════════════════════════════════

Durante il calciomercato i valori rosa cambiano di continuo (acquisti/cessioni).
Questo script scarica i valori ATTUALI (pagina marktwerteverein) di ogni
campionato, li aggancia alle squadre del DB e li salva come stagione "2026-2027"
in market_values_all → la dashboard usa sempre i valori freschi.

Lanciare periodicamente (es. settimanale durante il mercato):
    python update_values.py
"""
import os
import warnings; warnings.filterwarnings("ignore")
import sqlite3, time, urllib.request, difflib, unicodedata
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from scipy.optimize import linear_sum_assignment

DBFILE = os.environ.get("EURO_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "core", "euro_core.db"))
BASE = os.path.dirname(os.path.abspath(__file__))
CUR_SEASON = "2026-2027"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
from fetch_market_values import TM_ALIAS

# campionato → (codice TM, paese)
LEAGUES = {"I1":("IT1","ITA"),"E0":("GB1","ENG"),"SP1":("ES1","ESP"),"D1":("L1","GER"),
           "F1":("FR1","FRA"),"N1":("NL1","NED"),"P1":("PO1","POR"),"B1":("BE1","BEL"),
           "T1":("TR1","TUR"),"G1":("GR1","GRE"),"SC0":("SC1","SCO"),"I2":("IT2","ITA"),
           "E1":("GB2","ENG"),"SP2":("ES2","ESP"),"F2":("FR2","FRA"),"D2":("L2","GER")}
COUNTRY_DIVS = {"ITA":["I1","I2"],"ENG":["E0","E1","E2","E3","EC"],"ESP":["SP1","SP2"],"GER":["D1","D2"],
                "FRA":["F1","F2"],"NED":["N1"],"POR":["P1"],"BEL":["B1"],"TUR":["T1"],"GRE":["G1"],
                "SCO":["SC0","SC1","SC2","SC3"]}

def _na(s): return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
def _sim(a,b):
    al,bl=_na(a.lower()),_na(b.lower())
    if al==bl: return 1.0
    if al in bl or bl in al: return .90+.10*min(len(al),len(bl))/max(len(al),len(bl))
    return difflib.SequenceMatcher(None,al,bl).ratio()
def parse_eur(s):
    # GOTCHA (nov 2026): Transfermarkt ora scrive "€738.30M" e "€1.50B" (maiuscole, "B" non "bn");
    # prima "€738.3m"/"€1.5bn". Sbagliare il suffisso azzera i valori dei club piu' ricchi.
    # Senza .lower() il suffisso non veniva riconosciuto e TUTTI i valori diventavano ~0.
    s=s.replace("€","").strip().lower(); mult=1e9 if (s.endswith("bn") or s.endswith("b")) else (1e6 if s.endswith("m") else (1e3 if s.endswith("k") else 1))
    try: return float("".join(c for c in s if (c.isdigit() or c=="."))) * mult/1e6
    except: return None

def scrape_current(tmcode):
    """Valori CORRENTI delle squadre del campionato (marktwerteverein)."""
    url=f"https://www.transfermarkt.com/x/marktwerteverein/wettbewerb/{tmcode}"
    try:
        resp=urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=25)
        html=resp.read().decode("utf-8","replace")
    except Exception as e:
        print(f"    HTTP: {e}"); return []
    t=BeautifulSoup(html,"html.parser").find("table",class_="items"); out=[]
    if not t or not t.find("tbody"):
        ttl=BeautifulSoup(html,"html.parser").title
        print(f"    tabella assente (len={len(html)}, title={ttl.get_text(strip=True)[:80] if ttl else None})"); return out
    for tr in t.find("tbody").find_all("tr",recursive=False):
        nl=tr.find("td",class_="hauptlink")
        if not nl: continue
        v=next((td.get_text(strip=True) for td in reversed(tr.find_all("td",recursive=False)) if "€" in td.get_text()),None)
        if v and parse_eur(v): out.append((nl.get_text(strip=True), parse_eur(v)))
    return out

def match_db(tm_list, cand):
    tmn=[TM_ALIAS.get(n,n) for n,_ in tm_list]
    cost=np.array([[1-_sim(t,d) for d in cand] for t in tmn])
    ri,ci=linear_sum_assignment(cost); out={}
    for r,c in zip(ri,ci):
        if 1-cost[r,c]>=.55: out[cand[c]]=tm_list[r][1]
    return out

conn=sqlite3.connect(DBFILE)
rows=[]
print("Aggiorno i valori di mercato correnti…")
for div,(tmcode,country) in LEAGUES.items():
    try:
        tm=scrape_current(tmcode); time.sleep(1.5)
    except Exception as e:
        print(f"  {div}: ⚠ {e}"); continue
    if not tm:
        print(f"  {div}: nessun valore"); continue
    cand=[r[0] for r in conn.execute(
        f"SELECT DISTINCT home_team FROM matches WHERE div IN ({','.join('?'*len(COUNTRY_DIVS[country]))}) AND season>='2022-2023'",
        COUNTRY_DIVS[country])]
    m=match_db(tm,cand)
    for team,val in m.items():
        rows.append({"season":CUR_SEASON,"div":div,"team":team,"market_value_mln":round(val,2)})
    print(f"  ✓ {div}: {len(m)}/{len(tm)} squadre, valori correnti")
conn.close()

new=pd.DataFrame(rows)
if new.empty:
    raise SystemExit("✗ nessun valore scaricato: file valori NON modificato")
mv=pd.read_parquet(f"{BASE}/data/market_values_all.parquet")
mv=mv[mv["season"]!=CUR_SEASON]                       # rimuovi i vecchi current
mv=pd.concat([mv,new],ignore_index=True)
mv.to_parquet(f"{BASE}/data/market_values_all.parquet",index=False)
print(f"\n✓ {len(new)} valori correnti salvati (stagione {CUR_SEASON}). Totale: {len(mv)}")
top = new[new["div"]=="I1"].sort_values("market_value_mln",ascending=False).head(3)
print("  Esempio Serie A:", top[["team","market_value_mln"]].values.tolist())

# sanity check: un valore rosa quasi nullo per un club di prima divisione e' quasi sempre un errore di parsing
_bad = new[(new["market_value_mln"] < 1.0)]
if len(_bad):
    print(f"  ⚠ ATTENZIONE: {len(_bad)} valori < 1m (probabile errore di parsing):", _bad[["div","team"]].values.tolist()[:10])
