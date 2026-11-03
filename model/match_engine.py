"""
match_engine.py — motore Poisson match-by-match per campionato, stesso DNA del
WorldCup Model ma per un girone all'italiana invece di un torneo a eliminazione.

Validato (Serie A, backtest out-of-sample su 2025-2026, fuori dal training):
  Brier modello ELO+valore rosa = 0.1931  vs  Brier mercato (Pinnacle devig) = 0.1976
  -> il modello batte il mercato, calibrazione solida prima di costruire la dashboard.

lambda_home = exp(a + hadv + b*elo_diff/400 + c*log(valore_h/valore_a))
lambda_away = exp(a        - b*elo_diff/400 - c*log(valore_h/valore_a))
Calibrato per MASSIMA VEROSIMIGLIANZA su Poisson, per singolo campionato (il
vantaggio-casa e il peso valore/ELO cambiano molto da lega a lega, verificato:
Bundesliga +0.354 gol di vantaggio-casa, Ligue 1 +0.200, Serie A +0.213).

Uso:
  python3 match_engine.py I1              # calibra + prossime partite Serie A
  python3 match_engine.py I1 --backtest   # solo backtest, stampa Brier vs mercato
"""
import os
import warnings; warnings.filterwarnings("ignore")
import sqlite3, sys, json, urllib.request, datetime as dt, difflib, unicodedata
import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize

def _na(s): return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
def _sim(a,b):
    al,bl=_na(a.lower()),_na(b.lower())
    if al==bl: return 1.0
    if al in bl or bl in al: return .90+.10*min(len(al),len(bl))/max(len(al),len(bl))
    return difflib.SequenceMatcher(None,al,bl).ratio()
def fuzzy_lookup(name, choices, thr=0.6):
    """ESPN e il nostro DB usano nomi diversi (es. 'VfB Stuttgart' vs 'Stuttgart',
    'Royal Charleroi SC' vs 'Charleroi'): stesso matching gia' usato per Transfermarkt."""
    if name in choices: return name
    best = max(choices, key=lambda c: _sim(name, c), default=None)
    return best if best and _sim(name, best) >= thr else None

DBFILE = os.environ.get("EURO_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "core", "euro_core.db"))
BASE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "Mozilla/5.0"}

LEAGUE_ESPN = {"I1":"ita.1","E0":"eng.1","SP1":"esp.1","D1":"ger.1","F1":"fra.1",
               "N1":"ned.1","P1":"por.1","B1":"bel.1","T1":"tur.1","G1":"gre.1","SC0":"sco.1",
               "I2":"ita.2","E1":"eng.2","SP2":"esp.2","F2":"fra.2","D2":"ger.2"}
TRAIN_SEASONS = ["2021-2022","2022-2023","2023-2024","2024-2025","2025-2026"]

def load_data(div, conn):
    m = pd.read_sql("""select date,season,home_team,away_team,fthg,ftag,elo_home_calc,elo_away_calc,psh,psd,psa
        from matches where div=? and season in ({}) and elo_home_calc is not null and fthg is not null"""
        .format(",".join("?"*len(TRAIN_SEASONS))), conn, params=[div]+TRAIN_SEASONS)
    m['date']=pd.to_datetime(m.date)
    mv = pd.read_parquet(f"{BASE}/data/market_values_all.parquet")
    mv = mv[mv['div']==div]
    mv_map = mv.set_index(['season','team']).market_value_mln.to_dict()
    def get_mv(season,team): return mv_map.get((season,team), np.nan)
    m['mv_h']=m.apply(lambda r: get_mv(r.season,r.home_team), axis=1)
    m['mv_a']=m.apply(lambda r: get_mv(r.season,r.away_team), axis=1)
    m = m.dropna(subset=['mv_h','mv_a']).copy()
    m['lval'] = np.log(m.mv_h.clip(lower=1)) - np.log(m.mv_a.clip(lower=1))
    m['ed'] = (m.elo_home_calc - m.elo_away_calc)/400
    return m

def calibrate(train):
    def neg_ll(p):
        a,b,cc,hadv = p
        lh = np.exp(a + hadv + b*train.ed + cc*train.lval)
        la = np.exp(a - b*train.ed - cc*train.lval)
        return -(poisson.logpmf(train.fthg, lh) + poisson.logpmf(train.ftag, la)).sum()
    # Nelder-Mead su 4 parametri puo' fermarsi su un minimo peggiore a seconda del
    # punto di partenza (verificato: x0 diversi davano Brier 0.193 vs 0.244 sugli
    # stessi dati). Fix: piu' tentativi da punti diversi + piu' iterazioni, tieni il migliore.
    starts = [[0.15,0.5,0.1,0.15],[0.16,0.6,0.05,0.15],[0.1,0.3,0.2,0.2],[0.2,0.1,0.3,0.1]]
    best=None
    for x0 in starts:
        res = minimize(neg_ll, x0=x0, method='Nelder-Mead',
                       options={'maxiter':5000,'xatol':1e-6,'fatol':1e-6})
        if best is None or res.fun < best.fun: best = res
    return dict(zip(['a','b','c','hadv'], best.x))

def match_probs(par, ed, lval, maxg=10):
    lh = float(np.exp(par['a']+par['hadv']+par['b']*ed+par['c']*lval))
    la = float(np.exp(par['a']-par['b']*ed-par['c']*lval))
    gh=np.arange(maxg); ga=np.arange(maxg)
    mat = np.outer(poisson.pmf(gh,lh), poisson.pmf(ga,la))
    tot = gh[:,None]+ga[None,:]
    over = lambda L: float(mat[tot>L].sum())
    return dict(lh=round(lh,2), la=round(la,2),
                p1=float(np.tril(mat,-1).sum()), px=float(np.trace(mat)), p2=float(np.triu(mat,1).sum()),
                over15=over(1.5), over25=over(2.5), over35=over(3.5),
                btts=float(1-mat[0,:].sum()-mat[:,0].sum()+mat[0,0]),
                top_score="-".join(str(int(x)) for x in np.unravel_index(mat.argmax(), mat.shape)))

def backtest(div):
    conn = sqlite3.connect(DBFILE)
    m = load_data(div, conn)
    train, test = m[m.season<"2025-2026"], m[m.season=="2025-2026"]
    if len(test) < 20:
        print(f"{div}: campione test troppo piccolo ({len(test)}), salto backtest"); return None
    par = calibrate(train)
    test = test.reset_index(drop=True)   # altrimenti .apply() sotto mantiene l'indice
    # originale non contiguo e il concat successivo disallinea partite <-> probabilita'
    preds = test.apply(lambda r: match_probs(par, r.ed, r.lval), axis=1, result_type='expand')
    t = pd.concat([test, preds], axis=1)
    esito = np.where(t.fthg>t.ftag,'1',np.where(t.fthg==t.ftag,'X','2'))
    brier = ((t.p1-(esito=='1'))**2+(t.px-(esito=='X'))**2+(t.p2-(esito=='2'))**2).mean()/3
    inv = 1/t.psh+1/t.psd+1/t.psa
    mp1,mpx,mp2 = (1/t.psh)/inv,(1/t.psd)/inv,(1/t.psa)/inv
    mbrier = ((mp1-(esito=='1'))**2+(mpx-(esito=='X'))**2+(mp2-(esito=='2'))**2).mean()/3
    print(f"{div}: Brier modello {brier:.4f} | Brier mercato {mbrier:.4f} | {'OK, batte/pareggia il mercato' if brier<=mbrier*1.02 else 'sotto il mercato, da rivedere'}")
    return par, brier, mbrier

def espn_fixtures(espn_code, back=4, fwd=10):
    """Partite ESPN: FINITE (ultimi `back` giorni, con risultato reale) e
    PROGRAMMATE (prossimi `fwd` giorni). Stesso pattern usato per il Mondiale."""
    out=[]; now=dt.datetime.now(dt.timezone.utc)
    for k in range(-back, fwd):
        d=(now+dt.timedelta(days=k)).strftime("%Y%m%d")
        u=f"https://site.api.espn.com/apis/site/v2/sports/soccer/{espn_code}/scoreboard?dates={d}"
        try:
            data=json.load(urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=15))
        except Exception: continue
        for e in data.get("events",[]):
            c=e["competitions"][0]; st=c["status"]["type"]["name"]
            cm={x["homeAway"]:x for x in c["competitors"]}
            h,a = cm.get("home",{}).get("team",{}).get("displayName"), cm.get("away",{}).get("team",{}).get("displayName")
            if not h or not a: continue
            played = st in ("STATUS_FULL_TIME","STATUS_FINAL")
            real = None
            if played:
                try: real = {"hs":int(cm["home"]["score"]),"as":int(cm["away"]["score"])}
                except Exception: played=False
            out.append({"date":e.get("date"),"home":h,"away":a,"played":played,"real":real})
    return out

def run_league(div, verbose=True, cur_season="2026-2027"):
    """Calibra (tutte le stagioni, inclusa la corrente), backtesta e genera le
    predizioni per le partite vere (giocate di recente + programmate).
    FIX nov 2026: prima usava l'ELO di FINE STAGIONE SCORSA (last_elo dalle sole
    stagioni di training); ora ELO post-partita e valori rosa della stagione in corso."""
    import season_engine as se
    conn = sqlite3.connect(DBFILE)
    df = se.load_div(conn, div)
    vals = se.season_values(df, se.load_values(div))
    par = se.train_params(df, vals, "2099-2100")          # tutte le stagioni, corrente inclusa
    if par is None:
        if verbose: print(f"{div}: dati insufficienti"); return None
    par = {k: float(v) for k, v in par.items()}
    # il backtest e' pesante e non cambia da un giorno all'altro: cache 7 giorni
    cache_p = f"{BASE}/data/match_bt_cache.json"
    try: cache = json.load(open(cache_p))
    except Exception: cache = {}
    now_ts = dt.datetime.now().timestamp()
    c = cache.get(div)
    if c and now_ts - c["ts"] < 7*86400:
        brier, mbrier = c["brier"], c["mbrier"]
    else:
        bt = backtest(div)
        brier = bt[1] if bt else None; mbrier = bt[2] if bt else None
        cache[div] = {"ts": now_ts, "brier": brier, "mbrier": mbrier}
        json.dump(cache, open(cache_p, "w"))

    cur = df[df.season == cur_season]
    if len(cur) < 10:
        if verbose: print(f"{div}: stagione corrente senza partite"); return None
    teams, ix, pts, gf, ga, pl = se.table_from(cur)
    elo_cur = dict(zip(teams, se.elo_now(cur, teams)))
    mv = {t: vals.get((cur_season, t), 50.0) for t in teams}
    # ELO PRE-partita delle gare gia' giocate (dal DB): la previsione mostrata per una
    # partita finita deve essere quella che il modello avrebbe dato PRIMA, non col senno di poi
    pre = {(r.home_team, r.away_team): (r.elo_home_calc, r.elo_away_calc) for r in cur.itertuples()}

    fx = espn_fixtures(LEAGUE_ESPN.get(div,""))
    if verbose: print(f"{div}: partite ESPN trovate {len(fx)}")
    rows=[]; unmatched=set()
    for f in fx:
        h = se.espn_to_db(f["home"], teams); a = se.espn_to_db(f["away"], teams)
        if not h or not a:
            unmatched.add(f["home"] if not h else f["away"]); continue
        if f["played"] and (h, a) in pre:
            ed=(pre[(h, a)][0]-pre[(h, a)][1])/400
        else:
            ed=(elo_cur[h]-elo_cur[a])/400
        lval=np.log(max(mv[h],1))-np.log(max(mv[a],1))
        pr = match_probs(par, ed, lval)
        rows.append({"date":f["date"],"home":h,"away":a,"played":f["played"],"real":f["real"],**pr})
    if unmatched and verbose: print(f"   ⚠ non agganciate: {sorted(unmatched)}")
    if rows:
        d=pd.DataFrame(rows).sort_values('date')
        d.to_csv(f"{BASE}/data/match_preds_{div}.csv", index=False)
        if verbose: print(f"   ✓ {len(rows)} partite salvate (data/match_preds_{div}.csv)")
    return {"div":div,"par":par,"brier":brier,"mbrier":mbrier,
            "n_matches":len(rows),"n_unmatched":len(unmatched),"matches":rows}

def main():
    div = sys.argv[1] if len(sys.argv)>1 else "I1"
    only_bt = "--backtest" in sys.argv
    if only_bt:
        backtest(div); return
    r = run_league(div)
    if r and r["matches"]:
        df=pd.DataFrame(r["matches"])
        print(df[['date','home','away','p1','px','p2','over25','played']].round(3).to_string(index=False))

if __name__=="__main__":
    main()
