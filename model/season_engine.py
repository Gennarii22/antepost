"""
season_engine.py — Antepost IN-STAGIONE per tutti i campionati.

Dato lo stato reale di una stagione (punti, differenza reti, ELO attuale,
valore rosa), simula il RESTO del calendario partita per partita con il motore
Poisson validato (match_engine: ELO + valore rosa + vantaggio casa per lega) e
restituisce P(titolo), P(top4), P(retrocessione) e punti finali attesi.

Idee prese dal modello Mondiale: simulazione condizionata ai risultati reali +
rumore di forza correlato per squadra (allarga le code, e' quello che rende le
probabilita' calibrate invece che sovra-sicure).

Backtest walk-forward (parametri tarati SOLO su stagioni precedenti) con
confronto contro la regressione per-giornata storica ("PRO", gaussiana).

Uso:
  python3 season_engine.py backtest 9            # giornata 9, tutte le leghe
  python3 season_engine.py backtest 9 I1 E0      # solo alcune
  python3 season_engine.py live                  # stato attuale 2026-2027
"""
import os
import warnings; warnings.filterwarnings("ignore")
import sqlite3, sys, json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

DBFILE = os.environ.get("EURO_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "core", "euro_core.db"))
BASE = os.path.dirname(os.path.abspath(__file__))

LEAGUES = {"I1":"Serie A","E0":"Premier League","SP1":"LaLiga","D1":"Bundesliga","F1":"Ligue 1",
           "N1":"Eredivisie","P1":"Liga Portugal","B1":"Pro League","T1":"Süper Lig",
           "G1":"Super League GR","SC0":"Scottish Prem","I2":"Serie B","E1":"Championship",
           "SP2":"LaLiga 2","F2":"Ligue 2","D2":"2.Bundesliga"}
# formati con spareggi / split: la tabella a fine regular season NON decide il titolo
# (Belgio playoff a punti dimezzati, Scozia split dopo 33, Grecia playoff): stima indicativa
SPLIT_FORMAT = {"B1", "SC0", "G1"}
N_TOP, N_REL = 4, 3
# zone REALI per campionato (lette dalle note della classifica ESPN, 03/11/2026):
# (posti "alti", retrocessioni dirette, etichetta). Il backtest resta a 4/3 uniforme (metrica validata).
# B1 e G1 hanno playoff senza zona netta: restano 4/3 indicativi.
ZONES = {"I1":(4,3,"Champions League"), "E0":(4,3,"Champions League"), "SP1":(5,3,"Champions League"),
         "D1":(4,2,"Champions League"), "F1":(3,2,"Champions League diretta"), "N1":(2,2,"Champions League"),
         "P1":(1,2,"Champions League"), "B1":(4,3,"Top 4 (indicativo)"), "T1":(1,3,"Champions League"),
         "G1":(4,3,"Top 4 (indicativo)"), "SC0":(2,1,"Champions League (qualif.)"),
         "I2":(2,3,"Promozione diretta"), "E1":(2,3,"Promozione diretta"), "SP2":(2,4,"Promozione diretta"),
         "F2":(2,2,"Promozione diretta"), "D2":(2,2,"Promozione diretta")}
K_ELO, HFA_ELO = 20.0, 65.0
_SITE = os.path.join(BASE, "..", "site")
SITE = _SITE if os.path.isdir(_SITE) else BASE   # dove scrivere season_data.js / matches_data.js

def gd_mult(gd):
    if gd <= 1: return 1.0
    if gd == 2: return 1.5
    return (11 + gd) / 8.0

# ───────────────────────────── dati ─────────────────────────────────────
def load_values(div):
    mv = pd.read_parquet(f"{BASE}/data/market_values_all.parquet")
    mv = mv[(mv["div"] == div) & (mv["market_value_mln"] >= 0.3)]   # valori ~0 = errori di parsing: li imputa season_values
    return {(r.season, r.team): r.market_value_mln for r in mv.itertuples()}

def load_div(conn, div):
    df = pd.read_sql("""select id,season,date,home_team,away_team,fthg,ftag,elo_home_calc,elo_away_calc
                        from matches where div=? and fthg is not null and ftag is not null
                        and elo_home_calc is not null order by date,id""", conn, params=(div,))
    df["date"] = pd.to_datetime(df["date"])
    return df

def season_values(df, vmap):
    """valore rosa per (stagione, squadra); buchi imputati da regressione su ELO medio."""
    rows = []
    for s, g in df.groupby("season"):
        elo = pd.concat([g[["home_team","elo_home_calc"]].set_axis(["t","e"], axis=1),
                         g[["away_team","elo_away_calc"]].set_axis(["t","e"], axis=1)]).groupby("t").e.mean()
        for t, e in elo.items():
            rows.append({"season": s, "team": t, "elo": e, "v": vmap.get((s, t), np.nan)})
    st = pd.DataFrame(rows)
    known = st.dropna(subset=["v"])
    if len(known) >= 12:
        b = np.polyfit(known.elo, np.log(known.v.clip(lower=1)), 1)
        miss = st.v.isna()
        st.loc[miss, "v"] = np.exp(np.polyval(b, st.loc[miss, "elo"]))
    else:
        st["v"] = st.v.fillna(known.v.median() if len(known) else 50.0)
    return {(r.season, r.team): float(r.v) for r in st.itertuples()}

# ──────────────────────── calibrazione Poisson ─────────────────────────
def fast_calibrate(ed, lval, gh, ga):
    def nll(p):
        a, b, c, h = p
        lh = a + h + b*ed + c*lval
        la = a - b*ed - c*lval
        return -(gh*lh - np.exp(lh) + ga*la - np.exp(la)).sum()
    best = None
    for x0 in ([0.15,0.5,0.1,0.15], [0.1,0.2,0.3,0.1], [0.2,0.8,0.0,0.2]):
        r = minimize(nll, x0, method="Nelder-Mead", options={"maxiter":4000,"xatol":1e-5,"fatol":1e-5})
        if best is None or r.fun < best.fun: best = r
    return dict(zip(["a","b","c","hadv"], best.x))

def train_params(df, vals, before_season):
    tr = df[df.season < before_season].copy()
    tr["vh"] = [vals.get((s, t), np.nan) for s, t in zip(tr.season, tr.home_team)]
    tr["va"] = [vals.get((s, t), np.nan) for s, t in zip(tr.season, tr.away_team)]
    tr = tr.dropna(subset=["vh", "va"])
    tr = tr[tr.season >= "2015-2016"]          # i valori rosa partono da li'
    if len(tr) < 200: return None
    ed = ((tr.elo_home_calc - tr.elo_away_calc) / 400).values
    lv = (np.log(tr.vh.clip(lower=1)) - np.log(tr.va.clip(lower=1))).values
    return fast_calibrate(ed, lv, tr.fthg.values, tr.ftag.values)

# ───────────── allineamento alla classifica ufficiale (penalizzazioni) ─────────────
# football-data.co.uk registra solo i risultati: le penalizzazioni in classifica
# (es. -4 punti) non ci sono. La classifica ESPN le include: dove giocate e DR
# coincidono con le nostre ma i punti no, la differenza e' una penalizzazione.
import urllib.request, difflib, unicodedata
ESPN_CODE = {"I1":"ita.1","E0":"eng.1","SP1":"esp.1","D1":"ger.1","F1":"fra.1","N1":"ned.1","P1":"por.1",
             "B1":"bel.1","T1":"tur.1","G1":"gre.1","SC0":"sco.1","I2":"ita.2","E1":"eng.2","SP2":"esp.2",
             "F2":"fra.2","D2":"ger.2"}
ESPN_ALIAS = {"Ath Bilbao":"Athletic Club","Ath Madrid":"Atletico Madrid","La Coruna":"Deportivo","M'gladbach":"Borussia Monchengladbach",
              "Paris SG":"Paris Saint-Germain","Rennes":"Stade Rennais","Sp Lisbon":"Sporting CP","Amedspor":"Amed SFK",
              "Buyuksehyr":"Istanbul Basaksehir","Hearts":"Heart of Midlothian","QPR":"Queens Park Rangers",
              "Wolves":"Wolverhampton Wanderers","Celta B":"RC Celta Fortuna"}
def _na(x): return "".join(c for c in unicodedata.normalize("NFKD", x.lower()) if not unicodedata.combining(c))
def _simn(a, b):
    a, b = _na(ESPN_ALIAS.get(a, a)), _na(b)
    if a == b: return 1.0
    if a in b or b in a: return .9
    return difflib.SequenceMatcher(None, a, b).ratio()

def espn_to_db(name, teams, thr=.6):
    """Nome ESPN -> nome nel DB (alias + fuzzy). None se non agganciabile."""
    best = max(teams, key=lambda t: _simn(t, name), default=None)
    return best if best and _simn(best, name) >= thr else None

def espn_standings(div):
    """{nome ESPN: (giocate, punti, DR)} oppure None se ESPN non risponde."""
    code = ESPN_CODE.get(div)
    if not code: return None
    try:
        u = f"https://site.api.espn.com/apis/v2/sports/soccer/{code}/standings"
        d = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=25))
        out = {}
        for ch in d.get("children", []):
            for e in ch["standings"]["entries"]:
                st = {x["name"]: x.get("value") for x in e["stats"]}
                out[e["team"]["displayName"]] = (int(st.get("gamesPlayed", 0)), int(st.get("points", 0)), int(st.get("pointDifferential", 0)))
        return out or None
    except Exception:
        return None

def standings_check(div, teams, pl, pts0, gf0, ga0):
    """Confronta la nostra classifica con quella ESPN. Ritorna (adj, notes):
    adj = {squadra: delta punti} da applicare (penalizzazioni), notes = anomalie da segnalare."""
    es = espn_standings(div); adj = {}; notes = []
    if not es: return adj, ["ESPN non raggiungibile: nessun controllo di classifica"]
    used = set()
    for i, t in enumerate(teams):
        cands = [e for e in es if e not in used]
        best = max(cands, key=lambda e: _simn(t, e), default=None)
        if not best or _simn(t, best) < .6: notes.append(f"{t}: non trovata su ESPN"); continue
        used.add(best); g, p, gd = es[best]; og, op, ogd = int(pl[i]), int(pts0[i]), int(gf0[i] - ga0[i])
        if (g, gd) == (og, ogd) and p != op: adj[t] = p - op
        elif g != og: notes.append(f"{t}: ESPN {g} partite, DB {og} (risultato mancante nel DB)")
        elif (p, gd) != (op, ogd): notes.append(f"{t}: punti/DR diversi da ESPN ({op},{ogd} vs {p},{gd})")
    return adj, notes

# ─────────────────────────── stato stagione ────────────────────────────
def game_index(sd):
    """per ogni partita: n-esima partita stagionale di casa e di trasferta."""
    sd = sd.sort_values(["date", "id"]).reset_index(drop=True)
    cnt = {}; hi = []; ai = []
    for h, a in zip(sd.home_team, sd.away_team):
        cnt[h] = cnt.get(h, 0) + 1; cnt[a] = cnt.get(a, 0) + 1
        hi.append(cnt[h]); ai.append(cnt[a])
    sd["hg"], sd["ag"] = hi, ai
    return sd

def table_from(played):
    teams = sorted(set(played.home_team) | set(played.away_team))
    ix = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    pts = np.zeros(n); gf = np.zeros(n); ga = np.zeros(n); pl = np.zeros(n)
    for r in played.itertuples():
        i, j = ix[r.home_team], ix[r.away_team]
        gf[i] += r.fthg; ga[i] += r.ftag; gf[j] += r.ftag; ga[j] += r.fthg
        pl[i] += 1; pl[j] += 1
        if r.fthg > r.ftag: pts[i] += 3
        elif r.fthg < r.ftag: pts[j] += 3
        else: pts[i] += 1; pts[j] += 1
    return teams, ix, pts, gf, ga, pl

def elo_now(played, teams):
    """ELO dopo l'ultima partita giocata di ogni squadra (pre-match + aggiornamento)."""
    cur = {}
    for r in played.sort_values(["date", "id"]).itertuples():
        rh, ra = r.elo_home_calc, r.elo_away_calc
        exp_h = 1.0 / (1 + 10 ** ((ra - rh - HFA_ELO) / 400.0))
        res_h = 1.0 if r.fthg > r.ftag else (0.5 if r.fthg == r.ftag else 0.0)
        d = K_ELO * gd_mult(abs(int(r.fthg) - int(r.ftag))) * (res_h - exp_h)
        cur[r.home_team] = rh + d; cur[r.away_team] = ra - d
    return np.array([cur[t] for t in teams])

def espn_fill_missing(div, sd, days=21):
    """football-data carica i risultati con ~1 giorno di ritardo. Le partite finite su ESPN e
    assenti dal DB vengono aggiunte SOLO in memoria (il DB si riallinea da solo al CSV successivo).
    Ritorna (sd_aggiornato, lista partite aggiunte)."""
    import datetime as dt
    code = ESPN_CODE.get(div); teams = sorted(set(sd.home_team) | set(sd.away_team))
    if not code or not teams: return sd, []
    played = set(zip(sd.home_team, sd.away_team)); cur = dict(zip(teams, elo_now(sd, teams)))
    def to_db(name):
        best = max(teams, key=lambda t: _simn(t, name)); return best if _simn(best, name) >= .6 else None
    add = []; today = dt.date.today()
    for k in range(days + 1):
        d = (today - dt.timedelta(days=k)).strftime("%Y%m%d")
        try:
            u = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/scoreboard?dates={d}"
            j = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=20))
        except Exception: continue
        for e in j.get("events", []):
            if e["status"]["type"]["name"] != "STATUS_FULL_TIME": continue
            c = e["competitions"][0]["competitors"]
            hh = next(x for x in c if x["homeAway"] == "home"); aa = next(x for x in c if x["homeAway"] == "away")
            h, a = to_db(hh["team"]["displayName"]), to_db(aa["team"]["displayName"])
            if not h or not a or h == a or (h, a) in played: continue
            add.append({"id": 10**9 + len(add), "season": "2026-2027", "date": pd.Timestamp(d),
                        "home_team": h, "away_team": a, "fthg": int(hh["score"]), "ftag": int(aa["score"]),
                        "elo_home_calc": cur[h], "elo_away_calc": cur[a]}); played.add((h, a))
    if not add: return sd, []
    return pd.concat([sd, pd.DataFrame(add)], ignore_index=True), add

# ───────────────────────────── simulazione ─────────────────────────────
def simulate(par, sigma, teams, pts0, gf0, ga0, elo, val, fixtures, n_sims=20000, seed=42, n_top=None, n_rel=None):
    n_top = N_TOP if n_top is None else n_top; n_rel = N_REL if n_rel is None else n_rel
    """fixtures: lista (i_casa, i_trasf) da giocare. Ritorna dict di array per squadra."""
    rng = np.random.default_rng(seed)
    n = len(teams)
    delta = rng.normal(0, sigma, (n_sims, n)) if sigma > 0 else np.zeros((n_sims, n))
    pts = np.tile(pts0, (n_sims, 1)); gf = np.tile(gf0, (n_sims, 1)); ga = np.tile(ga0, (n_sims, 1))
    lv = np.log(np.clip(val, 1, None))
    for (i, j) in fixtures:
        ed = (elo[i] - elo[j]) / 400; l = lv[i] - lv[j]
        ln_h = par["a"] + par["hadv"] + par["b"]*ed + par["c"]*l
        ln_a = par["a"] - par["b"]*ed - par["c"]*l
        d = delta[:, i] - delta[:, j]
        gh = rng.poisson(np.exp(ln_h + d)); gaw = rng.poisson(np.exp(ln_a - d))
        pts[:, i] += 3*(gh > gaw) + (gh == gaw); pts[:, j] += 3*(gaw > gh) + (gh == gaw)
        gf[:, i] += gh; ga[:, i] += gaw; gf[:, j] += gaw; ga[:, j] += gh
    key = pts*1e6 + (gf - ga)*1e3 + gf + rng.random((n_sims, n))*0.5
    rank = (-key).argsort(1).argsort(1)            # 0 = primo
    return {"exp_pts": pts.mean(0), "p_title": (rank == 0).mean(0),
            "p_top4": (rank < n_top).mean(0), "p_rel": (rank >= n - n_rel).mean(0),
            "pts_sd": pts.std(0)}

# ───────────────────────────── baseline PRO ────────────────────────────
def snapshot_features(sd_all, vals, season, g, nteam_games):
    """feature per-squadra a giornata g (stesso stato del simulatore)."""
    sd = game_index(sd_all[sd_all.season == season])
    played = sd[(sd.hg <= g) & (sd.ag <= g)]
    teams, ix, pts, gf, ga, pl = table_from(played)
    elo = elo_now(played, teams)
    val = np.array([vals.get((season, t), 50.0) for t in teams])
    den = gf**1.275 + ga**1.275
    xpts = np.where(den > 0, 3*nteam_games*gf**1.275/den, 1.5*nteam_games)
    return teams, pd.DataFrame({"xpts": xpts, "cum_pts": pts, "gd": gf - ga, "elo": elo,
                                "lval": np.log(np.clip(val, 1, None))}), pts, pl

def baseline_probs(train_X, train_y, X, cum_pts, rem_games, n_sims=20000, seed=7):
    A = np.c_[np.ones(len(train_X)), train_X]
    beta, *_ = np.linalg.lstsq(A, train_y, rcond=1e-8)
    res = train_y - A @ beta
    dof = max(len(train_y) - A.shape[1], 1)
    sigma = np.sqrt((res**2).sum() / dof)
    mu = np.c_[np.ones(len(X)), X] @ beta
    rng = np.random.default_rng(seed); n = len(mu)
    lo = cum_pts; hi = cum_pts + 3*rem_games
    draws = np.clip(rng.normal(mu[:, None], sigma, (n, n_sims)), lo[:, None], hi[:, None])
    rank = (-draws).argsort(0).argsort(0)
    return mu, (rank == 0).mean(1), (rank < N_TOP).mean(1), (rank >= n - N_REL).mean(1)

# ──────────────────────────── backtest ─────────────────────────────────
def final_table(sd):
    teams, ix, pts, gf, ga, pl = table_from(sd)
    key = pts*1e6 + (gf - ga)*1e3 + gf
    rank = (-key).argsort().argsort()
    return teams, pts, rank

def brier(p, y): return float(np.mean((p - y)**2))
def logloss(p, y, eps=1e-3):
    p = np.clip(p, eps, 1 - eps); return float(-np.mean(y*np.log(p) + (1-y)*np.log(1-p)))

def backtest(g, divs, sigmas=(0.0, 0.08, 0.14, 0.20), n_sims=8000, first_test="2018-2019"):
    conn = sqlite3.connect(DBFILE)
    out = []; PRED = []
    def keep(model, teams, r_title, r_top4, r_rel, mu):
        for k, t in enumerate(teams):
            PRED.append({"div": div, "season": S, "g": g, "team": t, "model": model,
                         "p_title": r_title[k], "p_top4": r_top4[k], "p_rel": r_rel[k], "exp_pts": mu[k],
                         "y_title": y_title[k], "y_top4": y_top4[k], "y_rel": y_rel[k], "y_pts": y_pts[k],
                         "pts_now": pts0[k], "split": div in SPLIT_FORMAT})
    for div in divs:
        df = load_div(conn, div)
        vals = season_values(df, load_values(div))
        seasons = sorted(s for s in df.season.unique() if s >= first_test and s < "2026-2027")
        for S in seasons:
            sd_full = game_index(df[df.season == S])
            teams_f, pts_f, rank_f = final_table(sd_full)
            nteams = len(teams_f)
            if nteams < 10: continue
            games = 2 * (nteams - 1)
            if sd_full.hg.max() < games - 2: continue            # stagione incompleta
            played = sd_full[(sd_full.hg <= g) & (sd_full.ag <= g)]
            remain = sd_full[~sd_full.index.isin(played.index)]
            teams, ix, pts0, gf0, ga0, pl = table_from(played)
            if len(teams) != nteams: continue
            par = train_params(df, vals, S)
            if par is None: continue
            elo = elo_now(played, teams)
            val = np.array([vals.get((S, t), 50.0) for t in teams])
            fx = [(ix[r.home_team], ix[r.away_team]) for r in remain.itertuples()]
            ft = {t: (p, r) for t, p, r in zip(teams_f, pts_f, rank_f)}
            y_pts = np.array([ft[t][0] for t in teams]); y_rank = np.array([ft[t][1] for t in teams])
            y_title = (y_rank == 0).astype(float); y_top4 = (y_rank < N_TOP).astype(float)
            y_rel = (y_rank >= nteams - N_REL).astype(float)
            base = {"div": div, "season": S, "g": g, "n": nteams}
            # modello Poisson con piu' valori di sigma
            for sg in sigmas:
                r = simulate(par, sg, teams, pts0, gf0, ga0, elo, val, fx, n_sims=n_sims)
                keep(f"poisson_s{sg}", teams, r["p_title"], r["p_top4"], r["p_rel"], r["exp_pts"])
                out.append({**base, "model": f"poisson_s{sg}", "mae_pts": float(np.abs(r["exp_pts"] - y_pts).mean()),
                            "b_title": brier(r["p_title"], y_title), "b_top4": brier(r["p_top4"], y_top4),
                            "b_rel": brier(r["p_rel"], y_rel), "l_title": logloss(r["p_title"], y_title),
                            "l_top4": logloss(r["p_top4"], y_top4), "l_rel": logloss(r["p_rel"], y_rel),
                            "pt_title_winner": float(r["p_title"][y_title == 1].sum()),
                            "split": div in SPLIT_FORMAT})
            # baseline PRO: training sulle stagioni precedenti allo stesso g
            Xtr, ytr = [], []
            for s2 in sorted(df.season.unique()):
                if s2 >= S or s2 < "2015-2016": continue
                sdf = game_index(df[df.season == s2]); t2, p2, _ = final_table(sdf)
                if len(t2) != nteams or sdf.hg.max() < games - 2: continue
                tt, F, c2, _ = snapshot_features(df, vals, s2, g, games)
                fp = dict(zip(t2, p2)); Xtr.append(F.values); ytr.append(np.array([fp[t] for t in tt]))
            if len(Xtr) >= 3:
                Xtr = np.vstack(Xtr); ytr = np.concatenate(ytr)
                tt, F, c0, _ = snapshot_features(df, vals, S, g, games)
                mu, pt, p4, pr = baseline_probs(Xtr, ytr, F.values, c0, games - g)
                # riallinea alle squadre del backtest
                order = [list(tt).index(t) for t in teams]
                mu, pt, p4, pr = mu[order], pt[order], p4[order], pr[order]
                keep("pro_regressione", teams, pt, p4, pr, mu)
                out.append({**base, "model": "pro_regressione", "mae_pts": float(np.abs(mu - y_pts).mean()),
                            "b_title": brier(pt, y_title), "b_top4": brier(p4, y_top4), "b_rel": brier(pr, y_rel),
                            "l_title": logloss(pt, y_title), "l_top4": logloss(p4, y_top4), "l_rel": logloss(pr, y_rel),
                            "pt_title_winner": float(pt[y_title == 1].sum()), "split": div in SPLIT_FORMAT})
        print(f"  {div} fatto", flush=True)
    return pd.DataFrame(out), pd.DataFrame(PRED)

def summarize(res):
    r = res[~res.split]
    g = r.groupby("model").agg(n=("div", "size"), mae_pts=("mae_pts", "mean"),
        b_title=("b_title", "mean"), b_top4=("b_top4", "mean"), b_rel=("b_rel", "mean"),
        l_title=("l_title", "mean"), l_top4=("l_top4", "mean"), l_rel=("l_rel", "mean"),
        p_winner=("pt_title_winner", "mean"))
    return g

# ──────────────────────────── live ─────────────────────────────────────
def live_league(div, par=None, sigma=0.14, n_sims=30000, season="2026-2027"):
    conn = sqlite3.connect(DBFILE)
    df = load_div(conn, div)
    vmap = load_values(div)
    vals = season_values(df, vmap)
    sd = df[df.season == season]
    if len(sd) < 10: return None
    sd = game_index(sd)
    teams, ix, pts0, gf0, ga0, pl = table_from(sd)
    n = len(teams)
    if par is None: par = train_params(df, vals, season)
    elo = elo_now(sd, teams)
    val = np.array([vals.get((season, t), vals.get(("2026-2027", t), 50.0)) for t in teams])
    played_pairs = set(zip(sd.home_team, sd.away_team))
    fx = [(ix[h], ix[a]) for h in teams for a in teams if h != a and (h, a) not in played_pairs]
    r = simulate(par, sigma, teams, pts0, gf0, ga0, elo, val, fx, n_sims=n_sims)
    out = pd.DataFrame({"team": teams, "played": pl.astype(int), "pts": pts0.astype(int), "gd": (gf0 - ga0).astype(int),
                        "elo": elo.round(0), "valore": val.round(1), "exp_pts": r["exp_pts"].round(1),
                        "pts_sd": r["pts_sd"].round(1), "p_title": r["p_title"], "p_top4": r["p_top4"], "p_rel": r["p_rel"]})
    out = out.sort_values(["p_title", "exp_pts"], ascending=False).reset_index(drop=True)
    return out, par, len(fx), 2*(n-1)

def live_league_run(div, season="2026-2027", sigma_fn=lambda md: 0.0, n_sims=30000):
    """Stato attuale + simulazione del calendario residuo. Parametri Poisson tarati su
    TUTTE le stagioni disponibili (corrente inclusa). Ritorna (DataFrame, meta)."""
    conn = sqlite3.connect(DBFILE)
    df = load_div(conn, div)
    vals = season_values(df, load_values(div))
    sd = df[df.season == season]
    if len(sd) < 10: return None
    sd = game_index(sd)
    sd, filled = espn_fill_missing(div, sd)
    if filled: sd = game_index(sd)
    teams, ix, pts0, gf0, ga0, pl = table_from(sd)
    n = len(teams)
    adj, notes = standings_check(div, teams, pl, pts0, gf0, ga0)
    for t, dlt in adj.items(): pts0[ix[t]] += dlt
    par = train_params(df, vals, "2099-2100")
    elo = elo_now(sd, teams)
    val = np.array([vals.get((season, t), 50.0) for t in teams])
    played_pairs = set(zip(sd.home_team, sd.away_team))
    fx = [(ix[h], ix[a]) for h in teams for a in teams if h != a and (h, a) not in played_pairs]
    md = int(np.median(pl)); sigma = float(sigma_fn(md))
    nt, nr, tl = ZONES.get(div, (N_TOP, N_REL, "Top 4"))
    r = simulate(par, sigma, teams, pts0, gf0, ga0, elo, val, fx, n_sims=n_sims, n_top=nt, n_rel=nr)
    out = pd.DataFrame({"team": teams, "played": pl.astype(int), "pts": pts0.astype(int), "gd": (gf0 - ga0).astype(int),
                        "elo": elo.round(0), "valore": val.round(1), "exp_pts": r["exp_pts"].round(1),
                        "pts_sd": r["pts_sd"].round(1), "p_title": r["p_title"], "p_top4": r["p_top4"], "p_rel": r["p_rel"]})
    out = out.sort_values(["p_title", "exp_pts"], ascending=False).reset_index(drop=True)
    meta = {"games_total": 2*(n-1), "remaining": len(fx), "md_min": int(pl.min()), "md_max": int(pl.max()),
            "sigma": sigma, "par": {k: round(float(v), 4) for k, v in par.items()},
            "n_top": nt, "n_rel": nr, "top_label": tl, "adjustments": adj, "notes": notes,
            "filled": [f"{r['home_team']} {r['fthg']}-{r['ftag']} {r['away_team']}" for r in filled]}
    return out, meta

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "backtest"
    if cmd == "backtest":
        g = int(sys.argv[2]) if len(sys.argv) > 2 else 9
        divs = sys.argv[3:] or list(LEAGUES)
        res, pred = backtest(g, divs)
        res.to_csv(f"{BASE}/data/backtest_season_g{g}.csv", index=False)
        pred.to_parquet(f"{BASE}/data/backtest_preds_g{g}.parquet", index=False)
        pd.set_option("display.width", 200); pd.set_option("display.float_format", lambda v: f"{v:.4f}")
        print("\n=== RIEPILOGO (escluse leghe con spareggi) ===")
        print(summarize(res))
    elif cmd == "live":
        for div in LEAGUES:
            r = live_league(div)
            if r: print(div, r[0].head(3)[["team", "pts", "p_title"]].values.tolist())
