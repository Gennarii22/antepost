"""
gen_season_data.py — genera season_data.js per la dashboard Antepost in-stagione.

Per ogni campionato: stato reale (punti, DR, ELO attuale post-partita, valore rosa),
simulazione del calendario residuo, P(titolo/top4/retro), punti attesi.
Tiene uno storico degli snapshot (data/season_history.json) per mostrare di quanto si e'
mossa la probabilita' del modello rispetto a ieri/settimana scorsa (utile per il trading
delle quote: il mercato ha reagito o no?).

Parametri (rumore di forza sigma per giornata) letti da data/season_quality.json,
prodotto da season_quality.py a partire dal backtest.
"""
import sys, warnings; warnings.filterwarnings("ignore")
import json, os, datetime as dt
import season_engine as se

BASE = se.BASE
SEASON = "2026-2027"
N_SIMS = 30000

def sigma_for(matchday, quality):
    """rumore di forza ottimale (dal backtest) alla giornata piu' vicina."""
    best = quality.get("best_sigma_by_g", {})
    if not best: return 0.0
    g = min((int(k) for k in best), key=lambda k: abs(k - matchday))
    return float(best[str(g)])

def qa_check(leagues):
    """Controlli di coerenza sui dati prima di pubblicarli. Ritorna lista di problemi."""
    issues = []
    for l in leagues:
        t = l["teams"]; n = len(t); nm = f'{l["name"]}'
        if l["games_total"] != 2*(n-1): issues.append(f"{nm}: {n} squadre ma {l['games_total']} giornate (squadre duplicate?)")
        if abs(sum(x["p_title"] for x in t) - 1) > 0.02: issues.append(f"{nm}: somma P(titolo) = {sum(x['p_title'] for x in t):.3f}")
        if abs(sum(x["p_top4"] for x in t) - l["n_top"]) > 0.1: issues.append(f"{nm}: somma P(top4) = {sum(x['p_top4'] for x in t):.2f}")
        if abs(sum(x["p_rel"] for x in t) - l["n_rel"]) > 0.1: issues.append(f"{nm}: somma P(retro) = {sum(x['p_rel'] for x in t):.2f}")
        pl = [x["played"] for x in t]
        if max(pl) - min(pl) > 3: issues.append(f"{nm}: partite giocate molto diverse ({min(pl)}-{max(pl)}), nomi duplicati?")
        for x in t:
            if x["valore"] < 1: issues.append(f"{nm}: {x['team']} ha valore rosa {x['valore']}m (errore di parsing?)")
            if not (1200 <= x["elo"] <= 2100): issues.append(f"{nm}: {x['team']} ELO anomalo {x['elo']}")
        # la favorita non puo' essere una squadra con valore e ELO entrambi nella meta' bassa
        top = max(t, key=lambda x: x["p_title"])
        if top["p_title"] > 0.5:
            vs = sorted(x["valore"] for x in t); es = sorted(x["elo"] for x in t)
            if top["valore"] < vs[n//2] and top["elo"] < es[n//2]:
                issues.append(f"{nm}: favorita {top['team']} al {top['p_title']*100:.0f}% ma valore e ELO sotto la mediana")
    return issues

def main():
    q = json.load(open(f"{BASE}/data/season_quality.json")) if os.path.exists(f"{BASE}/data/season_quality.json") else {}
    hist_path = f"{BASE}/data/season_history.json"
    hist = json.load(open(hist_path)) if os.path.exists(hist_path) else []
    now = dt.datetime.now(dt.timezone.utc)
    prev = None
    for snap in reversed(hist):                         # ultimo snapshot vecchio di almeno 12h
        if (now - dt.datetime.fromisoformat(snap["ts"])).total_seconds() >= 12*3600:
            prev = snap; break

    leagues, snapshot = [], {}
    for div, name in se.LEAGUES.items():
        try:
            r = se.live_league_run(div, season=SEASON, sigma_fn=lambda md: sigma_for(md, q), n_sims=N_SIMS)
        except Exception as e:
            print(f"  {div}: errore {e}"); continue
        if r is None: continue
        out, meta = r
        snapshot[div] = {t.team: [round(t.p_title, 4), round(t.p_top4, 4), round(t.p_rel, 4)] for t in out.itertuples()}
        pv = (prev or {}).get("leagues", {}).get(div, {})
        teams = []
        for t in out.itertuples():
            d = {"team": t.team, "played": int(t.played), "pts": int(t.pts), "gd": int(t.gd), "elo": int(t.elo),
                 "valore": float(t.valore), "exp_pts": float(t.exp_pts), "pts_sd": float(t.pts_sd),
                 "p_title": round(float(t.p_title), 4), "p_top4": round(float(t.p_top4), 4), "p_rel": round(float(t.p_rel), 4)}
            if t.team in pv:
                d["d_title"] = round(d["p_title"] - pv[t.team][0], 4)
                d["d_top4"] = round(d["p_top4"] - pv[t.team][1], 4)
                d["d_rel"] = round(d["p_rel"] - pv[t.team][2], 4)
            teams.append(d)
        leagues.append({"div": div, "name": name, "split": div in se.SPLIT_FORMAT,
                        "games_total": meta["games_total"], "remaining": meta["remaining"],
                        "matchday_min": meta["md_min"], "matchday_max": meta["md_max"],
                        "sigma": meta["sigma"], "skill": q.get("league_skill", {}).get(div),
                        "n_top": meta["n_top"], "n_rel": meta["n_rel"], "top_label": meta["top_label"],
                        "adjustments": meta["adjustments"], "filled": meta["filled"], "notes": meta["notes"], "teams": teams})
        print(f"  {div} {name}: giornata {meta['md_min']}-{meta['md_max']}/{meta['games_total']}, sigma {meta['sigma']}, "
              f"favorita {teams[0]['team']} {teams[0]['p_title']*100:.1f}%", flush=True)

    issues = qa_check(leagues)
    for l in leagues:
        for nt in l["notes"]: issues.append(f'{l["name"]}: {nt}')     # discrepanze con la classifica ESPN
    if issues:
        print("\n⚠⚠ CONTROLLI DI COERENZA FALLITI:"); [print("   -", i) for i in issues]
    else:
        print("\n✓ controlli di coerenza ok (somme probabilità, squadre, valori, ELO, favorita)")
    if len(leagues) < len(se.LEAGUES):
        print(f"\n✗ STOP: solo {len(leagues)}/{len(se.LEAGUES)} campionati calcolati. season_data.js NON sovrascritto.")
        sys.exit(1)
    note = q.get("note", "")
    D = {"season": SEASON, "generated": now.strftime("%d/%m/%Y %H:%M UTC"), "n_sims": N_SIMS,
         "sigma": q.get("best_sigma_by_g", {}).get("9", 0), "quality_note": note, "leagues": leagues}
    open(f"{se.SITE}/season_data.js", "w").write("window.SEASON = " + json.dumps(D, ensure_ascii=False) + ";")
    # storico: un nuovo snapshot al massimo ogni 6 ore
    if not hist or (now - dt.datetime.fromisoformat(hist[-1]["ts"])).total_seconds() >= 6*3600:
        hist.append({"ts": now.isoformat(), "leagues": snapshot}); hist = hist[-80:]
        json.dump(hist, open(hist_path, "w"))
    print(f"\n✓ season_data.js: {len(leagues)} campionati")

if __name__ == "__main__":
    main()
