#!/usr/bin/env bash
# Push sicuro anti-race tra bot auto-update e modifiche manuali.
#  1) le MODIFICHE DI CODICE/CONFIG vincono sempre;
#  2) sui FILE-DATI generati vince il piu' recente (timestamp "generated" dentro season_data.js).
set -e
cd "$(dirname "$0")"
MSG="${1:-auto-update Antepost $(date -u +'%Y-%m-%d %H:%M UTC')}"
DATA_FILES="site/season_data.js site/matches_data.js model/data/season_history.json model/data/market_values_all.parquet model/data/match_bt_cache.json"

git add -A
git diff --cached --quiet && { echo "Niente da committare."; exit 0; }
git commit -q -m "$MSG"

for attempt in 1 2 3 4 5; do
  if git push origin main 2>/dev/null; then echo "✓ push riuscito (tentativo $attempt)"; exit 0; fi
  echo "Push rifiutato: riallineo preservando le mie modifiche…"
  git fetch -q origin main
  if ! git rebase -X theirs origin/main; then
    git rebase --abort
    git reset --soft origin/main; git commit -q -m "$MSG (re-applied)"; continue
  fi
done
echo "✗ push non riuscito dopo i tentativi"; exit 1
