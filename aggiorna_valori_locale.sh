#!/usr/bin/env bash
# Refresh dei valori rosa Transfermarkt DAL TUO MAC (da GitHub Transfermarkt blocca gli IP).
# Consigliato: quando apre il mercato (gennaio, giugno-agosto) e ogni tanto durante la stagione.
# Ricostruisce il DB compatto, scarica i valori, li valida (se sospetti tiene i vecchi),
# rigenera i dati e fa push: il sito si aggiorna da solo.
set -e
cd "$(dirname "$0")/model"
python3 cloud_update.py --valori
cd ..
git pull -q --rebase || true
bash safe_push.sh "valori rosa aggiornati $(date -u +'%Y-%m-%d')"
