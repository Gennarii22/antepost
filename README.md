# Antepost

Probabilità di titolo, zona alta e retrocessione per 16 campionati europei, ricalcolate a ogni giornata.
Modello: Poisson per partita su ELO interno + valore rosa Transfermarkt, calendario residuo simulato 30.000 volte, condizionato alla classifica reale.

- `site/` : sito statico (Netlify pubblica questa cartella)
- `model/` : motore e pipeline (`cloud_update.py` fa tutto)
- `.github/workflows/update.yml` : aggiornamento automatico ogni 2 ore
- `model/data/history_core.parquet` : storico fisso delle partite (il DB completo non sta su Git)

Aggiornamento manuale in locale: `cd model && python3 cloud_update.py` (`--valori` forza il refresh dei valori rosa).
Se il refresh dei valori rosa viene bloccato da Transfermarkt su GitHub, il workflow tiene i valori precedenti: rinfrescali in locale con `--valori` e fai push.
