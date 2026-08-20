# 🚢 Prédiction du temps de transit TIR Export — Tanger Med

> Système de Machine Learning de bout en bout pour prédire le temps de transit des camions TIR export au port de Tanger Med, depuis l'ingestion des données brutes jusqu'au tableau de bord interactif.

**Réalisé par :** EL-MOUTOUK MOHAMED YASSIR  
**Encadrant :** M. KAMAL ADDI  
**Organisme :** Tanger Med Port Authority  
**Période :** Juin – Août 2026

---

##  Contexte

Le port Tanger Med est l'un des principaux hubs logistiques de la Méditerranée. Chaque camion TIR export suit un parcours physique horodaté :

```
ZRE → Impression → Scanner export → Sortie SAS export → Entrée terminal → Embarquement
```

Le **temps de transit** (de `date_zre` à `date_embarquement`) n'est aujourd'hui connu qu'*a posteriori*. Ce projet construit un système capable de le **prédire à différents stades du parcours**.

---

##  Résultats clés

| Approche | MAE | RMSE | MAPE |
|---|---|---|---|
| Baseline naïve (médiane par couloir) | 9,54h | 13,49h | 53,8% |
| XGBoost — M1 ZRE | 8,71h | 12,57h | 44,3% |
| XGBoost — M2 Scanner | 8,27h | 11,99h | 42,2% |
| XGBoost — M3 SAS | 7,53h | 11,46h | 36,7% |
| **XGBoost — M4 Terminal** | **6,30h** | **10,24h** | **29,8%** |

**→ Réduction de 45% de l'erreur relative (MAPE) par rapport à l'approche naïve.**

---

##  Architecture

```
CSV brut (424 828 lignes)
        ↓
raw_tir_export (PostgreSQL)
        ↓ Prefect ETL Pipeline
clean_tir_export (394 662 lignes)
        ↓
training_tir_export (285 626 lignes)
        ↓
Feature Engineering (4 checkpoints)
        ↓
LightGBM + XGBoost + MLflow
        ↓
Optuna (hyperparameter tuning) + SHAP (interprétabilité)
        ↓
FastAPI (/predict)  →  Streamlit (dashboard)
```

---

##  Stack technique

| Catégorie | Technologies |
|---|---|
| **Ingestion / Stockage** | Python, Pandas, PostgreSQL, SQLAlchemy |
| **Orchestration ETL** | Prefect |
| **Machine Learning** | LightGBM, XGBoost, Scikit-learn |
| **Optimisation** | Optuna (TPE sampler) |
| **Interprétabilité** | SHAP |
| **Suivi expériences** | MLflow |
| **API** | FastAPI, Pydantic, Uvicorn |
| **Dashboard** | Streamlit |
| **Versionnement** | Git, GitHub |

---

##  Structure du projet

```
├── data/
│   └── raw/                    ← CSV source (non versionné)
├── ingestion/
│   ├── db_config.py            ← Connexion PostgreSQL
│   └── ingest_raw.py           ← CSV → raw_tir_export
├── pipeline/
│   ├── clean.py                ← Fonctions de nettoyage
│   └── etl_flow.py             ← Prefect flow (6 tâches)
├── ml/
│   ├── feature_engineering.py  ← 4 datasets parquet
│   ├── train.py                ← LightGBM + XGBoost + MLflow
│   ├── tune_optuna.py          ← Optuna hyperparameter tuning
│   ├── shap_analysis.py        ← Analyse SHAP
│   ├── baseline.py             ← Baseline naïve
│   ├── features/               ← Datasets parquet (non versionnés)
│   ├── models/                 ← Modèles .pkl (non versionnés)
│   └── shap_outputs/           ← Graphiques SHAP
├── api/
│   └── main.py                 ← FastAPI /predict
├── dashboard/
│   └── app.py                  ← Streamlit dashboard
├── notebooks/
│   └── eda.py                  ← Analyse exploratoire
├── .env                        ← Credentials (non versionné)
├── requirements.txt
└── README.md
```

---

##  Lancement

### Prérequis
- Python 3.10+
- PostgreSQL installé et configuré
- Fichier `.env` à la racine (voir `.env.example`)

### Installation

```bash
git clone https://github.com/elmoutoukmohamedyassir/Prediction-de-temps-de-transit-pour-les-unites-TIR.git
cd Prediction-de-temps-de-transit-pour-les-unites-TIR
python -m venv venv
source venv/bin/activate   # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

### Pipeline complet

```bash
# 1. Ingestion CSV → PostgreSQL
python -m ingestion.ingest_raw

# 2. ETL Pipeline (Prefect)
python -m pipeline.etl_flow

# 3. Feature Engineering
python -m ml.feature_engineering

# 4. Entraînement des modèles
python -m ml.train

# 5. Baseline naïve (comparaison)
python -m ml.baseline

# 6. Optimisation Optuna
python -m ml.tune_optuna

# 7. Analyse SHAP
python -m ml.shap_analysis
```

### API + Dashboard

```bash
# Terminal 1 — API FastAPI
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Dashboard Streamlit
streamlit run dashboard/app.py

# Terminal 3 — MLflow UI
mlflow ui
```

**Accès :**
- API : http://localhost:8000
- API Docs (Swagger) : http://localhost:8000/docs
- Dashboard : http://localhost:8501
- MLflow : http://localhost:5000

---

##  API — Exemple d'appel

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "couloir": "Couloir 4",
    "ss_type_unite": "Ensemble Routier",
    "vide_plein": "PLEIN",
    "nature_marchandise": "Textile / Confection",
    "poids": 12000,
    "is_groupage": 0,
    "matiere_danger": false,
    "date_zre": "2026-06-21T08:00:00+01:00",
    "checkpoint": 1
  }'
```

**Réponse :**
```json
{
  "transit_time_h_predicted": 15.6,
  "checkpoint": 1,
  "model_used": "xgb_M1_ZRE",
  "couloir": "Couloir 4",
  "ss_type_unite": "Ensemble Routier",
  "date_zre": "2026-06-21T08:00:00+01:00"
}
```

---

##  Modélisation — 4 checkpoints

| Checkpoint | Étape physique | Features disponibles |
|---|---|---|
| M1 | Arrivée au ZRE | Couloir, type unité, marchandise, poids, temporelles |
| M2 | Après scanner | M1 + passage scanner, durée scanner |
| M3 | Après SAS export | M2 + visite physique, durée SAS |
| M4 | Après entrée terminal | M3 + temps écoulé jusqu'au terminal |

---

##  SHAP — Variables les plus importantes (M4)

| Variable | Importance SHAP |
|---|---|
| temps_ecoule_apres_terminal | 0.1902 |
| mois_zre | 0.1042 |
| type_unite_enc | 0.0924 |
| temps_ecoule_apres_sas | 0.0780 |
| temps_ecoule_apres_scanner | 0.0778 |
| jour_semaine | 0.0246 |
| poids | 0.0223 |
| couloir_enc | 0.0150 |

---

##  Configuration `.env`

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=tir_transit
DB_USER=tir_user
DB_PASSWORD=your_password
```

---

##  Perspectives

- Régression quantile (intervalles de confiance)
- Modélisation à deux étages (classifieur retard + régresseur)
- Conteneurisation Docker complète
- Migration Prefect → Apache Airflow
- Pipeline de ré-entraînement automatisé

---
