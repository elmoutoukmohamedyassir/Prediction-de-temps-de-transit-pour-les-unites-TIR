"""
api/main.py
API REST de prédiction du temps de transit TIR export – Tanger Med.
Conforme à la spécification du rapport (section 7.5).

Endpoints :
    GET  /           → health check
    GET  /info       → infos sur les modèles chargés
    POST /predict    → prédiction selon le checkpoint

Usage :
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import pickle
import numpy as np
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ══════════════════════════════════════════════════════════════════════════════
# ENCODAGES — identiques à ml/feature_engineering.py
# ══════════════════════════════════════════════════════════════════════════════

COULOIR_MAP = {
    "Couloir 5": 1,
    "Couloir 4": 2,
    "Couloir 2": 3,
    "Couloir 1": 4,
    "Couloir 3": 5,
}

TYPE_MAP = {
    "Camion 12m ou moins": 1,
    "Tracteur":            2,
    "Plateau":             3,
    "Ensemble Routier":    4,
    "Semi Remorque":       5,
}

NATURE_MAP = {
    "Machine / Appareils Electriques":                                    1,
    "Meuble":                                                             2,
    "Matériel de transport (ferroviaire, terrestre, maritime et aérien)": 3,
    "Ouvrages divers":                                                    4,
    "Légumes et fruits":                                                  5,
    "Matières plastiques et caoutchouc":                                  6,
    "Métaux et ouvrages associés":                                        7,
    "Textile / Confection":                                               8,
    "Agroalimentaire":                                                    9,
    "Produits de la mer":                                                10,
}

MODELS_DIR = "ml/models"

# Mapping checkpoint → fichier modèle + liste de features
# M1, M2, M3 : XGBoost par défaut  |  M4 : XGBoost optimisé Optuna
CHECKPOINT_CONFIG = {
    1: {
        "model_file":   "xgb_M1_ZRE.pkl",
        "model_name":   "xgb_M1_ZRE",
        "features": [
            "heure_zre", "jour_semaine", "mois_zre", "est_weekend", "est_nuit",
            "couloir_enc", "type_unite_enc", "est_plein",
            "nature_marc_enc", "poids", "is_groupage", "is_danger",
        ],
    },
    2: {
        "model_file":   "xgb_M2_Scanner.pkl",
        "model_name":   "xgb_M2_Scanner",
        "features": [
            "heure_zre", "jour_semaine", "mois_zre", "est_weekend", "est_nuit",
            "couloir_enc", "type_unite_enc", "est_plein",
            "nature_marc_enc", "poids", "is_groupage", "is_danger",
            "a_eu_scanner", "duree_scanner", "temps_ecoule_apres_scanner",
        ],
    },
    3: {
        "model_file":   "xgb_M3_SAS.pkl",
        "model_name":   "xgb_M3_SAS",
        "features": [
            "heure_zre", "jour_semaine", "mois_zre", "est_weekend", "est_nuit",
            "couloir_enc", "type_unite_enc", "est_plein",
            "nature_marc_enc", "poids", "is_groupage", "is_danger",
            "a_eu_scanner", "duree_scanner", "temps_ecoule_apres_scanner",
            "a_eu_visite_physique", "duree_visite_physique", "temps_ecoule_apres_sas",
        ],
    },
    4: {
        "model_file":    "xgb_M4_Terminal_tuned.pkl",
        "features_file": "xgb_M4_Terminal_tuned_features.pkl",
        "model_name":    "xgb_M4_Terminal_tuned",
        "features": [
            "heure_zre", "jour_semaine", "mois_zre", "est_weekend", "est_nuit",
            "couloir_enc", "type_unite_enc", "est_plein",
            "nature_marc_enc", "poids", "is_groupage", "is_danger",
            "a_eu_scanner", "duree_scanner", "temps_ecoule_apres_scanner",
            "a_eu_visite_physique", "duree_visite_physique", "temps_ecoule_apres_sas",
            "temps_ecoule_apres_terminal",
        ],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES MODÈLES au démarrage
# ══════════════════════════════════════════════════════════════════════════════

def _load_models() -> dict:
    loaded = {}
    for cp, cfg in CHECKPOINT_CONFIG.items():
        path = os.path.join(MODELS_DIR, cfg["model_file"])
        if os.path.exists(path):
            with open(path, "rb") as f:
                loaded[cp] = pickle.load(f)
            print(f"✅ Checkpoint M{cp} chargé : {cfg['model_file']}")
        else:
            print(f"⚠️  Checkpoint M{cp} introuvable : {path}")
    return loaded


MODELS = _load_models()


# ══════════════════════════════════════════════════════════════════════════════
# SCHÉMAS PYDANTIC
# ══════════════════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    # ── Features disponibles au ZRE (obligatoires pour tous les checkpoints) ──
    couloir: str = Field(..., example="Couloir 4")
    ss_type_unite: str = Field(..., example="Ensemble Routier")
    vide_plein: str = Field(..., example="PLEIN")
    nature_marchandise: Optional[str] = Field(None, example="Textile / Confection")
    poids: Optional[float] = Field(None, example=12000.0)
    is_groupage: int = Field(0, example=0, description="1 = envoi groupé, 0 = individuel")
    matiere_danger: bool = Field(False, example=False)
    date_zre: str = Field(..., example="2026-06-21T08:00:00+01:00")
    checkpoint: int = Field(..., example=1, ge=1, le=4)

    # ── Features post-scanner (obligatoires si checkpoint >= 2) ──
    a_eu_scanner: Optional[int] = Field(None, example=1)
    duree_scanner: Optional[float] = Field(None, example=1.5)
    temps_ecoule_apres_scanner: Optional[float] = Field(None, example=2.0)

    # ── Features post-SAS (obligatoires si checkpoint >= 3) ──
    a_eu_visite_physique: Optional[int] = Field(None, example=0)
    duree_visite_physique: Optional[float] = Field(None, example=0.0)
    temps_ecoule_apres_sas: Optional[float] = Field(None, example=4.5)

    # ── Features post-terminal (obligatoires si checkpoint == 4) ──
    temps_ecoule_apres_terminal: Optional[float] = Field(None, example=7.0)


class PredictResponse(BaseModel):
    transit_time_h_predicted: float
    checkpoint: int
    model_used: str
    couloir: str
    ss_type_unite: str
    date_zre: str


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING — même logique que ml/feature_engineering.py
# ══════════════════════════════════════════════════════════════════════════════

def build_features(req: PredictRequest) -> dict:
    """Transforme la requête brute en dict de features numériques."""
    try:
        dt = datetime.fromisoformat(req.date_zre)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"date_zre invalide : {req.date_zre}")

    heure       = dt.hour
    jour_semaine = dt.weekday()  # 0=Lundi, 6=Dimanche
    mois        = dt.month

    couloir_enc  = COULOIR_MAP.get(req.couloir, 3)         # défaut = médian
    type_enc     = TYPE_MAP.get(req.ss_type_unite, 4)       # défaut = Ensemble Routier
    nature_enc   = NATURE_MAP.get(req.nature_marchandise, 5.5)  # défaut = médiane
    est_plein    = 1 if req.vide_plein == "PLEIN" else 0
    poids        = req.poids if req.poids is not None else 12000.0  # médiane approx.

    feat = {
        "heure_zre":    heure,
        "jour_semaine": jour_semaine,
        "mois_zre":     mois,
        "est_weekend":  int(jour_semaine >= 5),
        "est_nuit":     int(heure >= 22 or heure <= 6),
        "couloir_enc":  couloir_enc,
        "type_unite_enc": type_enc,
        "est_plein":    est_plein,
        "nature_marc_enc": nature_enc,
        "poids":        poids,
        "is_groupage":  req.is_groupage,
        "is_danger":    int(req.matiere_danger),
    }

    # Checkpoint >= 2 : features scanner
    if req.checkpoint >= 2:
        for field, default in [
            ("a_eu_scanner", 0),
            ("duree_scanner", 0.0),
            ("temps_ecoule_apres_scanner", 0.0),
        ]:
            feat[field] = getattr(req, field) if getattr(req, field) is not None else default

    # Checkpoint >= 3 : features SAS
    if req.checkpoint >= 3:
        for field, default in [
            ("a_eu_visite_physique", 0),
            ("duree_visite_physique", 0.0),
            ("temps_ecoule_apres_sas", feat.get("temps_ecoule_apres_scanner", 0.0)),
        ]:
            feat[field] = getattr(req, field) if getattr(req, field) is not None else default

    # Checkpoint 4 : features terminal
    if req.checkpoint == 4:
        feat["temps_ecoule_apres_terminal"] = (
            req.temps_ecoule_apres_terminal
            if req.temps_ecoule_apres_terminal is not None
            else feat.get("temps_ecoule_apres_sas", 0.0)
        )

    return feat


# ══════════════════════════════════════════════════════════════════════════════
# APPLICATION FASTAPI
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Tanger Med — Prédiction du temps de transit TIR Export",
    description=(
        "API de prédiction du temps de transit des camions TIR export "
        "à Tanger Med. Supporte 4 checkpoints successifs du parcours physique."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "modèles_chargés": list(MODELS.keys()),
        "checkpoints_disponibles": [f"M{cp}" for cp in MODELS],
    }


@app.get("/info", tags=["Info"])
def model_info():
    return {
        "checkpoints": {
            f"M{cp}": {
                "model_name": CHECKPOINT_CONFIG[cp]["model_name"],
                "n_features": len(CHECKPOINT_CONFIG[cp]["features"]),
                "features": CHECKPOINT_CONFIG[cp]["features"],
                "chargé": cp in MODELS,
            }
            for cp in CHECKPOINT_CONFIG
        }
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prédiction"])
def predict(req: PredictRequest):
    # Vérifier que le modèle est disponible
    if req.checkpoint not in MODELS:
        raise HTTPException(
            status_code=503,
            detail=f"Modèle checkpoint M{req.checkpoint} non disponible. "
                   f"Disponibles : {list(MODELS.keys())}",
        )

    # Validation des champs obligatoires selon le checkpoint
    if req.checkpoint >= 2 and req.a_eu_scanner is None:
        raise HTTPException(
            status_code=422,
            detail="a_eu_scanner obligatoire pour checkpoint >= 2",
        )
    if req.checkpoint >= 3 and req.temps_ecoule_apres_sas is None:
        raise HTTPException(
            status_code=422,
            detail="temps_ecoule_apres_sas obligatoire pour checkpoint >= 3",
        )
    if req.checkpoint == 4 and req.temps_ecoule_apres_terminal is None:
        raise HTTPException(
            status_code=422,
            detail="temps_ecoule_apres_terminal obligatoire pour checkpoint 4",
        )

    # Construire les features
    feat_dict = build_features(req)
    feature_cols = CHECKPOINT_CONFIG[req.checkpoint]["features"]

    # Vérifier que toutes les features attendues sont présentes
    missing = [f for f in feature_cols if f not in feat_dict]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Features manquantes dans le vecteur construit : {missing}",
        )

    # Construire le vecteur numpy dans l'ordre exact des features
    X = np.array([[feat_dict[f] for f in feature_cols]])

    # Prédiction en log-espace + reconversion en heures
    model = MODELS[req.checkpoint]
    log_pred = model.predict(X)[0]
    predicted_hours = float(np.expm1(log_pred))
    predicted_hours = max(0.0, round(predicted_hours, 2))

    return PredictResponse(
        transit_time_h_predicted=predicted_hours,
        checkpoint=req.checkpoint,
        model_used=CHECKPOINT_CONFIG[req.checkpoint]["model_name"],
        couloir=req.couloir,
        ss_type_unite=req.ss_type_unite,
        date_zre=req.date_zre,
    )