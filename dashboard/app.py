"""
dashboard/app.py
Tableau de bord interactif – Prédiction du temps de transit TIR Export
Tanger Med.
Conforme au rapport (section 7.6) : formulaire camion → prédiction
instantanée → positionnement par rapport à l'historique.

Usage :
    streamlit run dashboard/app.py
"""

import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────────
API_URL = "http://127.0.0.1:8000"

COULOIRS       = ["Couloir 1", "Couloir 2", "Couloir 3", "Couloir 4", "Couloir 5"]
TYPES_UNITE    = ["Camion 12m ou moins", "Tracteur", "Plateau", "Ensemble Routier", "Semi Remorque"]
NATURES        = [
    "Machine / Appareils Electriques",
    "Meuble",
    "Matériel de transport (ferroviaire, terrestre, maritime et aérien)",
    "Ouvrages divers",
    "Légumes et fruits",
    "Matières plastiques et caoutchouc",
    "Métaux et ouvrages associés",
    "Textile / Confection",
    "Agroalimentaire",
    "Produits de la mer",
]

# Médianes historiques par couloir (issues de l'EDA)
MEDIANES_HISTORIQUES = {
    "Couloir 5": 16.5,
    "Couloir 4": 17.1,
    "Couloir 2": 18.8,
    "Couloir 1": 19.7,
    "Couloir 3": 20.0,
}

CHECKPOINT_LABELS = {
    1: "M1 — Arrivée au ZRE",
    2: "M2 — Après scanner",
    3: "M3 — Après SAS export",
    4: "M4 — Après entrée terminal",
}

# ── Mise en page ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tanger Med — Transit TIR",
    
    layout="wide",
)

st.title("Prédiction du temps de transit TIR Export")
st.caption("Port Tanger Med — Système de Machine Learning (XGBoost, 4 checkpoints)")
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Formulaire
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("📋 Informations du dossier")

    checkpoint = st.selectbox(
        "Checkpoint de prédiction",
        options=[1, 2, 3, 4],
        format_func=lambda x: CHECKPOINT_LABELS[x],
        help="Sélectionne le stade du parcours physique du camion",
    )

    st.subheader("Caractéristiques du camion")
    couloir         = st.selectbox("Couloir douanier", COULOIRS, index=3)
    ss_type_unite   = st.selectbox("Type d'unité", TYPES_UNITE, index=3)
    vide_plein      = st.radio("Chargement", ["PLEIN", "VIDE"], horizontal=True)
    nature_march    = st.selectbox("Nature de la marchandise", NATURES, index=7)
    poids           = st.number_input("Poids (kg)", min_value=0.0, value=12000.0, step=500.0)
    is_groupage     = st.checkbox("Envoi groupé (groupage)")
    matiere_danger  = st.checkbox("Matière dangereuse")
    date_zre        = st.datetime_input(
        "Date/heure d'arrivée au ZRE",
        value=datetime(2026, 6, 21, 8, 0),
    )

    # Champs conditionnels selon le checkpoint
    a_eu_scanner                = None
    duree_scanner               = None
    temps_ecoule_apres_scanner  = None
    a_eu_visite_physique        = None
    duree_visite_physique       = None
    temps_ecoule_apres_sas      = None
    temps_ecoule_apres_terminal = None

    if checkpoint >= 2:
        st.subheader(" Données scanner")
        a_eu_scanner = int(st.checkbox("Passé au scanner", value=True))
        duree_scanner = st.number_input("Durée scanner (h)", min_value=0.0, value=1.5, step=0.25)
        temps_ecoule_apres_scanner = st.number_input(
            "Temps écoulé depuis ZRE (h)", min_value=0.0, value=2.0, step=0.25
        )

    if checkpoint >= 3:
        st.subheader(" Données SAS export")
        a_eu_visite_physique = int(st.checkbox("Visite physique effectuée", value=False))
        duree_visite_physique = st.number_input(
            "Durée visite physique (h)", min_value=0.0, value=0.0, step=0.25
        )
        temps_ecoule_apres_sas = st.number_input(
            "Temps écoulé depuis ZRE (h)", min_value=0.0, value=4.5, step=0.25, key="sas"
        )

    if checkpoint == 4:
        st.subheader(" Données terminal")
        temps_ecoule_apres_terminal = st.number_input(
            "Temps écoulé depuis ZRE (h)", min_value=0.0, value=7.0, step=0.25, key="terminal"
        )

    predict_btn = st.button(" Prédire le temps de transit", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# COLONNES PRINCIPALES
# ══════════════════════════════════════════════════════════════════════════════

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader(" Résumé du dossier")

    st.markdown(f"""
    | Paramètre | Valeur |
    |---|---|
    | Couloir | {couloir} |
    | Type d'unité | {ss_type_unite} |
    | Chargement | {vide_plein} |
    | Marchandise | {nature_march} |
    | Poids | {poids:,.0f} kg |
    | Groupage | {" Oui" if is_groupage else " Non"} |
    | Matière dangereuse | {"Oui" if matiere_danger else " Non"} |
    | Date ZRE | {date_zre.strftime("%d/%m/%Y %H:%M")} |
    | **Checkpoint** | **{CHECKPOINT_LABELS[checkpoint]}** |
    """)

    # Référence naïve (médiane historique du couloir)
    ref_naive = MEDIANES_HISTORIQUES.get(couloir, 18.0)
    st.info(f" Référence naïve (médiane historique {couloir}) : **{ref_naive:.1f}h**")


with col2:
    st.subheader(" Prédiction ML")

    if predict_btn:
        # Construction du body
        body = {
            "couloir":            couloir,
            "ss_type_unite":      ss_type_unite,
            "vide_plein":         vide_plein,
            "nature_marchandise": nature_march,
            "poids":              poids,
            "is_groupage":        int(is_groupage),
            "matiere_danger":     matiere_danger,
            "date_zre":           date_zre.isoformat(),
            "checkpoint":         checkpoint,
        }

        if checkpoint >= 2:
            body["a_eu_scanner"]               = a_eu_scanner
            body["duree_scanner"]              = duree_scanner
            body["temps_ecoule_apres_scanner"] = temps_ecoule_apres_scanner

        if checkpoint >= 3:
            body["a_eu_visite_physique"]   = a_eu_visite_physique
            body["duree_visite_physique"]  = duree_visite_physique
            body["temps_ecoule_apres_sas"] = temps_ecoule_apres_sas

        if checkpoint == 4:
            body["temps_ecoule_apres_terminal"] = temps_ecoule_apres_terminal

        try:
            resp = requests.post(f"{API_URL}/predict", json=body, timeout=10)
            resp.raise_for_status()
            result = resp.json()

            predicted = result["transit_time_h_predicted"]
            model_used = result["model_used"]

            # Affichage principal
            st.metric(
                label="Temps de transit prédit",
                value=f"{predicted:.1f} heures",
                delta=f"{predicted - ref_naive:+.1f}h vs référence naïve",
                delta_color="inverse",
            )

            # Interprétation
            h = int(predicted)
            m = int((predicted - h) * 60)
            st.success(f" Soit environ **{h}h{m:02d}** depuis le ZRE")

            # Modèle utilisé
            st.caption(f"Modèle : `{model_used}` | Checkpoint : {CHECKPOINT_LABELS[checkpoint]}")

            # Graphique de positionnement historique
            st.subheader(" Positionnement par rapport à l'historique")

            hist_data = {
                "Couloir 5": 16.5,
                "Couloir 4": 17.1,
                "Couloir 2": 18.8,
                "Couloir 1": 19.7,
                "Couloir 3": 20.0,
            }

            fig, ax = plt.subplots(figsize=(7, 3))
            couloirs_labels = list(hist_data.keys())
            medianes = list(hist_data.values())
            colors = ["#2196F3" if c != couloir else "#FF5722" for c in couloirs_labels]

            bars = ax.barh(couloirs_labels, medianes, color=colors, edgecolor="white", height=0.5)
            ax.axvline(
                predicted,
                color="#4CAF50",
                linestyle="--",
                linewidth=2,
                label=f"Prédiction : {predicted:.1f}h",
            )
            ax.set_xlabel("Médiane temps de transit (h)")
            ax.set_title("Médiane historique par couloir vs prédiction")
            ax.legend()
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        except requests.exceptions.ConnectionError:
            st.error(
                " Impossible de joindre l'API FastAPI. "
                "Vérifiez que le serveur tourne sur http://127.0.0.1:8000\n\n"
                "`uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload`"
            )
        except requests.exceptions.HTTPError as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                pass
            st.error(f" Erreur API : {e}\n\n{detail}")
        except Exception as e:
            st.error(f" Erreur inattendue : {e}")

    else:
        st.info("Renseigne les informations du dossier dans le panneau gauche, puis clique sur **Prédire**.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION INFO — Architecture multi-checkpoints
# ══════════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("ℹ Architecture multi-checkpoints")

cols = st.columns(4)
checkpoint_data = [
    ("M1 – ZRE", "8,71h", "44,3%", "0,052"),
    ("M2 – Scanner", "8,27h", "42,2%", "0,138"),
    ("M3 – SAS", "7,53h", "36,7%", "0,212"),
    ("M4 – Terminal", "6,30h", "29,8%", "0,371"),
]
for col, (label, mae, mape, r2) in zip(cols, checkpoint_data):
    with col:
        st.metric(label=label, value=mae, help=f"MAPE={mape} | R²={r2}")
        st.caption(f"MAPE : {mape} | R² : {r2}")

st.caption(
    "Baseline naïve (médiane par couloir) : MAE=9,54h, MAPE=53,8% — "
    "le modèle ML réduit l'erreur relative de **45%**."
)