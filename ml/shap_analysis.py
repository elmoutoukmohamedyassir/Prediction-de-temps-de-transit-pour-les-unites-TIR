"""
shap_analysis.py
Analyse SHAP du modèle final (XGBoost tuné, checkpoint M4_Terminal).
Génère un summary plot et un bar plot d'importance des features,
sauvegardés en PNG pour le rapport/la présentation.
Usage : python -m ml.shap_analysis
"""

import os
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

FEATURES_DIR = "ml/features"
MODELS_DIR   = "ml/models"
OUTPUT_DIR   = "ml/shap_outputs"

MODEL_PATH   = os.path.join(MODELS_DIR, "xgb_M4_Terminal_tuned.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "xgb_M4_Terminal_tuned_features.pkl")
DATASET_PATH = os.path.join(FEATURES_DIR, "features_m4.parquet")

# Nombre de lignes utilisées pour SHAP — un échantillon suffit,
# pas besoin de tout le test set (SHAP peut être lent sur de gros volumes)
SAMPLE_SIZE = 5000


def temporal_split(df: pd.DataFrame, test_size: float = 0.2):
    n = len(df)
    split_idx = int(n * (1 - test_size))
    return df.iloc[:split_idx], df.iloc[split_idx:]


if __name__ == "__main__":
    print("=" * 55)
    print("  ANALYSE SHAP — XGBoost M4_Terminal (tuné)")
    print("=" * 55)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Charger le modèle et la liste exacte des features (même ordre
    # que l'entraînement — critique pour que SHAP interprète les
    # bonnes colonnes)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        feature_cols = pickle.load(f)

    print(f"Modèle chargé : {MODEL_PATH}")
    print(f"Features ({len(feature_cols)}) : {feature_cols}")

    # Charger le dataset, reproduire le même split que l'entraînement
    df = pd.read_parquet(DATASET_PATH)
    _, test_df = temporal_split(df)

    X_test = test_df[feature_cols]

    # Échantillonnage pour accélérer le calcul SHAP si le test set est gros
    if len(X_test) > SAMPLE_SIZE:
        X_sample = X_test.sample(n=SAMPLE_SIZE, random_state=42)
    else:
        X_sample = X_test

    print(f"Calcul SHAP sur un échantillon de {len(X_sample):,} lignes...")

    # TreeExplainer : algorithme exact et rapide pour les modèles d'arbres
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # ── Summary plot (vue détaillée) ────────────────────────
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    summary_path = os.path.join(OUTPUT_DIR, "shap_summary_plot.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Summary plot sauvegardé → {summary_path}")

    # ── Bar plot d'importance (vue simplifiée, pour l'encadrant) ──
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, "shap_bar_importance.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" Bar plot d'importance sauvegardé → {bar_path}")

    # ── Classement texte (pour référence rapide / rapport) ──────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    print("\n" + "=" * 55)
    print("  CLASSEMENT DES FEATURES PAR IMPORTANCE SHAP")
    print("=" * 55)
    for _, row in importance_df.iterrows():
        print(f"  {row['feature']:35s} {row['mean_abs_shap']:.4f}")

    importance_df.to_csv(os.path.join(OUTPUT_DIR, "shap_importance_ranking.csv"), index=False)
    print(f"\n Classement sauvegardé → {OUTPUT_DIR}/shap_importance_ranking.csv")