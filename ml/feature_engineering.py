import pandas as pd
import numpy as np
from ingestion.db_config import get_engine

# ══════════════════════════════════════════════════════
# 1. CHARGEMENT
# ══════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    print("Chargement depuis PostgreSQL...")
    engine = get_engine()

    df = pd.read_sql("SELECT * FROM training_tir_export", engine)

    # Parser les dates
    date_cols = [
        "date_zre", "date_impression", "entree_couloir",
        "entree_scanner_export", "debut_visite_physique_export",
        "fin_visite_physique_export", "sortie_sas_export",
        "entree_terminal_export", "date_embarquement",
    ]
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)

    print(f" {len(df):,} lignes chargées")
    return df


# ══════════════════════════════════════════════════════
# 2. FEATURES DE BASE — disponibles dès le ZRE
# ══════════════════════════════════════════════════════
def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features connues au moment où le camion arrive au ZRE.
    Zéro leakage — aucune info sur ce qui va se passer après.
    """
    feat = pd.DataFrame()

    # ── Cible ──────────────────────────────────────────
    feat["transit_time_h"]     = df["transit_time_h"]
    feat["log_transit_time_h"] = np.log1p(df["transit_time_h"])

    # ── Temporelles (depuis date_zre) ──────────────────
    feat["heure_zre"]    = df["date_zre"].dt.hour
    feat["jour_semaine"] = df["date_zre"].dt.dayofweek  # 0=Lundi, 6=Dimanche
    feat["mois_zre"]     = df["date_zre"].dt.month
    feat["est_weekend"]  = (feat["jour_semaine"] >= 5).astype(int)
    feat["est_nuit"]     = ((feat["heure_zre"] >= 22) | (feat["heure_zre"] <= 6)).astype(int)

    # ── Couloir (encodage ordinal — ordre = vitesse médiane EDA) ──
    # Couloir 5 (le + rapide) → 1, Couloir 3 (le + lent) → 5
    couloir_map = {
        "Couloir 5": 1,
        "Couloir 4": 2,
        "Couloir 2": 3,
        "Couloir 1": 4,
        "Couloir 3": 5,
    }
    feat["couloir_enc"] = df["couloir"].map(couloir_map).fillna(3)

    # ── Type d'unité (encodage ordinal — ordre = vitesse médiane EDA) ──
    # Camion (le + rapide) → 1, Semi Remorque (le + lent) → 5
    type_map = {
        "Camion 12m ou moins": 1,
        "Tracteur":            2,
        "Plateau":             3,
        "Ensemble Routier":    4,
        "Semi Remorque":       5,
    }
    feat["type_unite_enc"] = df["ss_type_unite"].map(type_map).fillna(4)

    # ── Vide / Plein ──────────────────────────────────
    feat["est_plein"] = (df["vide_plein"] == "PLEIN").astype(int)

    # ── Nature marchandise (encodage ordinal — ordre = vitesse médiane EDA) ──
    nature_map = {
        "Machine / Appareils Electriques":                                     1,
        "Meuble":                                                              2,
        "Matériel de transport (ferroviaire, terrestre, maritime et aérien)":  3,
        "Ouvrages divers":                                                     4,
        "Légumes et fruits":                                                   5,
        "Matières plastiques et caoutchouc":                                   6,
        "Métaux et ouvrages associés":                                         7,
        "Textile / Confection":                                                8,
        "Agroalimentaire":                                                     9,
        "Produits de la mer":                                                 10,
    }
    feat["nature_marc_enc"] = df["nature_marchandise"].map(nature_map)
    # Valeurs manquantes (5.5%) → médiane
    feat["nature_marc_enc"] = feat["nature_marc_enc"].fillna(
        feat["nature_marc_enc"].median()
    )

    # ── Poids ─────────────────────────────────────────
    # Valeurs manquantes (5.5%) → médiane par type d'unité
    feat["poids"] = df["poids"].copy()
    mediane_poids = feat.groupby("type_unite_enc")["poids"].transform("median")
    feat["poids"] = feat["poids"].fillna(mediane_poids)
    # Si encore NaN (type inconnu) → médiane globale
    feat["poids"] = feat["poids"].fillna(feat["poids"].median())

    # ── Groupage ──────────────────────────────────────
    # 82.8% vide → flag binaire (vide = pas groupé)
    feat["is_groupage"] = df["groupage"].notna().astype(int)

    # ── Matière dangereuse ────────────────────────────
    feat["is_danger"] = (df["matiere_danger"] == "true").astype(int)

    print(f" Features de base : {feat.shape[1]} colonnes")
    return feat


# ══════════════════════════════════════════════════════
# 3. FEATURES CHECKPOINT 2 — après scanner
# ══════════════════════════════════════════════════════
def add_scanner_features(df: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    """
    Features disponibles après que le camion sort du scanner.
    """
    feat = feat.copy()

    # A eu un scanner ? (1 = oui, 0 = non)
    feat["a_eu_scanner"] = df["entree_scanner_export"].notna().astype(int)

    # Durée passée au scanner (heures)
    # fin_visite_physique_export utilisée comme proxy de fin scanner si dispo
    duree_scanner = (
        df["fin_visite_physique_export"] - df["entree_scanner_export"]
    ).dt.total_seconds() / 3600

    # Si pas eu de scanner → durée = 0
    feat["duree_scanner"] = duree_scanner.fillna(0).clip(lower=0, upper=24)

    # Temps écoulé depuis ZRE jusqu'à fin scanner (ou fin impression si pas scanner)
    temps_ecoule = (
        df["fin_visite_physique_export"].fillna(df["date_impression"]) - df["date_zre"]
    ).dt.total_seconds() / 3600
    feat["temps_ecoule_apres_scanner"] = temps_ecoule.fillna(0).clip(lower=0, upper=48)

    print(f" Features scanner ajoutées — total : {feat.shape[1]} colonnes")
    return feat


# ══════════════════════════════════════════════════════
# 4. FEATURES CHECKPOINT 3 — après SAS export
# ══════════════════════════════════════════════════════
def add_sas_features(df: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    """
    Features disponibles après que le camion sort du SAS export.
    """
    feat = feat.copy()

    # A eu une visite physique ?
    feat["a_eu_visite_physique"] = df["debut_visite_physique_export"].notna().astype(int)

    # Durée de la visite physique (heures)
    duree_visite = (
        df["fin_visite_physique_export"] - df["debut_visite_physique_export"]
    ).dt.total_seconds() / 3600
    feat["duree_visite_physique"] = duree_visite.fillna(0).clip(lower=0, upper=24)

    # Temps écoulé depuis ZRE jusqu'à sortie SAS
    temps_sas = (
        df["sortie_sas_export"] - df["date_zre"]
    ).dt.total_seconds() / 3600
    feat["temps_ecoule_apres_sas"] = temps_sas.fillna(
        feat["temps_ecoule_apres_scanner"]
    ).clip(lower=0, upper=72)

    print(f" Features SAS ajoutées — total : {feat.shape[1]} colonnes")
    return feat


# ══════════════════════════════════════════════════════
# 5. FEATURES CHECKPOINT 4 — après entrée terminal
# ══════════════════════════════════════════════════════
def add_terminal_features(df: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    """
    Features disponibles après que le camion entre au terminal.
    """
    feat = feat.copy()

    # Temps écoulé depuis ZRE jusqu'à entrée terminal
    temps_terminal = (
        df["entree_terminal_export"] - df["date_zre"]
    ).dt.total_seconds() / 3600
    feat["temps_ecoule_apres_terminal"] = temps_terminal.fillna(
        feat["temps_ecoule_apres_sas"]
    ).clip(lower=0, upper=120)

    # Durée dans le terminal (avant embarquement)
    # = transit_time_h - temps_jusqu_au_terminal
    feat["duree_restante_estimee"] = (
        feat["transit_time_h"] - feat["temps_ecoule_apres_terminal"]
    ).clip(lower=0)

    print(f" Features terminal ajoutées — total : {feat.shape[1]} colonnes")
    return feat


# ══════════════════════════════════════════════════════
# 6. SAUVEGARDE
# ══════════════════════════════════════════════════════
def save_features(feat_m1, feat_m2, feat_m3, feat_m4):
    import os
    os.makedirs("ml/features", exist_ok=True)

    feat_m1.to_parquet("ml/features/features_m1.parquet", index=False)
    feat_m2.to_parquet("ml/features/features_m2.parquet", index=False)
    feat_m3.to_parquet("ml/features/features_m3.parquet", index=False)
    feat_m4.to_parquet("ml/features/features_m4.parquet", index=False)

    print("\n Features sauvegardées :")
    print(f"  M1 (ZRE)       : {feat_m1.shape[1]} features, {len(feat_m1):,} lignes")
    print(f"  M2 (Scanner)   : {feat_m2.shape[1]} features, {len(feat_m2):,} lignes")
    print(f"  M3 (SAS)       : {feat_m3.shape[1]} features, {len(feat_m3):,} lignes")
    print(f"  M4 (Terminal)  : {feat_m4.shape[1]} features, {len(feat_m4):,} lignes")


# ══════════════════════════════════════════════════════
# 7. DIAGNOSTIC — vérifier la qualité des features
# ══════════════════════════════════════════════════════
def diagnostic_features(feat: pd.DataFrame, nom: str):
    print(f"\n--- Diagnostic {nom} ---")
    pct_null = feat.isnull().mean()
    problemes = pct_null[pct_null > 0]
    if len(problemes):
        print(f"  Colonnes avec NaN restants :")
        for col, pct in problemes.items():
            print(f"    {col:40s} {pct:.1%}")
    else:
        print(f"   Aucun NaN — dataset propre")

    print(f"  Shape : {feat.shape}")


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  FEATURE ENGINEERING — 4 checkpoints")
    print("=" * 55)

    # Charger les données
    df = load_data()

    # Construire les features par checkpoint
    feat_base = build_base_features(df)

    feat_m1 = feat_base.copy()
    feat_m2 = add_scanner_features(df, feat_base)
    feat_m3 = add_sas_features(df, feat_m2)
    feat_m4 = add_terminal_features(df, feat_m3)

    # Diagnostics
    diagnostic_features(feat_m1, "Modèle 1 — ZRE")
    diagnostic_features(feat_m2, "Modèle 2 — Scanner")
    diagnostic_features(feat_m3, "Modèle 3 — SAS")
    diagnostic_features(feat_m4, "Modèle 4 — Terminal")

    # Sauvegarder
    save_features(feat_m1, feat_m2, feat_m3, feat_m4)

    print("\n Feature engineering terminé")
    print("   Prochaine étape : python -m ml.train")