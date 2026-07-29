import pandas as pd

PLACEHOLDERS = {"/", "_", "", "NULL", "null", "NaN"}

DATE_COLS = [
    "date_creation", "date_validation", "date_operation", "date_impression",
    "date_zre", "entree_park_visite", "entree_couloir", "entree_scanner_export",
    "debut_visite_physique_export", "fin_visite_physique_export",
    "sortie_sas_export", "entree_terminal_export", "entree_accroche_decroche",
    "sortie_accroche_decroche", "embarquement_export", "date_embarquement",
    "date_send", "date_annulation", "date_commande_tr", "date_ordre_tr",
    "eta", "etd", "rta", "rtd",
]


def fix_mojibake(s):
    if not isinstance(s, str):
        return s
    if "Ã" in s:
        try:
            return s.encode("latin1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            return s
    return s


def safe_strip(s):
    return s.strip() if isinstance(s, str) else s


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le DataFrame brut — appelé par etl_flow.py"""

    # 1. Réparer mojibake + strip sur colonnes texte
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        df[c] = df[c].map(fix_mojibake)
        df[c] = df[c].map(safe_strip)

    # 2. Placeholders → NaN
    df = df.replace(list(PLACEHOLDERS), pd.NA)

    # 3. Parser les dates
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)

    # 4. Colonnes numériques
    for c in ["amp_id", "poids", "nb_colis", "year", "month", "day"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 5. Dédupliquer par version
    if "version" in df.columns and "no_amp" in df.columns:
        df["version"] = pd.to_numeric(df["version"], errors="coerce")
        df = df.sort_values("version").drop_duplicates("no_amp", keep="last")

    # 6. Exclure Annulé / Rejeté
    if "statut_exploitation" in df.columns:
        mask_exclu = df["statut_exploitation"].isin(["Annulé", "Rejeté"])
        print(f"Dossiers annulés/rejetés exclus : {mask_exclu.sum()}")
        df = df[~mask_exclu]

    return df


def build_training_set(df: pd.DataFrame) -> pd.DataFrame:
    """Construit le dataset final ML avec la cible transit_time_h"""

    # Garder uniquement Clôturé + Embarqué
    pop = df[df["statut_exploitation"].isin(["Clôturé", "Embarqué"])].copy()
    print(f"Clôturé + Embarqué : {len(pop):,}")

    # Garder uniquement les dossiers avec date_zre ET date_embarquement
    pop = pop[pop["date_zre"].notna() & pop["date_embarquement"].notna()]
    print(f"Avec date_zre ET date_embarquement : {len(pop):,}")

    # Calculer la cible
    pop["transit_time_h"] = (
        pop["date_embarquement"] - pop["date_zre"]
    ).dt.total_seconds() / 3600

    # Filtrer outliers
    pop = pop[(pop["transit_time_h"] > 0) & (pop["transit_time_h"] < 72)]
    print(f"Après filtre outliers : {len(pop):,}")

    # Supprimer colonnes 100% vides
    fully_empty = [c for c in pop.columns if pop[c].notna().sum() == 0]
    if fully_empty:
        print(f"Colonnes 100% vides supprimées : {fully_empty}")
        pop = pop.drop(columns=fully_empty)

    return pop