"""
etl_flow.py
Pipeline ETL orchestré par Prefect.
raw_tir_export → clean_tir_export → training_tir_export

Usage :
    python -m pipeline.etl_flow
"""

import time
import pandas as pd
from prefect import flow, task, get_run_logger
from sqlalchemy import text

from ingestion.db_config import get_engine
from pipeline.clean import clean, build_training_set

CHUNKSIZE = 10_000


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — EXTRACT
# Lit raw_tir_export depuis PostgreSQL
# ══════════════════════════════════════════════════════════════════════════════
@task(name="Extract raw data", retries=2, retry_delay_seconds=10, log_prints=True)
def extract() -> pd.DataFrame:
    logger = get_run_logger()
    logger.info("Lecture de raw_tir_export depuis PostgreSQL...")

    engine = get_engine()
    start  = time.time()

    df = pd.read_sql("SELECT * FROM raw_tir_export", engine)

    elapsed = time.time() - start
    logger.info(f"✅ {len(df):,} lignes × {df.shape[1]} colonnes chargées en {elapsed:.1f}s")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — TRANSFORM
# Applique toutes les fonctions de clean.py
# ══════════════════════════════════════════════════════════════════════════════
@task(name="Transform / Clean data", retries=1, log_prints=True)
def transform(df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info(f"Nettoyage de {len(df):,} lignes...")

    df_clean = clean(df)

    logger.info(f"✅ Après nettoyage : {len(df_clean):,} lignes")
    return df_clean


# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 — LOAD CLEAN
# Écrit df_clean dans la table clean_tir_export
# ══════════════════════════════════════════════════════════════════════════════
@task(name="Load clean data → PostgreSQL", retries=2, retry_delay_seconds=10, log_prints=True)
def load_clean(df: pd.DataFrame) -> None:
    logger  = get_run_logger()
    engine  = get_engine()
    table   = "clean_tir_export"
    n_total = len(df)

    logger.info(f"Écriture de {n_total:,} lignes dans '{table}'...")
    start = time.time()

    for i, chunk_start in enumerate(range(0, n_total, CHUNKSIZE)):
        chunk = df.iloc[chunk_start : chunk_start + CHUNKSIZE]
        mode  = "replace" if i == 0 else "append"
        chunk.to_sql(name=table, con=engine, if_exists=mode, index=False)

    elapsed = time.time() - start

    # Vérification
    with engine.connect() as conn:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()

    logger.info(f"✅ {count:,} lignes dans '{table}' — {elapsed:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 4 — BUILD TRAINING SET
# Filtre df_clean et calcule transit_time_h
# ══════════════════════════════════════════════════════════════════════════════
@task(name="Build training dataset", retries=1, log_prints=True)
def build_training(df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()
    logger.info("Construction du dataset d'entraînement ML...")

    df_training = build_training_set(df)

    logger.info(f"✅ Dataset ML : {len(df_training):,} lignes × {df_training.shape[1]} colonnes")
    logger.info(f"   transit_time_h — médiane : {df_training['transit_time_h'].median():.1f}h  "
                f"moyenne : {df_training['transit_time_h'].mean():.1f}h")
    return df_training


# ══════════════════════════════════════════════════════════════════════════════
# TASK 5 — LOAD TRAINING
# Écrit df_training dans training_tir_export
# ══════════════════════════════════════════════════════════════════════════════
@task(name="Load training dataset → PostgreSQL", retries=2, retry_delay_seconds=10, log_prints=True)
def load_training(df: pd.DataFrame) -> None:
    logger  = get_run_logger()
    engine  = get_engine()
    table   = "training_tir_export"
    n_total = len(df)

    logger.info(f"Écriture de {n_total:,} lignes dans '{table}'...")
    start = time.time()

    for i, chunk_start in enumerate(range(0, n_total, CHUNKSIZE)):
        chunk = df.iloc[chunk_start : chunk_start + CHUNKSIZE]
        mode  = "replace" if i == 0 else "append"
        chunk.to_sql(name=table, con=engine, if_exists=mode, index=False)

    elapsed = time.time() - start

    # Vérification
    with engine.connect() as conn:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()

    logger.info(f"✅ {count:,} lignes dans '{table}' — {elapsed:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 6 — LOG STATS
# Résumé final dans les logs Prefect
# ══════════════════════════════════════════════════════════════════════════════
@task(name="Log final stats", log_prints=True)
def log_stats(df_clean: pd.DataFrame, df_training: pd.DataFrame) -> None:
    logger = get_run_logger()

    logger.info("=" * 50)
    logger.info("RÉSUMÉ PIPELINE ETL")
    logger.info("=" * 50)
    logger.info(f"raw_tir_export      → {df_clean.shape[0]:,} lignes (après dédup/exclusion)")
    logger.info(f"clean_tir_export    → {df_clean.shape[0]:,} lignes")
    logger.info(f"training_tir_export → {df_training.shape[0]:,} lignes")
    logger.info(f"Colonnes ML         → {df_training.shape[1]}")
    logger.info(f"transit_time_h :")
    logger.info(f"  min    : {df_training['transit_time_h'].min():.1f}h")
    logger.info(f"  médiane: {df_training['transit_time_h'].median():.1f}h")
    logger.info(f"  moyenne: {df_training['transit_time_h'].mean():.1f}h")
    logger.info(f"  max    : {df_training['transit_time_h'].max():.1f}h")
    logger.info(f"  std    : {df_training['transit_time_h'].std():.1f}h")
    logger.info("=" * 50)
    logger.info("✅ Pipeline ETL terminé avec succès")


# ══════════════════════════════════════════════════════════════════════════════
# FLOW PRINCIPAL
# Orchestre les 6 tâches dans l'ordre
# ══════════════════════════════════════════════════════════════════════════════
@flow(
    name="Tanger Med ETL Pipeline",
    description="raw_tir_export → clean_tir_export → training_tir_export",
)
def etl_pipeline() -> None:
    # Task 1 — lire depuis PostgreSQL
    df_raw = extract()

    # Task 2 — nettoyer
    df_clean = transform(df_raw)

    # Task 3 — écrire clean dans PostgreSQL
    load_clean(df_clean)

    # Task 4 — construire le dataset ML
    df_training = build_training(df_clean)

    # Task 5 — écrire training dans PostgreSQL
    load_training(df_training)

    # Task 6 — résumé final
    log_stats(df_clean, df_training)


if __name__ == "__main__":
    etl_pipeline()