import os
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from sqlalchemy import text
from ingestion.db_config import get_engine

load_dotenv()

RAW_CSV_PATH  = os.path.join("data", "raw", "dm_amp_202606250946.csv")
TABLE_NAME    = "raw_tir_export"
CHUNKSIZE     = 10_000   # lignes par batch


def ingest_raw():
    print("=" * 55)
    print("  INGESTION CSV → PostgreSQL (table raw_tir_export)")
    print("=" * 55)

    # 1. Vérifier que le fichier existe
    if not os.path.exists(RAW_CSV_PATH):
        print(f" Fichier introuvable : {RAW_CSV_PATH}")
        return False

    
    print(f"\n Lecture du CSV : {RAW_CSV_PATH}")
    start = time.time()

    df = pd.read_csv(
        RAW_CSV_PATH,
        dtype=str,          
        low_memory=False,   
    )

    
    df.columns = [c.strip() for c in df.columns]

    # Supprimer les colonnes fantômes sans nom
    ghost_cols = [c for c in df.columns if c.startswith("Unnamed") or c == ""]
    if ghost_cols:
        print(f"  Colonnes fantômes supprimées : {ghost_cols}")
        df = df.drop(columns=ghost_cols)

    elapsed = time.time() - start
    print(f"   {len(df):,} lignes × {df.shape[1]} colonnes chargées en {elapsed:.1f}s")

    # 3. Connexion PostgreSQL
    print(f"\n🔌 Connexion à PostgreSQL...")
    engine = get_engine()

    # 4. Vérifier si la table existe déjà et combien de lignes elle a
    with engine.connect() as conn:
        table_exists = engine.dialect.has_table(conn, TABLE_NAME)
        if table_exists:
            count = conn.execute(
                text(f"SELECT COUNT(*) FROM {TABLE_NAME}")
            ).scalar()
            print(f"    Table '{TABLE_NAME}' existe déjà ({count:,} lignes)")
            print(f"  → Elle sera remplacée complètement")

    # 5. Écriture dans PostgreSQL par chunks avec barre de progression
    print(f"\n Écriture dans PostgreSQL (table : {TABLE_NAME}) ...")
    print(f"  Batch size : {CHUNKSIZE:,} lignes\n")

    n_chunks = (len(df) // CHUNKSIZE) + 1
    start = time.time()

    for i, chunk_start in enumerate(tqdm(
        range(0, len(df), CHUNKSIZE),
        total=n_chunks,
        desc="Ingestion",
        unit="batch"
    )):
        chunk = df.iloc[chunk_start : chunk_start + CHUNKSIZE]

        # Premier chunk → replace (crée/remplace la table)
        # Chunks suivants → append (ajoute à la table existante)
        mode = "replace" if i == 0 else "append"

        chunk.to_sql(
            name=TABLE_NAME,
            con=engine,
            if_exists=mode,
            index=False,
        )

    elapsed = time.time() - start
    print(f"\n Ingestion terminée en {elapsed:.1f}s")

    # 6. Vérification finale — compte les lignes dans PostgreSQL
    with engine.connect() as conn:
        count_pg = conn.execute(
            text(f"SELECT COUNT(*) FROM {TABLE_NAME}")
        ).scalar()

    print(f"\n Vérification :")
    print(f"  Lignes dans le CSV       : {len(df):,}")
    print(f"  Lignes dans PostgreSQL   : {count_pg:,}")

    if len(df) == count_pg:
        print(f"   Parfait — toutes les lignes sont bien insérées")
    else:
        print(f"    Différence détectée — vérifier les logs")

    return True


if __name__ == "__main__":
    ingest_raw()