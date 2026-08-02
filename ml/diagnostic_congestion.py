import pandas as pd
from ml.feature_engineering import load_data, add_congestion_features

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", None)

print("=" * 60)
print("  DIAGNOSTIC — congestion_1h / 3h / 6h")
print("=" * 60)

# ── Étape 1 : charger les données brutes ──────────────────
df = load_data()
n_avant = len(df)
print(f"\n[1] Lignes chargées (avant congestion) : {n_avant:,}")

# ── Étape 2 : vérifier les valeurs manquantes critiques ───
n_nan_couloir = df["couloir"].isna().sum()
n_nat_date = df["date_zre"].isna().sum()
print(f"[2] NaN dans 'couloir'   : {n_nan_couloir:,} ({n_nan_couloir/n_avant:.2%})")
print(f"    NaT dans 'date_zre' : {n_nat_date:,} ({n_nat_date/n_avant:.2%})")

if n_nan_couloir > 0:
    print("    ⚠️ groupby('couloir') va SILENCIEUSEMENT supprimer ces lignes")
if n_nat_date > 0:
    print("    ⚠️ Un index avec NaT peut casser rolling('1h') silencieusement")

# ── Étape 3 : vérifier les doublons exacts de date_zre ────
n_doublons = df.duplicated(subset=["couloir", "date_zre"]).sum()
print(f"[3] Doublons exacts (même couloir + même date_zre à la seconde près) : {n_doublons:,}")

# ── Étape 4 : appliquer add_congestion_features et comparer ─
df2 = add_congestion_features(df)
n_apres = len(df2)
print(f"\n[4] Lignes après add_congestion_features : {n_apres:,}")
if n_apres != n_avant:
    print(f"    ⚠️⚠️⚠️ PERTE DE {n_avant - n_apres:,} LIGNES — cause probable du problème")
else:
    print("    ✓ Aucune perte de lignes")

# ── Étape 5 : distribution des 3 colonnes ─────────────────
print("\n[5] Distribution des colonnes de congestion :")
print(df2[["congestion_1h", "congestion_3h", "congestion_6h"]].describe())

n_nan_cong = df2[["congestion_1h", "congestion_3h", "congestion_6h"]].isna().sum()
print("\n    NaN restants par colonne :")
print(n_nan_cong)

# ── Étape 6 : cohérence logique (1h <= 3h <= 6h toujours vrai) ─
incoherent_13 = (df2["congestion_1h"] > df2["congestion_3h"]).sum()
incoherent_36 = (df2["congestion_3h"] > df2["congestion_6h"]).sum()
print(f"\n[6] Incohérences logiques :")
print(f"    congestion_1h > congestion_3h : {incoherent_13:,} lignes")
print(f"    congestion_3h > congestion_6h : {incoherent_36:,} lignes")
if incoherent_13 > 0 or incoherent_36 > 0:
    print("    ⚠️⚠️⚠️ BUG CONFIRMÉ — une fenêtre plus large doit TOUJOURS")
    print("    contenir au moins autant de camions qu'une fenêtre plus petite.")
    print("    Montre-moi quelques lignes incohérentes ci-dessous :")
    mask = (df2["congestion_1h"] > df2["congestion_3h"]) | (df2["congestion_3h"] > df2["congestion_6h"])
    print(df2.loc[mask, ["couloir", "date_zre", "congestion_1h", "congestion_3h", "congestion_6h"]].head(10))

# ── Étape 7 : vérifier le tri à l'intérieur de chaque couloir ─
print("\n[7] Vérification que date_zre est bien croissant DANS chaque couloir :")
pb_tri = 0
for couloir, g in df2.groupby("couloir", sort=False):
    if not g["date_zre"].is_monotonic_increasing:
        pb_tri += 1
        print(f"    ⚠️ Couloir '{couloir}' n'est PAS trié par date_zre après concat/re-tri final")
if pb_tri == 0:
    print("    ✓ Tri correct dans tous les couloirs")

# ── Étape 8 : corrélation avec la cible et les autres features ─
print("\n[8] Corrélation avec transit_time_h (la cible) :")
print(df2[["congestion_1h", "congestion_3h", "congestion_6h"]].corrwith(df2["transit_time_h"]))

print("\n[8bis] Corrélation entre les 3 colonnes de congestion (colinéarité) :")
print(df2[["congestion_1h", "congestion_3h", "congestion_6h"]].corr())

print("\n" + "=" * 60)
print("  FIN DU DIAGNOSTIC — copie toute cette sortie pour analyse")
print("=" * 60)