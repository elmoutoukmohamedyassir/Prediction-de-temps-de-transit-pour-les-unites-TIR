import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from ingestion.db_config import get_engine
engine = get_engine()
df = pd.read_sql("SELECT * FROM training_tir_export", engine)

# Parser les dates
DATE_COLS = [
    "date_zre", "date_impression", "entree_couloir",
    "entree_scanner_export", "debut_visite_physique_export",
    "fin_visite_physique_export", "sortie_sas_export",
    "entree_terminal_export", "date_embarquement", "embarquement_export",
]
for c in DATE_COLS:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)

print(f"Dataset chargé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
print(f"Cible transit_time_h — min: {df['transit_time_h'].min():.1f}h  "
      f"médiane: {df['transit_time_h'].median():.1f}h  "
      f"max: {df['transit_time_h'].max():.1f}h\n")

# FIGURE 1 — Distribution de la cible
# 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Distribution du temps de transit", fontsize=14, fontweight="bold")

axes[0].hist(df["transit_time_h"], bins=100, color="#2196F3", edgecolor="white", linewidth=0.3)
axes[0].axvline(df["transit_time_h"].median(), color="red", linestyle="--",
                label=f"Médiane : {df['transit_time_h'].median():.1f}h")
axes[0].axvline(df["transit_time_h"].mean(), color="orange", linestyle="--",
                label=f"Moyenne : {df['transit_time_h'].mean():.1f}h")
axes[0].set_xlabel("Heures")
axes[0].set_ylabel("Nombre de dossiers")
axes[0].set_title("Distribution brute")
axes[0].legend()

axes[1].hist(np.log1p(df["transit_time_h"]), bins=100,
             color="#4CAF50", edgecolor="white", linewidth=0.3)
axes[1].set_xlabel("log(1 + heures)")
axes[1].set_ylabel("Nombre de dossiers")
axes[1].set_title("Distribution log-transformée (plus symétrique = meilleur pour le ML)")

skew = df["transit_time_h"].skew()
print(f"Skewness (asymétrie) de transit_time_h : {skew:.2f}")
print(f"  > 1 = distribution très asymétrique → log-transform recommandé")
print(f"  % dossiers > 3 jours (72h)  : {(df['transit_time_h']>72).mean():.1%}")
print(f"  % dossiers > 5 jours (120h) : {(df['transit_time_h']>120).mean():.1%}\n")

plt.tight_layout()
plt.savefig("eda_1_distribution_cible.png", dpi=150, bbox_inches="tight")
plt.close()
print("Sauvegardé → eda_1_distribution_cible.png")

# FIGURE 2 — Transit time par couloir
if "couloir" in df.columns:
    couloir_stats = df.groupby("couloir")["transit_time_h"].agg(
        ["median", "mean", "count"]
    ).sort_values("median")

    print("\nTransit time par couloir :")
    print(couloir_stats.to_string())

    fig, ax = plt.subplots(figsize=(10, 5))
    couloirs = couloir_stats.index.tolist()
    medians  = couloir_stats["median"].values
    counts   = couloir_stats["count"].values

    bars = ax.bar(couloirs, medians, color="#FF5722", edgecolor="white")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"n={count:,}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Couloir")
    ax.set_ylabel("Médiane transit (heures)")
    ax.set_title("Temps de transit médian par couloir")
    plt.tight_layout()
    plt.savefig("eda_2_transit_par_couloir.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Sauvegardé → eda_2_transit_par_couloir.png")

# FIGURE 3 — Transit time par nature de marchandise (top 10)
if "nature_marchandise" in df.columns:
    top_nat = df["nature_marchandise"].value_counts().head(10).index
    nat_stats = df[df["nature_marchandise"].isin(top_nat)].groupby(
        "nature_marchandise")["transit_time_h"].median().sort_values()

    print("\nTransit time médian par nature marchandise (top 10) :")
    print(nat_stats.to_string())

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(nat_stats.index, nat_stats.values, color="#9C27B0", edgecolor="white")
    ax.set_xlabel("Médiane transit (heures)")
    ax.set_title("Temps de transit médian par nature de marchandise (top 10)")
    plt.tight_layout()
    plt.savefig("eda_3_transit_par_marchandise.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Sauvegardé → eda_3_transit_par_marchandise.png")

# 
# FIGURE 4 — Scanner et visite physique
# 
df["a_eu_scanner"]         = df["entree_scanner_export"].notna()
df["a_eu_visite_physique"] = df["debut_visite_physique_export"].notna()

scanner_med = df.groupby("a_eu_scanner")["transit_time_h"].median()
visite_med  = df.groupby("a_eu_visite_physique")["transit_time_h"].median()

print("\nImpact du scanner sur transit_time_h (médiane) :")
print(f"  Sans scanner    : {scanner_med[False]:.1f}h")
print(f"  Avec scanner    : {scanner_med[True]:.1f}h")
print(f"  Différence      : +{scanner_med[True]-scanner_med[False]:.1f}h")

print("\nImpact de la visite physique sur transit_time_h (médiane) :")
print(f"  Sans visite     : {visite_med[False]:.1f}h")
print(f"  Avec visite     : {visite_med[True]:.1f}h")
print(f"  Différence      : +{visite_med[True]-visite_med[False]:.1f}h")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Impact des contrôles sur le temps de transit", fontsize=13, fontweight="bold")

for ax, (col, label) in zip(axes, [
    ("a_eu_scanner", "Scanner"),
    ("a_eu_visite_physique", "Visite physique")
]):
    groups = [
        df[df[col] == False]["transit_time_h"].clip(upper=100),
        df[df[col] == True]["transit_time_h"].clip(upper=100),
    ]
    ax.boxplot(groups, tick_labels=["Sans", "Avec"], patch_artist=True,
           boxprops=dict(facecolor="#E3F2FD"),
           medianprops=dict(color="red", linewidth=2))
    ax.set_title(f"Impact {label}")
    ax.set_ylabel("Transit time (h) — plafonné à 100h")

plt.tight_layout()
plt.savefig("eda_4_impact_controles.png", dpi=150, bbox_inches="tight")
plt.close()
print("Sauvegardé → eda_4_impact_controles.png")


# FIGURE 5 — Saisonnalité (heure, jour de semaine, mois)
# 
df["heure_zre"]     = df["date_zre"].dt.hour
df["jour_semaine"]  = df["date_zre"].dt.dayofweek   # 0=Lundi, 6=Dimanche
df["mois_zre"]      = df["date_zre"].dt.month

jours_labels = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Saisonnalité du temps de transit", fontsize=13, fontweight="bold")

# Par heure
h_med = df.groupby("heure_zre")["transit_time_h"].median()
axes[0].plot(h_med.index, h_med.values, marker="o", color="#F44336")
axes[0].set_xlabel("Heure d'arrivée au ZRE")
axes[0].set_ylabel("Médiane transit (h)")
axes[0].set_title("Par heure de la journée")
axes[0].set_xticks(range(0, 24, 2))

# Par jour de semaine
j_med = df.groupby("jour_semaine")["transit_time_h"].median()
axes[1].bar([jours_labels[i] for i in j_med.index], j_med.values, color="#2196F3")
axes[1].set_xlabel("Jour de la semaine")
axes[1].set_ylabel("Médiane transit (h)")
axes[1].set_title("Par jour de la semaine")

# Par mois
m_med = df.groupby("mois_zre")["transit_time_h"].median()
axes[2].bar(m_med.index, m_med.values, color="#4CAF50")
axes[2].set_xlabel("Mois")
axes[2].set_ylabel("Médiane transit (h)")
axes[2].set_title("Par mois")
axes[2].set_xticks(range(1, 13))

plt.tight_layout()
plt.savefig("eda_5_saisonnalite.png", dpi=150, bbox_inches="tight")
plt.close()
print("Sauvegardé → eda_5_saisonnalite.png")


# FIGURE 6 — Groupage impact
if "groupage" in df.columns:
    df["is_groupage"] = df["groupage"].notna()
    grp_med = df.groupby("is_groupage")["transit_time_h"].median()
    grp_cnt = df.groupby("is_groupage")["transit_time_h"].count()

    print("\nImpact du groupage :")
    print(f"  Sans groupage : {grp_med[False]:.1f}h  (n={grp_cnt[False]:,})")
    print(f"  Avec groupage : {grp_med[True]:.1f}h   (n={grp_cnt[True]:,})")
    print(f"  Différence    : +{grp_med[True]-grp_med[False]:.1f}h")

# FIGURE 7 — ss_type_unite
if "ss_type_unite" in df.columns:
    type_stats = df.groupby("ss_type_unite")["transit_time_h"].agg(
        ["median", "count"]).sort_values("median")
    print("\nTransit time par type d'unité :")
    print(type_stats.to_string())

print("\n" + "="*60)
print("RÉSUMÉ EDA — Valeurs manquantes dans les features clés")
print("="*60)
features_check = [
    "couloir", "ss_type_unite", "nature_marchandise",
    "vide_plein", "poids", "groupage",
    "entree_scanner_export", "sortie_sas_export",
    "entree_terminal_export", "debut_visite_physique_export"
]
for f in features_check:
    if f in df.columns:
        pct_missing = df[f].isna().mean()
        print(f"  {f:40s} {pct_missing:.1%} manquant")

print("\n EDA terminée — 5 graphiques générés")
