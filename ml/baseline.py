import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES_DIR = "ml/features"

FILENAME = "features_m1.parquet"


def temporal_split(df, test_size=0.2):
    n = len(df)
    split_idx = int(n * (1 - test_size))
    return df.iloc[:split_idx], df.iloc[split_idx:]


if __name__ == "__main__":
    df = pd.read_parquet(os.path.join(FEATURES_DIR, FILENAME))

    train_df, test_df = temporal_split(df)

    # La baseline est calculée UNIQUEMENT sur train_df, jamais sur test_df
    mediane_par_couloir = train_df.groupby("couloir_enc")["transit_time_h"].median()
    print("Médianes apprises sur train (par couloir encodé) :")
    print(mediane_par_couloir)

    # Applique sur test_df
    test_df = test_df.copy()
    test_df["baseline_pred"] = test_df["couloir_enc"].map(mediane_par_couloir)

    # Si un couloir du test n'existe pas dans train (rare mais possible),
    # on retombe sur la médiane globale
    mediane_globale = train_df["transit_time_h"].median()
    test_df["baseline_pred"] = test_df["baseline_pred"].fillna(mediane_globale)

    y_true = test_df["transit_time_h"]
    y_pred = test_df["baseline_pred"]

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    print("\n" + "=" * 50)
    print("  BASELINE NAÏVE — médiane historique par couloir")
    print("=" * 50)
    print(f"  MAE  = {mae:.2f}h")
    print(f"  RMSE = {rmse:.2f}h")
    print(f"  MAPE = {mape:.1f}%")