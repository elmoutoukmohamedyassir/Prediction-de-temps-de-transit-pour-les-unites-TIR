import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import xgboost as xgb

FEATURES_DIR = "ml/features"
MODELS_DIR   = "ml/models"
os.makedirs(MODELS_DIR, exist_ok=True)

CHECKPOINTS = {
    "M1_ZRE":      "features_m1.parquet",
    "M2_Scanner":  "features_m2.parquet",
    "M3_SAS":      "features_m3.parquet",
    "M4_Terminal": "features_m4.parquet",
}

# Colonnes cibles à exclure des features
TARGET_COLS = ["transit_time_h", "log_transit_time_h", "duree_restante_estimee"]


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in TARGET_COLS]


def temporal_split(df: pd.DataFrame, test_size: float = 0.2):
    """
    Split temporel — les 80% premières lignes = train,
    les 20% dernières = test.
    JAMAIS de split aléatoire sur des données temporelles.
    """
    n = len(df)
    split_idx = int(n * (1 - test_size))
    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]
    return train, test


def evaluate(y_true_log, y_pred_log, nom_modele: str):
    """
    Évalue en espace log ET en espace original (heures).
    """
    # Reconvertir en heures réelles
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    print(f"  {nom_modele:20s} → MAE={mae:.2f}h  RMSE={rmse:.2f}h  MAPE={mape:.1f}%")
    return {"mae": mae, "rmse": rmse, "mape": mape}


def train_lgb(X_train, y_train, X_test, y_test):
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    return model


def train_xgb(X_train, y_train, X_test, y_test):
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        eval_metric="rmse",
        early_stopping_rounds=50,
        verbosity=1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
    )
    return model


def save_model(model, name: str):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Sauvegardé → {path}")


if __name__ == "__main__":
    results = {}

    for checkpoint, filename in CHECKPOINTS.items():
        print("\n" + "=" * 55)
        print(f"  CHECKPOINT : {checkpoint}")
        print("=" * 55)

        # Charger le dataset
        df = pd.read_parquet(os.path.join(FEATURES_DIR, filename))
        feature_cols = get_feature_cols(df)

        print(f"  Features : {len(feature_cols)} colonnes")
        print(f"  Lignes   : {len(df):,}")

        # Split temporel
        train_df, test_df = temporal_split(df)
        X_train = train_df[feature_cols]
        y_train = train_df["log_transit_time_h"]
        X_test  = test_df[feature_cols]
        y_test  = test_df["log_transit_time_h"]

        print(f"  Train : {len(train_df):,} | Test : {len(test_df):,}")
        print()

        # LightGBM
        print("  Entraînement LightGBM...")
        lgb_model = train_lgb(X_train, y_train, X_test, y_test)
        lgb_preds = lgb_model.predict(X_test)
        lgb_metrics = evaluate(y_test.values, lgb_preds, "LightGBM")
        save_model(lgb_model, f"lgb_{checkpoint}")

        # XGBoost
        print("  Entraînement XGBoost...")
        xgb_model = train_xgb(X_train, y_train, X_test, y_test)
        xgb_preds = xgb_model.predict(X_test)
        xgb_metrics = evaluate(y_test.values, xgb_preds, "XGBoost")
        save_model(xgb_model, f"xgb_{checkpoint}")

        results[checkpoint] = {
            "LightGBM": lgb_metrics,
            "XGBoost":  xgb_metrics,
        }

    # Résumé comparatif final
    print("\n" + "=" * 55)
    print("  RÉSUMÉ COMPARATIF")
    print("=" * 55)
    print(f"{'Checkpoint':<15} {'Modèle':<12} {'MAE':>8} {'RMSE':>8} {'MAPE':>8}")
    print("-" * 55)
    for checkpoint, models in results.items():
        for model_name, metrics in models.items():
            print(f"{checkpoint:<15} {model_name:<12} "
                  f"{metrics['mae']:>7.2f}h "
                  f"{metrics['rmse']:>7.2f}h "
                  f"{metrics['mape']:>7.1f}%")