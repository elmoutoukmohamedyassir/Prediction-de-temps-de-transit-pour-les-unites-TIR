"""
tune_optuna.py
Recherche d'hyperparamètres optimaux pour le meilleur checkpoint (M4_Terminal)
via Optuna (TPE sampler). Split 3-way train/val/test pour éviter le data
leakage de tuning — Optuna ne voit jamais le test set.
Usage : python -m ml.tune_optuna
"""

import os
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES_DIR = "ml/features"
BEST_CHECKPOINT_FILE = "features_m4.parquet"
N_TRIALS = 50

TARGET_COLS = ["transit_time_h", "log_transit_time_h", "duree_restante_estimee"]


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in TARGET_COLS]


def temporal_split_3way(df: pd.DataFrame, val_size: float = 0.2, test_size: float = 0.2):
    """
    Split en 3, toujours temporel (jamais aléatoire).
    Ordre chronologique : train (plus ancien) → validation → test (plus récent).
    """
    n = len(df)
    test_start = int(n * (1 - test_size))
    val_start  = int(test_start * (1 - val_size))

    train = df.iloc[:val_start]
    val   = df.iloc[val_start:test_start]
    test  = df.iloc[test_start:]
    return train, val, test


def compute_metrics(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return {"mae": mae, "rmse": rmse, "mape": mape}


if __name__ == "__main__":
    print("=" * 55)
    print("  OPTUNA TUNING — checkpoint M4_Terminal")
    print("=" * 55)

    df = pd.read_parquet(os.path.join(FEATURES_DIR, BEST_CHECKPOINT_FILE))
    feature_cols = get_feature_cols(df)

    train_df, val_df, test_df = temporal_split_3way(df)
    print(f"Train : {len(train_df):,} | Val : {len(val_df):,} | Test : {len(test_df):,}")

    X_train, y_train = train_df[feature_cols], train_df["log_transit_time_h"]
    X_val,   y_val   = val_df[feature_cols],   val_df["log_transit_time_h"]
    X_test,  y_test  = test_df[feature_cols],  test_df["log_transit_time_h"]

    # ── Objectif Optuna pour LightGBM ──────────────────────
    def objective_lgb(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "verbosity": -1,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        return rmse

    print("\n--- Tuning LightGBM ---")
    study_lgb = optuna.create_study(direction="minimize", study_name="lgb_m4_tuning")
    study_lgb.optimize(objective_lgb, n_trials=N_TRIALS, show_progress_bar=True)
    print(f"Meilleurs params LightGBM : {study_lgb.best_params}")
    print(f"Meilleur RMSE (log-space) validation : {study_lgb.best_value:.4f}")

    # ── Objectif Optuna pour XGBoost ───────────────────────
    def objective_xgb(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 20),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "eval_metric": "rmse",
            "early_stopping_rounds": 50,
            "verbosity": 0,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        return rmse

    print("\n--- Tuning XGBoost ---")
    study_xgb = optuna.create_study(direction="minimize", study_name="xgb_m4_tuning")
    study_xgb.optimize(objective_xgb, n_trials=N_TRIALS, show_progress_bar=True)
    print(f"Meilleurs params XGBoost : {study_xgb.best_params}")
    print(f"Meilleur RMSE (log-space) validation : {study_xgb.best_value:.4f}")

    # ── Réentraînement final sur train+val, évaluation UNIQUE sur test ──
    print("\n" + "=" * 55)
    print("  ÉVALUATION FINALE SUR TEST SET (jamais touché avant)")
    print("=" * 55)

    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])

    best_lgb = lgb.LGBMRegressor(**study_lgb.best_params, random_state=42, verbosity=-1)
    best_lgb.fit(X_trainval, y_trainval)
    preds_lgb = best_lgb.predict(X_test)
    metrics_lgb = compute_metrics(y_test.values, preds_lgb)
    print(f"LightGBM (tuné) → MAE={metrics_lgb['mae']:.2f}h  "
          f"RMSE={metrics_lgb['rmse']:.2f}h  MAPE={metrics_lgb['mape']:.1f}%")

    best_xgb_params = {k: v for k, v in study_xgb.best_params.items()}
    best_xgb = xgb.XGBRegressor(**best_xgb_params, random_state=42)
    best_xgb.fit(X_trainval, y_trainval)
    preds_xgb = best_xgb.predict(X_test)
    metrics_xgb = compute_metrics(y_test.values, preds_xgb)
    print(f"XGBoost  (tuné) → MAE={metrics_xgb['mae']:.2f}h  "
          f"RMSE={metrics_xgb['rmse']:.2f}h  MAPE={metrics_xgb['mape']:.1f}%")

    print("\n--- Comparaison avec les résultats par défaut (référence) ---")
    print("XGBoost M4_Terminal (défaut) : MAE=6.30h  RMSE=10.24h  MAPE=29.8%")