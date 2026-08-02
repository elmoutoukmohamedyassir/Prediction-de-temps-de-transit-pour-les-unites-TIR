import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import mlflow
import mlflow.lightgbm
import mlflow.xgboost

FEATURES_DIR = "ml/features"
MODELS_DIR   = "ml/models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Nom de l'experiment MLflow
EXPERIMENT_NAME = "tanger-med-transit"

CHECKPOINTS = {
    "M1_ZRE":      "features_m1.parquet",
    "M2_Scanner":  "features_m2.parquet",
    "M3_SAS":      "features_m3.parquet",
    "M4_Terminal": "features_m4.parquet",
}

TARGET_COLS = ["transit_time_h", "log_transit_time_h", "duree_restante_estimee"]


def get_feature_cols(df):
    return [c for c in df.columns if c not in TARGET_COLS]


def temporal_split(df, test_size=0.2):
    """
    Split temporel — 80% anciennes données = train
                     20% récentes = test
    JAMAIS aléatoire sur données temporelles.
    """
    n = len(df)
    split_idx = int(n * (1 - test_size))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def compute_metrics(y_true_log, y_pred_log):
    """
    Calcule MAE/RMSE/MAPE/R² en espace réel (heures)
    après avoir reconverti depuis l'espace log.
    """
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2   = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def train_and_log_lgb(X_train, y_train, X_test, y_test, checkpoint):
    """
    Entraîne LightGBM et log tout dans MLflow.
    """
    params = {
        "n_estimators":  500,
        "learning_rate": 0.05,
        "max_depth":     6,
        "num_leaves":    31,
        "random_state":  42,
        "n_jobs":        -1,
    }

    # mlflow.start_run() ouvre un nouveau run MLflow
    # tout ce qui est à l'intérieur est enregistré automatiquement
    with mlflow.start_run(run_name=f"LightGBM_{checkpoint}"):

        # Log les paramètres du modèle
        mlflow.log_params(params)
        mlflow.log_param("checkpoint", checkpoint)
        mlflow.log_param("model_type", "LightGBM")
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test",  len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])

        # Entraînement
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
        )

        # Prédictions + métriques
        preds   = model.predict(X_test)
        metrics = compute_metrics(y_test.values, preds)

        # Log les métriques dans MLflow
        mlflow.log_metric("MAE_heures",  metrics["mae"])
        mlflow.log_metric("RMSE_heures", metrics["rmse"])
        mlflow.log_metric("R2",          metrics["r2"])

        

        # Log le modèle dans MLflow (sauvegarde automatique)
        mlflow.lightgbm.log_model(model, artifact_path="model")

        # Sauvegarde aussi en .pkl local
        pkl_path = os.path.join(MODELS_DIR, f"lgb_{checkpoint}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)

        print(f"  LightGBM {checkpoint} → "
              f"MAE={metrics['mae']:.2f}h  "
              f"RMSE={metrics['rmse']:.2f}h  "
              f"MAPE={metrics['mape']:.1f}%  "
              f"R²={metrics['r2']:.3f}")

    return model, metrics


def train_and_log_xgb(X_train, y_train, X_test, y_test, checkpoint):
    """
    Entraîne XGBoost et log tout dans MLflow.
    """
    params = {
        "n_estimators":       500,
        "learning_rate":      0.05,
        "max_depth":          6,
        "random_state":       42,
        "n_jobs":             -1,
        "eval_metric":        "rmse",
        "early_stopping_rounds": 50,
        "verbosity":          1,
    }

    with mlflow.start_run(run_name=f"XGBoost_{checkpoint}"):

        mlflow.log_params(params)
        mlflow.log_param("checkpoint", checkpoint)
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_train",    len(X_train))
        mlflow.log_param("n_test",     len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
        )

        preds   = model.predict(X_test)
        metrics = compute_metrics(y_test.values, preds)

        mlflow.log_metric("MAE_heures",  metrics["mae"])
        mlflow.log_metric("RMSE_heures", metrics["rmse"])
        mlflow.log_metric("MAPE_pct",    metrics["mape"])
        mlflow.log_metric("R2",          metrics["r2"])

        mlflow.xgboost.log_model(model, artifact_path="model")

        pkl_path = os.path.join(MODELS_DIR, f"xgb_{checkpoint}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)

        print(f"  XGBoost   {checkpoint} → "
              f"MAE={metrics['mae']:.2f}h  "
              f"RMSE={metrics['rmse']:.2f}h  "
              f"MAPE={metrics['mape']:.1f}%")

    return model, metrics


if __name__ == "__main__":

    # Créer/récupérer l'experiment MLflow
    # Si l'experiment n'existe pas → MLflow le crée
    # Si il existe → MLflow ajoute les nouveaux runs dedans
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("=" * 55)
    print("  ENTRAÎNEMENT — LightGBM + XGBoost × 4 checkpoints")
    print(f"  MLflow UI → mlflow ui → http://localhost:5000")
    print("=" * 55)

    all_results = {}

    for checkpoint, filename in CHECKPOINTS.items():
        print(f"\n{'='*55}")
        print(f"  CHECKPOINT : {checkpoint}")
        print(f"{'='*55}")

        df           = pd.read_parquet(os.path.join(FEATURES_DIR, filename))
        feature_cols = get_feature_cols(df)

        train_df, test_df = temporal_split(df)
        X_train = train_df[feature_cols]
        y_train = train_df["log_transit_time_h"]
        X_test  = test_df[feature_cols]
        y_test  = test_df["log_transit_time_h"]

        print(f"  Features : {len(feature_cols)} | "
              f"Train : {len(train_df):,} | Test : {len(test_df):,}")

        _, lgb_metrics = train_and_log_lgb(
            X_train, y_train, X_test, y_test, checkpoint
        )
        _, xgb_metrics = train_and_log_xgb(
            X_train, y_train, X_test, y_test, checkpoint
        )

        all_results[checkpoint] = {
            "LightGBM": lgb_metrics,
            "XGBoost":  xgb_metrics,
        }

    # Résumé comparatif
    print(f"\n{'='*60}")
    print("  RÉSUMÉ COMPARATIF FINAL")
    print(f"{'='*60}")
    print(f"{'Checkpoint':<15} {'Modèle':<12} "
          f"{'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'R²':>7}")
    print("-" * 60)
    for cp, models in all_results.items():
        for model_name, m in models.items():
            print(f"{cp:<15} {model_name:<12} "
                  f"{m['mae']:>7.2f}h "
                  f"{m['rmse']:>7.2f}h "
                  f"{m['mape']:>7.1f}% "
                  f"{m['r2']:>7.3f}")

    print(f"\n Tous les modèles entraînés et loggés dans MLflow")
    print(f"   Lance 'mlflow ui' pour voir les résultats visuellement")