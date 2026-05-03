"""
Pipeline principal de entrenamiento.

Ejecuta:
    python -m src.train

Hace:
    1. Carga config.yaml
    2. Descarga + binning + preprocesamiento del dataset Bike Sharing
    3. Entrena XGBClassifier multiclase (4 niveles de demanda)
    4. Evalua con accuracy, f1_macro, precision, recall, log_loss
    5. Registra en MLflow: params, metrics, signature, input_example, log_model
    6. Guarda matriz de confusion como artefacto
    7. Persiste copia local en artifacts/model.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature
from xgboost import XGBClassifier

from src.data import build_dataset
from src.evaluate import classification_metrics, save_confusion_matrix
from src.utils import load_config, setup_logging

logger = logging.getLogger("train")


def train_model(X_train, y_train, params: dict) -> XGBClassifier:
    """Entrena un XGBClassifier multiclase."""
    logger.info("Entrenando XGBClassifier con params=%s", params)
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


def save_local_artifacts(model, metrics: dict, output_dir: Path, model_filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / model_filename
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Modelo guardado en %s", model_path)
    logger.info("Metricas guardadas en %s", metrics_path)
    return model_path


def run_pipeline(config_path: str = "config.yaml") -> dict:
    setup_logging()
    cfg = load_config(config_path)

    # 1. Dataset
    ds = build_dataset(cfg)
    X_train, X_test = ds["X_train"], ds["X_test"]
    y_train, y_test = ds["y_train"], ds["y_test"]
    logger.info("Dataset listo: train=%d, test=%d, features=%d",
                len(X_train), len(X_test), X_train.shape[1])

    # 2. MLflow setup
    tracking_uri = cfg["mlflow"]["tracking_uri"]
    experiment_name = cfg["mlflow"]["experiment_name"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow tracking URI=%s, experimento='%s'", tracking_uri, experiment_name)

    model_params = cfg["model"]["params"]
    artifacts_cfg = cfg["artifacts"]
    output_dir = Path(artifacts_cfg["output_dir"])

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info("MLflow run iniciado: %s", run_id)

        # ---- log de parametros ----
        mlflow.log_param("model_name", cfg["model"]["name"])
        mlflow.log_param("test_size", cfg["split"]["test_size"])
        mlflow.log_param("random_state", cfg["split"]["random_state"])
        mlflow.log_param("stratify", cfg["split"].get("stratify", False))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_classes", cfg["data"]["n_classes"])
        mlflow.log_param("bin_edges", str(ds["bin_edges"]))
        for k, v in model_params.items():
            mlflow.log_param(f"model__{k}", v)

        # ---- entrenamiento ----
        model = train_model(X_train, y_train, model_params)

        # ---- evaluacion ----
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_proba_test = model.predict_proba(X_test)

        train_metrics = classification_metrics(y_train, y_pred_train)
        test_metrics = classification_metrics(y_test, y_pred_test, y_proba=y_proba_test)

        for name, value in train_metrics.items():
            mlflow.log_metric(f"train_{name}", value)
        for name, value in test_metrics.items():
            mlflow.log_metric(f"test_{name}", value)

        logger.info("Metricas TRAIN: %s", train_metrics)
        logger.info("Metricas TEST : %s", test_metrics)

        # ---- matriz de confusion ----
        cm_path = save_confusion_matrix(y_test, y_pred_test, output_dir / "confusion_matrix.json")
        mlflow.log_artifact(str(cm_path))

        # ---- features.json ----
        features_path = output_dir / "features.json"
        features_path.write_text(json.dumps(ds["feature_names"], indent=2))
        mlflow.log_artifact(str(features_path))

        # ---- signature + input_example ----
        input_example = X_train.head(5)
        signature = infer_signature(X_train, y_pred_train)

        # ---- log_model con XGBoost flavor ----
        mlflow.xgboost.log_model(
            xgb_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=cfg["mlflow"].get("registered_model_name"),
        )

        # ---- copia local del modelo ----
        save_local_artifacts(
            model,
            {"train": train_metrics, "test": test_metrics, "bin_edges": ds["bin_edges"]},
            output_dir,
            artifacts_cfg["model_filename"],
        )

        # ---- resumen ----
        print("\n" + "=" * 60)
        print(" RESUMEN DEL ENTRENAMIENTO ")
        print("=" * 60)
        print(f" run_id      : {run_id}")
        print(f" experimento : {experiment_name}")
        print(f" features    : {X_train.shape[1]}")
        print(f" muestras    : train={len(X_train)} | test={len(X_test)}")
        print(f" clases      : {cfg['data']['n_classes']}")
        print(" Metricas TEST:")
        for k, v in test_metrics.items():
            print(f"   - {k:<18}: {v:.4f}")
        print("=" * 60 + "\n")

        return {
            "run_id": run_id,
            "experiment": experiment_name,
            "metrics": {"train": train_metrics, "test": test_metrics},
        }


def main():
    parser = argparse.ArgumentParser(description="Entrena XGBClassifier y registra en MLflow.")
    parser.add_argument("--config", default="config.yaml", help="Ruta al config YAML.")
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
