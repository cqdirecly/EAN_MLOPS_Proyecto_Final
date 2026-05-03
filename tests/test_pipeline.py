"""
Pruebas del pipeline de clasificacion multiclase Bike Sharing.

Cubre:
    - Carga del config
    - Generacion del sample sintetico
    - Binning con pd.qcut (4 clases balanceadas)
    - Drop de columnas con leakage (casual, registered, cnt)
    - One-Hot encoding de categoricas
    - Stratified split
    - Metricas de clasificacion
    - Smoke test end-to-end con XGBoost
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import (
    build_dataset,
    create_target_classes,
    generate_synthetic_sample,
    load_raw_dataframe,
    preprocess,
    split_data,
)
from src.evaluate import classification_metrics
from src.train import train_model
from src.utils import load_config

REPO = Path(__file__).resolve().parent.parent
SAMPLE_CSV = REPO / "data" / "sample_bike_hour.csv"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def sample_csv() -> Path:
    if not SAMPLE_CSV.exists():
        generate_synthetic_sample(SAMPLE_CSV, n=2000)
    return SAMPLE_CSV


@pytest.fixture(scope="session")
def config(sample_csv) -> dict:
    cfg = load_config(REPO / "config.yaml")
    cfg["data"]["raw_path"] = str(sample_csv)
    return cfg


@pytest.fixture(scope="session")
def raw_df(config) -> pd.DataFrame:
    return load_raw_dataframe(config)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_config_loads(config):
    """El YAML tiene las 5 secciones principales y el target esperado."""
    for key in ("data", "split", "model", "mlflow", "artifacts"):
        assert key in config
    assert config["data"]["target"] == "demand_level"
    assert config["data"]["n_classes"] == 4


def test_synthetic_sample_schema(sample_csv):
    """La muestra sintetica debe tener todas las columnas del Bike Sharing."""
    df = pd.read_csv(sample_csv)
    columnas_esperadas = {
        "instant", "dteday", "season", "yr", "mnth", "hr",
        "holiday", "weekday", "workingday", "weathersit",
        "temp", "atemp", "hum", "windspeed",
        "casual", "registered", "cnt",
    }
    assert columnas_esperadas.issubset(set(df.columns))
    assert len(df) >= 200


def test_create_target_classes_balanced(raw_df, config):
    """El binning con pd.qcut produce 4 clases aprox. balanceadas."""
    df, bin_edges = create_target_classes(raw_df, config)
    counts = df["demand_level"].value_counts().sort_index()
    assert len(counts) == 4
    # Cada clase debe tener entre 15% y 35% del total (balance)
    proporciones = counts / len(df)
    assert (proporciones > 0.15).all() and (proporciones < 0.35).all()
    # Los bin edges deben ser monotonos crecientes
    assert all(bin_edges[i] < bin_edges[i + 1] for i in range(len(bin_edges) - 1))


def test_preprocess_drops_leakage(raw_df, config):
    """casual, registered y cnt deben desaparecer del input."""
    df, _ = create_target_classes(raw_df, config)
    X, _, features = preprocess(df, config)
    for col in ("casual", "registered", "cnt", "instant", "dteday"):
        assert col not in X.columns
    assert "demand_level" not in X.columns


def test_preprocess_one_hot(raw_df, config):
    """season, mnth, hr, weekday, weathersit son convertidas a dummies."""
    df, _ = create_target_classes(raw_df, config)
    X, _, features = preprocess(df, config)
    for col in ("season", "mnth", "hr", "weekday", "weathersit"):
        assert col not in features
    assert any(f.startswith("season_") for f in features)
    assert any(f.startswith("hr_") for f in features)


def test_split_stratified_proportion(raw_df, config):
    """Con stratify, las proporciones de clases en train y test son similares."""
    df, _ = create_target_classes(raw_df, config)
    X, y, _ = preprocess(df, config)
    X_train, X_test, y_train, y_test = split_data(X, y, config)

    p_train = y_train.value_counts(normalize=True).sort_index()
    p_test = y_test.value_counts(normalize=True).sort_index()
    # La diferencia entre proporciones de cada clase debe ser pequenia
    assert (abs(p_train - p_test) < 0.05).all()


def test_classification_metrics_keys():
    """classification_metrics retorna las claves esperadas."""
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    proba = np.eye(4)[np.tile([0, 1, 2, 3], 2)]  # one-hot perfecto
    metrics = classification_metrics(y_true, y_pred, y_proba=proba)
    for key in ("accuracy", "f1_macro", "precision_macro", "recall_macro", "log_loss"):
        assert key in metrics
    assert metrics["accuracy"] == 1.0


def test_train_model_learns(config):
    """XGBClassifier alcanza accuracy razonable en train (no es aleatorio)."""
    ds = build_dataset(config)
    # En tests usamos pocos estimadores para que sea rapido
    fast_params = {**config["model"]["params"], "n_estimators": 50}
    model = train_model(ds["X_train"], ds["y_train"], fast_params)
    train_acc = model.score(ds["X_train"], ds["y_train"])
    assert train_acc > 0.5, f"Modelo demasiado debil: train_acc={train_acc:.3f}"


def test_pipeline_smoke(config):
    """End-to-end: build_dataset + train + predict + metrics."""
    ds = build_dataset(config)
    fast_params = {**config["model"]["params"], "n_estimators": 50}
    model = train_model(ds["X_train"], ds["y_train"], fast_params)
    y_pred = model.predict(ds["X_test"])
    assert len(y_pred) == len(ds["y_test"])
    # Las predicciones son clases validas (0..n_classes-1)
    assert set(np.unique(y_pred)).issubset(set(range(config["data"]["n_classes"])))
    metrics = classification_metrics(ds["y_test"], y_pred)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0
