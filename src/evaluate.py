"""Calculo de metricas multiclase y matriz de confusion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def classification_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    """
    Metricas relevantes para clasificacion multiclase.

    El proyecto exige minimo 2 metricas; aqui reportamos:
        - accuracy
        - f1_macro       : F1 promediado sin pesar por frecuencia
        - precision_macro
        - recall_macro
        - log_loss       : si se proveen probabilidades
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_proba is not None:
        metrics["log_loss"] = float(log_loss(y_true, y_proba))
    return metrics


def save_confusion_matrix(y_true, y_pred, out_path) -> Path:
    """Guarda la matriz de confusion como JSON (filas=real, cols=predicho)."""
    cm = confusion_matrix(y_true, y_pred)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"confusion_matrix": cm.tolist()}, f, indent=2)
    return out_path
