"""
Modulo de carga y preprocesamiento del dataset Bike Sharing Demand (UCI).
Convierte la regresion sobre 'cnt' en clasificacion multiclase de 4 niveles.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Mirrors CSV directos (pueden caerse cuando los repos cambian)
_DEFAULT_MIRRORS: List[str] = [
    "https://raw.githubusercontent.com/JackyP/testing/master/datasets/hour.csv",
    "https://raw.githubusercontent.com/LiYangHart/Hyperparameter-Optimization-of-Machine-Learning-Algorithms/master/datasets/hour.csv",
]

# UCI publica el dataset como ZIP. Es la fuente mas estable.
_UCI_ZIP_URLS: List[str] = [
    "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip",
]


# ---------------------------------------------------------------------------
# Descarga / cache
# ---------------------------------------------------------------------------
def _download_csv(url: str, dest: Path, timeout: int = 30) -> bool:
    """Descarga un CSV directo desde una URL publica."""
    import requests
    try:
        logger.info("Intentando descargar CSV desde %s", url)
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        logger.info("Dataset guardado en %s (%d bytes)", dest, dest.stat().st_size)
        return True
    except Exception as exc:
        logger.warning("Fallo al descargar %s: %s", url, exc)
        return False


def _download_zip_and_extract(url: str, dest: Path, csv_name: str = "hour.csv",
                              timeout: int = 60) -> bool:
    """Descarga un ZIP y extrae el CSV especifico (ej. hour.csv) a dest."""
    import requests
    try:
        logger.info("Intentando descargar ZIP desde %s", url)
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            members = zf.namelist()
            target = next((m for m in members if m.endswith(csv_name)), None)
            if target is None:
                logger.warning("'%s' no esta en el ZIP. Miembros: %s", csv_name, members)
                return False
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(target) as src, open(dest, "wb") as out:
                out.write(src.read())
        logger.info("Extraido %s a %s (%d bytes)", csv_name, dest, dest.stat().st_size)
        return True
    except Exception as exc:
        logger.warning("Fallo al descargar/extraer %s: %s", url, exc)
        return False


def _ensure_dataset_local(config: Dict) -> Path:
    """Garantiza un CSV local. Orden: cache -> CSV mirrors -> UCI zip -> sample."""
    raw_path = Path(config["data"]["raw_path"])

    if raw_path.exists() and raw_path.stat().st_size > 1000:
        logger.info("Usando dataset cacheado en %s", raw_path)
        return raw_path

    # 1. Intentar mirrors CSV directos
    candidatos_csv = []
    for u in [config["data"].get("url"), config["data"].get("url_fallback"), *_DEFAULT_MIRRORS]:
        if u and u not in candidatos_csv:
            candidatos_csv.append(u)
    for url in candidatos_csv:
        if _download_csv(url, raw_path):
            return raw_path

    # 2. Intentar el ZIP oficial de UCI (mas estable)
    for zip_url in _UCI_ZIP_URLS:
        if _download_zip_and_extract(zip_url, raw_path, csv_name="hour.csv"):
            return raw_path

    # 3. Fallback: muestra sintetica
    sample = Path(__file__).resolve().parent.parent / "data" / "sample_bike_hour.csv"
    if sample.exists():
        logger.warning("Mirrors no disponibles. Usando muestra sintetica %s", sample)
        return sample

    raise RuntimeError(
        f"No se pudo obtener el dataset. Coloca hour.csv manualmente en {raw_path}."
    )


def load_raw_dataframe(config: Dict) -> pd.DataFrame:
    csv_path = _ensure_dataset_local(config)
    df = pd.read_csv(csv_path)
    logger.info("Dataset cargado: %d filas, %d columnas", *df.shape)
    return df


# ---------------------------------------------------------------------------
# Discretizacion del target (binning con cuartiles)
# ---------------------------------------------------------------------------
def create_target_classes(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, List[float]]:
    """Discretiza 'cnt' en N clases usando cuartiles (pd.qcut)."""
    df = df.copy()
    source = config["data"]["source_column"]
    target = config["data"]["target"]
    n = config["data"]["n_classes"]

    if source not in df.columns:
        raise KeyError(f"Columna fuente '{source}' no esta en el dataset.")

    df[target], bin_edges = pd.qcut(
        df[source], q=n, labels=list(range(n)), retbins=True, duplicates="drop"
    )
    df[target] = df[target].astype(int)

    bin_edges = [float(b) for b in bin_edges]
    logger.info("Target '%s' creado con cuartiles. Bins: %s", target, [round(b, 2) for b in bin_edges])
    logger.info("Distribucion de clases:\n%s", df[target].value_counts().sort_index().to_string())
    return df, bin_edges


# ---------------------------------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    df = df.copy()
    target = config["data"]["target"]

    if target not in df.columns:
        raise KeyError(f"Falta '{target}'. Llama create_target_classes() antes.")

    drop_cols = [c for c in config["data"].get("drop_columns", []) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        logger.info("Columnas eliminadas: %s", drop_cols)

    n_nulos = int(df.isna().sum().sum())
    if n_nulos:
        logger.info("Imputando %d nulos", n_nulos)
        for col in df.columns:
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode().iloc[0])

    y = df[target].astype(int)
    X = df.drop(columns=[target])

    cat_cols = [c for c in config["data"].get("categorical_columns", []) if c in X.columns]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=int)
        logger.info("One-Hot aplicado a %s -> %d columnas", cat_cols, X.shape[1])

    bin_cols = [c for c in X.columns if X[c].dropna().isin([0, 1]).all()]
    num_cols = [c for c in X.columns if c not in bin_cols]
    if num_cols:
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])
        logger.info("StandardScaler aplicado a %d columnas numericas", len(num_cols))

    return X, y, X.columns.tolist()


# ---------------------------------------------------------------------------
# Split (estratificado)
# ---------------------------------------------------------------------------
def split_data(X: pd.DataFrame, y: pd.Series, config: Dict):
    test_size = config["split"]["test_size"]
    random_state = config["split"]["random_state"]
    stratify = y if config["split"].get("stratify", False) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    logger.info("Split: train=%d, test=%d, stratify=%s",
                len(X_train), len(X_test), stratify is not None)
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Pipeline de alto nivel
# ---------------------------------------------------------------------------
def build_dataset(config: Dict) -> Dict:
    df = load_raw_dataframe(config)
    df, bin_edges = create_target_classes(df, config)
    X, y, feature_names = preprocess(df, config)
    X_train, X_test, y_train, y_test = split_data(X, y, config)
    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feature_names": feature_names,
        "bin_edges": bin_edges,
        "n_samples": int(len(df)),
    }


# ---------------------------------------------------------------------------
# Generador de muestra sintetica (fallback offline)
# ---------------------------------------------------------------------------
def generate_synthetic_sample(out_path, n: int = 2000, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    n = int(n)
    df = pd.DataFrame({
        "instant": np.arange(1, n + 1),
        "dteday": pd.date_range("2011-01-01", periods=n, freq="h").strftime("%Y-%m-%d"),
        "season": rng.integers(1, 5, size=n),
        "yr": rng.integers(0, 2, size=n),
        "mnth": rng.integers(1, 13, size=n),
        "hr": rng.integers(0, 24, size=n),
        "holiday": rng.integers(0, 2, size=n),
        "weekday": rng.integers(0, 7, size=n),
        "workingday": rng.integers(0, 2, size=n),
        "weathersit": rng.integers(1, 5, size=n),
        "temp": rng.uniform(0, 1, size=n).round(2),
        "atemp": rng.uniform(0, 1, size=n).round(4),
        "hum": rng.uniform(0, 1, size=n).round(2),
        "windspeed": rng.uniform(0, 1, size=n).round(4),
    })
    base = (300 * df["temp"] + 8 * df["hr"] + 30 * df["workingday"]
            + rng.normal(0, 50, size=n))
    df["casual"] = np.clip(base * 0.3, 0, None).round().astype(int)
    df["registered"] = np.clip(base * 0.7, 0, None).round().astype(int)
    df["cnt"] = df["casual"] + df["registered"]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    repo = Path(__file__).resolve().parent.parent
    out = generate_synthetic_sample(repo / "data" / "sample_bike_hour.csv", n=2000)
    print(f"Muestra sintetica generada: {out}")
