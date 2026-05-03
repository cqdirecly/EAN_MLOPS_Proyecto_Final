# =====================================================================
# Makefile - Bike Demand Classification (XGBoost + MLflow)
# Reglas: install, lint, test, train, mlflow-ui, clean, all
# =====================================================================

PYTHON ?= python
PIP    ?= pip

.PHONY: help install lint test train mlflow-ui clean all

help:
	@echo ""
	@echo "Comandos disponibles:"
	@echo "  make install     -> Instala dependencias desde requirements.txt"
	@echo "  make lint        -> Corre flake8 sobre src/ y tests/"
	@echo "  make test        -> Ejecuta los tests con pytest"
	@echo "  make train       -> Pipeline completo (data + train + MLflow)"
	@echo "  make mlflow-ui   -> Levanta la UI de MLflow en http://127.0.0.1:5000"
	@echo "  make clean       -> Borra artefactos generados"
	@echo "  make all         -> install + lint + test + train"
	@echo ""

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

lint:
	$(PYTHON) -m flake8 src/ tests/ --max-line-length=110 --extend-ignore=E501,W503,E203 --exclude=__pycache__,.venv,venv --statistics

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

train:
	$(PYTHON) -m src.train --config config.yaml

mlflow-ui:
	$(PYTHON) -m mlflow ui --host 127.0.0.1 --port 5000

clean:
	rm -rf mlruns artifacts data/bike_hour.csv .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Limpieza completa."

all: install lint test train