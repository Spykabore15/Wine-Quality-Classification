# Wine Quality Classification

Classification binaire de la qualité du vin (bon vin si `quality >= 7`) à partir de ses caractéristiques physico-chimiques. Projet Machine Learning 1, EFREI 2026-2027.

Données : vins rouges et blancs "Vinho Verde" fusionnés (Cortez et al., 2009).

## Structure

```
Datasets/   données brutes
notebooks/  01_eda, 02_modeling, 03_tuning (exploration, modélisation, optimisation)
src/        code réutilisable (config, données, features, modèles, évaluation, tuning, vote, entraînement)
tests/      tests unitaires (pytest)
app/        API FastAPI et page web de prédiction
models/     modèle entraîné (joblib, généré)
report/     figures du rapport (le rapport est `main.tex` à la racine)
```

## Avancement

- [x] EDA (`notebooks/01_eda.ipynb`)
- [x] Features et sélection de variables (`src/features.py`)
- [x] Modèles (`src/models.py`)
- [x] Évaluation (`src/evaluate.py`, `notebooks/02_modeling.ipynb`)
- [x] Optimisation des hyperparamètres et vote souple final (`src/tuning.py`, `src/ensemble.py`, `notebooks/03_tuning.ipynb`)
- [x] API FastAPI et page web (`app/`)
- [x] Dockerfile (build à valider sur une machine avec Docker démarré)
- [x] Rapport LaTeX (`main.tex`, figures dans `report/figures/`)

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest
```

## Modèle et application web

Le modèle final (vote souple forêt + SVM + boosting + k-NN, seuil F1 réglé en validation croisée) est reconstruit par :

```powershell
python -m src.train          # écrit models/wine_model.joblib (environ 1 à 2 min)
uvicorn app.main:app         # http://127.0.0.1:8000 (page web), /docs (API), /health
```

Exemple d'appel :

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{\"fixed_acidity\":6.6,\"volatile_acidity\":0.29,\"citric_acid\":0.31,\"residual_sugar\":3.9,\"chlorides\":0.027,\"free_sulfur_dioxide\":39,\"total_sulfur_dioxide\":96,\"density\":0.9904,\"pH\":3.24,\"sulphates\":0.6,\"alcohol\":12.6,\"type\":\"white\"}'
```

## Docker

```bash
docker build -t wine-quality .   # réentraîne le modèle dans l'image
docker run -p 8000:8000 wine-quality
```
