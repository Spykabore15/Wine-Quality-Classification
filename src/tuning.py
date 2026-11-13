import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from .config import RANDOM_STATE
from .evaluate import default_cv, scoring_dict
from .features import VIFFilter, build_preprocessor
from .models import build_model

# Choix structurels explorés pour tous les modèles : transformation log1p et filtre VIF
SHARED = {
    "prep": [build_preprocessor(log_skewed=False), build_preprocessor(log_skewed=True)],
    "select": ["passthrough", VIFFilter(threshold=10.0)],
}

SPACES = {
    "logreg": {
        "method": "grid",
        "params": {"clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10], "clf__class_weight": [None, "balanced"]},
    },
    "knn": {
        "method": "grid",
        "params": {
            "clf__n_neighbors": [3, 5, 9, 15, 25, 41, 61],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],
        },
    },
    "tree": {
        "method": "grid",
        "params": {
            "clf__max_depth": [3, 5, 7, 10, 15, None],
            "clf__min_samples_leaf": [1, 5, 10, 20],
            "clf__criterion": ["gini", "entropy"],
        },
    },
    "nb": {
        "method": "grid",
        "params": {"clf__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]},
    },
    "svm": {
        "method": "random",
        "n_iter": 40,
        "params": {
            "clf__C": loguniform(0.1, 100),
            "clf__gamma": loguniform(1e-3, 1),
            "clf__class_weight": [None, "balanced"],
        },
    },
    "rf": {
        "method": "random",
        "n_iter": 30,
        "params": {
            "clf__n_estimators": [200, 400, 600],
            "clf__max_depth": [None, 8, 12, 16, 24],
            "clf__min_samples_leaf": [1, 2, 3, 5, 8],
            "clf__max_features": ["sqrt", "log2", 0.5],
            "clf__class_weight": ["balanced", "balanced_subsample", None],
        },
    },
    "gb": {
        "method": "random",
        "n_iter": 30,
        "params": {
            "clf__n_estimators": randint(100, 500),
            "clf__learning_rate": loguniform(0.01, 0.3),
            "clf__max_depth": randint(2, 6),
            "clf__subsample": [0.6, 0.8, 1.0],
            "clf__min_samples_leaf": randint(1, 20),
        },
    },
}

SCORING = ("roc_auc", "average_precision", "f1", "recall", "precision")


def tune_model(name: str, X, y, cv=None, method: str | None = None, n_iter: int | None = None, n_jobs: int = -1):
    """Grid ou Random Search (méthode par défaut de `SPACES`) optimisant le ROC-AUC en CV stratifiée ; refit sur tout (X, y)."""
    if name not in SPACES:
        raise ValueError(f"Pas d'espace de recherche pour {name!r} (choix : {tuple(SPACES)})")
    space = SPACES[name]
    method = method or space["method"]
    params = {**SHARED, **space["params"]}
    # Évite la sur-souscription de threads : la parallélisation se fait au niveau de la recherche
    base = build_model(name, **({"n_jobs": 1} if name == "rf" else {}))
    scoring = {k: v for k, v in scoring_dict().items() if k in SCORING}
    common = dict(scoring=scoring, refit="roc_auc", cv=cv or default_cv(), n_jobs=n_jobs)

    if method == "grid":
        search = GridSearchCV(base, params, **common)
    elif method == "random":
        search = RandomizedSearchCV(
            base, params, n_iter=n_iter or space.get("n_iter", 30), random_state=RANDOM_STATE, **common
        )
    else:
        raise ValueError(f"Méthode inconnue : {method!r} (grid ou random)")
    return search.fit(X, y)


def _label(key: str, value):
    if key == "prep":
        return "log1p" if len(value.transformers) > 2 else "std"
    if key == "select":
        return "none" if isinstance(value, str) else "vif"
    return value


def search_results(search, top: int | None = None) -> pd.DataFrame:
    """Résultats lisibles d'une recherche, triés par rang (ROC-AUC moyen en CV)."""
    res = search.cv_results_
    cols = {}
    for key in search.param_distributions.keys() if hasattr(search, "param_distributions") else search.param_grid.keys():
        cols[key.replace("clf__", "")] = [_label(key, v) for v in res[f"param_{key}"]]
    for m in SCORING:
        cols[f"{m}_mean"] = res[f"mean_test_{m}"]
        cols[f"{m}_std"] = res[f"std_test_{m}"]
    cols["fit_time"] = res["mean_fit_time"]
    cols["rank"] = res["rank_test_roc_auc"]
    out = pd.DataFrame(cols).sort_values("rank").reset_index(drop=True)
    return out.head(top) if top else out


def n_candidates(search) -> int:
    return len(search.cv_results_["params"])


def best_params_readable(search) -> dict:
    return {k.replace("clf__", ""): _label(k, v) for k, v in search.best_params_.items()}
