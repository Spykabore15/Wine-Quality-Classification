"""Reconstruit le modèle final (vote souple) : `python -m src.train [--output chemin]`."""

import argparse
import time
from pathlib import Path

import joblib

from .config import CV_FOLDS, MODEL_PATH
from .data import add_binary_target, drop_duplicate_wines, load_wine_data, split_data
from .ensemble import ENSEMBLE_MEMBERS, build_voting_ensemble
from .evaluate import best_f1_threshold, default_cv, evaluate_at_threshold, oof_scores
from .tuning import tune_model


def train_final_model(output: Path = MODEL_PATH, n_iter: int | None = None, verbose: bool = True) -> dict:
    """Optimise les quatre membres, règle le seuil F1 hors-pli sur le train, ré-entraîne le vote et l'exporte."""
    log = print if verbose else (lambda *a, **k: None)
    df = drop_duplicate_wines(add_binary_target(load_wine_data()))
    X_train, X_test, y_train, y_test = split_data(df)
    cv = default_cv()

    members = {}
    for name in ENSEMBLE_MEMBERS:
        t0 = time.perf_counter()
        search = tune_model(name, X_train, y_train, cv=cv, n_iter=n_iter if name in ("svm", "rf", "gb") else None)
        members[name] = search.best_estimator_
        log(f"{name:4s} ROC-AUC CV = {search.best_score_:.4f} ({time.perf_counter() - t0:.0f}s)")

    vote = build_voting_ensemble(members)
    scores, _ = oof_scores(vote, X_train, y_train, cv=cv)
    threshold = best_f1_threshold(y_train, scores)
    vote.fit(X_train, y_train)
    # En service, une requête isolée ne justifie pas de paralléliser
    vote.set_params(n_jobs=1)

    artifact = {
        "model": vote,
        "threshold": float(threshold),
        "name": "vote",
        "features": list(X_train.columns),
        "members": list(ENSEMBLE_MEMBERS),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)

    test = evaluate_at_threshold(vote, X_test, y_test, threshold)
    log(f"Seuil F1 ({CV_FOLDS} plis) = {threshold:.4f}")
    log("Test (information, aucun réglage) : " + ", ".join(f"{k}={v:.3f}" for k, v in test.items()))
    log(f"Modèle exporté : {output}")
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    train_final_model(parser.parse_args().output)
