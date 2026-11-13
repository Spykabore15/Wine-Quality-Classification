import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    ConfusionMatrixDisplay,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, learning_curve

from .config import CV_FOLDS, RANDOM_STATE
from .models import build_model

METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "average_precision")


def _pyplot():
    # matplotlib n'est requis que pour les figures : absent de l'image de service
    import matplotlib.pyplot as plt

    return plt


def default_cv() -> StratifiedKFold:
    return StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def get_scores(model, X) -> np.ndarray:
    """Score continu de la classe positive : probabilité si disponible, sinon fonction de décision."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def default_threshold(model) -> float:
    """Seuil de décision implicite de `predict` pour les scores de `get_scores`."""
    return 0.5 if hasattr(model, "predict_proba") else 0.0


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
        out["average_precision"] = average_precision_score(y_true, y_score)
    return out


def scoring_dict() -> dict:
    return {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }


def cross_validate_model(model, X, y, cv=None) -> dict:
    """Moyenne et écart-type de chaque métrique en CV stratifiée, plus le F1 d'entraînement (écart biais/variance)."""
    res = cross_validate(model, X, y, cv=cv or default_cv(), scoring=scoring_dict(), return_train_score=True)
    out = {}
    for m in METRICS:
        out[f"{m}_mean"] = res[f"test_{m}"].mean()
        out[f"{m}_std"] = res[f"test_{m}"].std()
    out["f1_train"] = res["train_f1"].mean()
    out["f1_gap"] = out["f1_train"] - out["f1_mean"]
    return out


def compare_models(names, X, y, cv=None, **build_kwargs) -> pd.DataFrame:
    """Une ligne par modèle (réglages par défaut ou `build_kwargs`), validation croisée sur (X, y)."""
    rows = {n: cross_validate_model(build_model(n, **build_kwargs), X, y, cv) for n in names}
    return pd.DataFrame(rows).T


def oof_scores(model, X, y, cv=None) -> tuple[np.ndarray, float]:
    """Scores hors-pli (chaque ligne est notée par un modèle qui ne l'a pas vue) et seuil par défaut."""
    method = "predict_proba" if hasattr(model, "predict_proba") else "decision_function"
    scores = cross_val_predict(model, X, y, cv=cv or default_cv(), method=method)
    if scores.ndim == 2:
        scores = scores[:, 1]
    return scores, default_threshold(model)


def plot_confusion_matrices(y_true, preds: dict, normalize: str | None = "true", ncols: int = 3):
    nrows = int(np.ceil(len(preds) / ncols))
    fig, axes = _pyplot().subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)
    for ax, (name, pred) in zip(axes.ravel(), preds.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_true,
            pred,
            normalize=normalize,
            display_labels=["mauvais", "bon"],
            cmap="Purples",
            colorbar=False,
            values_format=".2f" if normalize else "d",
            ax=ax,
        )
        ax.set_title(name)
    for ax in axes.ravel()[len(preds) :]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_roc_curves(y_true, scores: dict):
    fig, ax = _pyplot().subplots(figsize=(6.5, 5.5))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_true, s):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="hasard")
    ax.set(xlabel="Taux de faux positifs", ylabel="Taux de vrais positifs (rappel)", title="Courbes ROC")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_pr_curves(y_true, scores: dict):
    fig, ax = _pyplot().subplots(figsize=(6.5, 5.5))
    for name, s in scores.items():
        p, r, _ = precision_recall_curve(y_true, s)
        ax.plot(r, p, label=f"{name} (AP = {average_precision_score(y_true, s):.3f})")
    ax.axhline(np.mean(y_true), color="k", ls="--", lw=1, label="hasard (prévalence)")
    ax.set(xlabel="Rappel", ylabel="Précision", title="Courbes précision-rappel")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_learning_curves(models: dict, X, y, scoring: str = "f1", cv=None, ncols: int = 3):
    """Score d'entraînement vs validation selon la taille du train : diagnostic biais/variance."""
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = _pyplot().subplots(nrows, ncols, figsize=(4.8 * ncols, 3.8 * nrows), squeeze=False)
    for ax, (name, model) in zip(axes.ravel(), models.items()):
        sizes, tr, va = learning_curve(
            model, X, y, cv=cv or default_cv(), scoring=scoring, train_sizes=np.linspace(0.1, 1.0, 6), n_jobs=-1
        )
        for scores, label, color in ((tr, "entraînement", "#8c2d4b"), (va, "validation (CV)", "#2b6a99")):
            ax.plot(sizes, scores.mean(axis=1), "o-", color=color, label=label)
            ax.fill_between(
                sizes, scores.mean(axis=1) - scores.std(axis=1), scores.mean(axis=1) + scores.std(axis=1),
                color=color, alpha=0.15,
            )
        ax.set(title=name, xlabel="taille du train", ylabel=scoring)
        ax.set_ylim(0, 1.05)
    axes.ravel()[0].legend(fontsize=8)
    for ax in axes.ravel()[len(models) :]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def best_f1_threshold(y_true, scores) -> float:
    """Seuil maximisant le F1 de la classe positive (à estimer sur des scores hors-pli, jamais sur le test)."""
    p, r, t = precision_recall_curve(y_true, scores)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(t[int(np.argmax(f1))])


def evaluate_at_threshold(model, X, y, threshold: float) -> dict:
    """Métriques d'un modèle ajusté, la décision étant `score >= threshold`."""
    scores = get_scores(model, X)
    return compute_metrics(y, (scores >= threshold).astype(int), scores)
