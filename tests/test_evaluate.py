import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.data import add_binary_target, drop_duplicate_wines, load_wine_data, split_data
from src.evaluate import (
    METRICS,
    compare_models,
    compute_metrics,
    default_threshold,
    get_scores,
    oof_scores,
    plot_confusion_matrices,
    plot_learning_curves,
    plot_pr_curves,
    plot_roc_curves,
)
from src.models import build_model


@pytest.fixture(scope="module")
def train():
    X_train, _, y_train, _ = split_data(drop_duplicate_wines(add_binary_target(load_wine_data())))
    return X_train, y_train


def test_compute_metrics_known_values():
    y = np.array([1, 1, 0, 0, 0, 0])
    pred = np.array([1, 0, 1, 0, 0, 0])
    m = compute_metrics(y, pred, np.array([0.9, 0.4, 0.6, 0.1, 0.2, 0.3]))
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert m["roc_auc"] == pytest.approx(0.875)


def test_metrics_without_positive_prediction_do_not_crash():
    m = compute_metrics([0, 1, 0, 1], [0, 0, 0, 0])
    assert m["precision"] == 0 and m["recall"] == 0 and "roc_auc" not in m


def test_get_scores_uses_decision_function_for_svm(train):
    X, y = train
    svm = build_model("svm").fit(X, y)
    assert not hasattr(svm, "predict_proba") and default_threshold(svm) == 0.0
    assert get_scores(svm, X.head(5)).shape == (5,)
    rf = build_model("logreg").fit(X, y)
    s = get_scores(rf, X.head(5))
    assert default_threshold(rf) == 0.5 and ((s >= 0) & (s <= 1)).all()


def test_compare_models_table(train):
    X, y = train
    table = compare_models(["dummy", "logreg"], X, y)
    assert list(table.index) == ["dummy", "logreg"]
    assert {f"{m}_mean" for m in METRICS} <= set(table.columns)
    assert table.loc["dummy", "roc_auc_mean"] == pytest.approx(0.5)
    assert table.loc["logreg", "roc_auc_mean"] > 0.75


def test_oof_scores_cover_every_row_once(train):
    X, y = train
    scores, thr = oof_scores(build_model("logreg"), X, y)
    assert scores.shape == (len(X),) and thr == 0.5
    assert np.isfinite(scores).all()


def test_plots_return_figures(train):
    X, y = train
    scores, _ = oof_scores(build_model("logreg"), X, y)
    pred = (scores > 0.5).astype(int)
    figs = [
        plot_confusion_matrices(y, {"logreg": pred}),
        plot_roc_curves(y, {"logreg": scores}),
        plot_pr_curves(y, {"logreg": scores}),
        plot_learning_curves({"logreg": build_model("logreg")}, X, y),
    ]
    assert all(isinstance(f, plt.Figure) for f in figs)
    plt.close("all")
