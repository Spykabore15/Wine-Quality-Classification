import numpy as np
import pytest

from src.data import add_binary_target, drop_duplicate_wines, load_wine_data, split_data
from src.evaluate import best_f1_threshold, evaluate_at_threshold
from src.models import build_model
from src.tuning import SCORING, SPACES, best_params_readable, n_candidates, search_results, tune_model


@pytest.fixture(scope="module")
def small_train():
    X_train, _, y_train, _ = split_data(drop_duplicate_wines(add_binary_target(load_wine_data())))
    return X_train.iloc[:900], y_train.iloc[:900]


def test_best_f1_threshold_on_separable_scores():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
    thr = best_f1_threshold(y, scores)
    assert 0.3 < thr <= 0.7
    assert ((scores >= thr).astype(int) == y).all()


def test_evaluate_at_threshold_changes_with_threshold(small_train):
    X, y = small_train
    model = build_model("logreg").fit(X, y)
    low = evaluate_at_threshold(model, X, y, 0.2)
    high = evaluate_at_threshold(model, X, y, 0.8)
    assert low["recall"] > high["recall"]
    assert low["roc_auc"] == pytest.approx(high["roc_auc"])


def test_every_search_space_matches_model_params():
    for name, space in SPACES.items():
        pipe = build_model(name)
        valid = pipe.get_params()
        assert all(k in valid for k in space["params"]), name


def test_grid_search_nb(small_train):
    X, y = small_train
    search = tune_model("nb", X, y, n_jobs=1)
    table = search_results(search)
    assert len(table) == n_candidates(search) == 7 * 2 * 2
    assert table.loc[0, "rank"] == 1
    assert {f"{m}_mean" for m in SCORING} <= set(table.columns)
    assert set(table["prep"]) == {"std", "log1p"} and set(table["select"]) == {"none", "vif"}
    assert search.best_score_ > 0.6
    assert "var_smoothing" in best_params_readable(search)


def test_random_search_respects_n_iter(small_train):
    X, y = small_train
    search = tune_model("svm", X, y, n_iter=3, n_jobs=1)
    assert n_candidates(search) == 3
    assert search.best_estimator_.predict(X.head(5)).shape == (5,)


def test_unknown_method_and_model(small_train):
    X, y = small_train
    with pytest.raises(ValueError):
        tune_model("nope", X, y)
    with pytest.raises(ValueError):
        tune_model("nb", X, y, method="bayes")
