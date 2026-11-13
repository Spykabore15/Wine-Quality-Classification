import numpy as np
import pandas as pd
import pytest

from src.config import FEATURES
from src.data import add_binary_target, drop_duplicate_wines, load_wine_data, split_data
from src.features import (
    SELECTION_METHODS,
    VIFFilter,
    build_feature_pipeline,
    build_preprocessor,
    build_selector,
    selected_features,
    selection_table,
)


@pytest.fixture(scope="module")
def splits():
    df = drop_duplicate_wines(add_binary_target(load_wine_data()))
    return split_data(df)


def test_drop_duplicates_removes_exact_copies():
    df = add_binary_target(load_wine_data())
    out = drop_duplicate_wines(df)
    assert len(out) < len(df)
    assert not out.duplicated(subset=FEATURES + ["type", "quality"]).any()


def test_split_is_stratified_and_disjoint(splits):
    X_train, X_test, y_train, y_test = splits
    assert abs(len(X_test) / (len(X_train) + len(X_test)) - 0.2) < 0.01
    assert abs(y_train.mean() - y_test.mean()) < 0.01
    assert set(X_train.index).isdisjoint(X_test.index)
    assert abs((X_train["type"] == "red").mean() - (X_test["type"] == "red").mean()) < 0.01


@pytest.mark.parametrize("log_skewed", [False, True])
def test_preprocessor_output(splits, log_skewed):
    X_train, X_test, _, _ = splits
    pre = build_preprocessor(log_skewed=log_skewed).fit(X_train)
    out = pre.transform(X_test)
    assert out.shape == (len(X_test), 12)
    assert "type_white" in out.columns
    assert set(out["type_white"]) <= {0.0, 1.0}
    assert not out.isna().any().any()


def test_preprocessor_scales_with_train_statistics_only(splits):
    X_train, X_test, _, _ = splits
    pre = build_preprocessor().fit(X_train)
    train_out = pre.transform(X_train)
    assert np.allclose(train_out[FEATURES].mean(), 0, atol=1e-9)
    assert np.allclose(train_out[FEATURES].std(ddof=0), 1, atol=1e-9)
    # Le test n'est pas recentré : sa moyenne reste proche mais différente de 0
    assert not np.allclose(pre.transform(X_test)[FEATURES].mean(), 0, atol=1e-9)


def test_vif_filter_drops_collinear_column():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(2, 500))
    X = pd.DataFrame({"a": a, "b": b, "c": a + 0.01 * rng.normal(size=500)})
    kept = VIFFilter(threshold=10).fit(X).get_feature_names_out()
    assert len(kept) == 2 and "b" in kept


def test_vif_filter_keeps_independent_columns():
    X = pd.DataFrame(np.random.default_rng(1).normal(size=(300, 4)), columns=list("abcd"))
    assert len(VIFFilter().fit(X).get_feature_names_out()) == 4


def test_vif_removes_density_on_wine_data(splits):
    X_train, _, y_train, _ = splits
    pipe = build_feature_pipeline("vif").fit(X_train, y_train)
    assert "density" not in selected_features(pipe)
    assert "alcohol" in selected_features(pipe)


@pytest.mark.parametrize("method", SELECTION_METHODS)
def test_every_selector_works_in_pipeline(splits, method):
    X_train, X_test, y_train, _ = splits
    pipe = build_feature_pipeline(method).fit(X_train, y_train)
    out = pipe.transform(X_test)
    assert list(out.columns) == selected_features(pipe)
    assert 1 <= out.shape[1] <= 12


def test_mutual_information_selects_k_features(splits):
    X_train, _, y_train, _ = splits
    pipe = build_feature_pipeline("mi", k=4).fit(X_train, y_train)
    assert len(selected_features(pipe)) == 4


def test_unknown_selector_raises():
    with pytest.raises(ValueError):
        build_selector("nope")


def test_selection_table_marks_alcohol_everywhere(splits):
    X_train, _, y_train, _ = splits
    table = selection_table(X_train, y_train)
    assert table.loc["alcohol", list(SELECTION_METHODS)].all()
    assert table.index.is_unique
