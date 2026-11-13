import numpy as np
import pytest

from src.data import add_binary_target, drop_duplicate_wines, load_wine_data, split_data
from src.models import MODEL_NAMES, build_model, make_classifier, model_summary


@pytest.fixture(scope="module")
def splits():
    return split_data(drop_duplicate_wines(add_binary_target(load_wine_data())))


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_fits_and_predicts(splits, name):
    X_train, X_test, y_train, _ = splits
    model = build_model(name).fit(X_train, y_train)
    pred = model.predict(X_test)
    assert pred.shape == (len(X_test),)
    assert set(np.unique(pred)) <= {0, 1}


def test_model_with_selection_and_log_transform(splits):
    X_train, X_test, y_train, _ = splits
    model = build_model("logreg", selection="vif", log_skewed=True).fit(X_train, y_train)
    assert model.predict(X_test).shape == (len(X_test),)


def test_params_override_defaults():
    assert make_classifier("knn", n_neighbors=11).n_neighbors == 11
    assert build_model("svm", C=3.0).named_steps["clf"].C == 3.0


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        make_classifier("nope")


def test_summary_covers_all_models():
    table = model_summary()
    assert list(table.index) == list(MODEL_NAMES)
    assert not table.isna().any().any()
