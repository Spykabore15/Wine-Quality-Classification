import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.data import add_binary_target, drop_duplicate_wines, load_wine_data, split_data
from src.ensemble import ENSEMBLE_MEMBERS, build_voting_ensemble
from src.evaluate import default_threshold, get_scores, oof_scores
from src.models import build_model


@pytest.fixture(scope="module")
def data():
    X_train, X_test, y_train, y_test = split_data(drop_duplicate_wines(add_binary_target(load_wine_data())))
    return X_train.iloc[:1200], X_test.iloc[:200], y_train.iloc[:1200], y_test.iloc[:200]


@pytest.fixture(scope="module")
def members():
    return {
        "rf": build_model("rf", n_estimators=30, n_jobs=1),
        "svm": build_model("svm"),
        "gb": build_model("gb", n_estimators=30),
        "knn": build_model("knn", n_neighbors=15),
    }


def test_default_members_are_the_four_expected_models():
    assert ENSEMBLE_MEMBERS == ("rf", "svm", "gb", "knn")


def test_svm_member_is_calibrated_without_mutating_input(members):
    vote = build_voting_ensemble(members)
    svm = dict(vote.estimators)["svm"]
    assert isinstance(svm.named_steps["clf"], CalibratedClassifierCV)
    assert isinstance(members["svm"].named_steps["clf"], SVC)
    assert isinstance(dict(vote.estimators)["rf"].named_steps["clf"], RandomForestClassifier)


def test_soft_vote_is_mean_of_member_probabilities(data, members):
    X_train, X_test, y_train, _ = data
    vote = build_voting_ensemble(members).fit(X_train, y_train)
    manual = np.mean([est.predict_proba(X_test)[:, 1] for est in vote.estimators_], axis=0)
    assert np.allclose(get_scores(vote, X_test), manual)


def test_vote_outputs_valid_probabilities_and_labels(data, members):
    X_train, X_test, y_train, _ = data
    vote = build_voting_ensemble(members).fit(X_train, y_train)
    s = get_scores(vote, X_test)
    assert ((s >= 0) & (s <= 1)).all()
    assert default_threshold(vote) == 0.5
    assert set(np.unique(vote.predict(X_test))) <= {0, 1}


def test_vote_works_in_cross_validated_scoring(data, members):
    X_train, _, y_train, _ = data
    scores, thr = oof_scores(build_voting_ensemble(members), X_train, y_train)
    assert scores.shape == (len(X_train),) and thr == 0.5


def test_weights_are_forwarded(members):
    vote = build_voting_ensemble(members, weights=[2, 1, 1, 1])
    assert list(vote.weights) == [2, 1, 1, 1]
