import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.data import add_binary_target, drop_duplicate_wines, load_wine_data, split_data
from src.ensemble import build_voting_ensemble
from src.models import build_model

VALID = {
    "fixed_acidity": 6.6, "volatile_acidity": 0.29, "citric_acid": 0.31, "residual_sugar": 3.9, "chlorides": 0.027,
    "free_sulfur_dioxide": 39, "total_sulfur_dioxide": 96, "density": 0.9904, "pH": 3.24, "sulphates": 0.6,
    "alcohol": 12.6, "type": "white",
}


@pytest.fixture(scope="module")
def artifact_path(tmp_path_factory):
    X_train, _, y_train, _ = split_data(drop_duplicate_wines(add_binary_target(load_wine_data())))
    members = {
        "rf": build_model("rf", n_estimators=30, n_jobs=1),
        "svm": build_model("svm"),
        "knn": build_model("knn", n_neighbors=15),
    }
    vote = build_voting_ensemble(members).set_params(n_jobs=1).fit(X_train.iloc[:1500], y_train.iloc[:1500])
    path = tmp_path_factory.mktemp("model") / "wine_model.joblib"
    joblib.dump({"model": vote, "threshold": 0.35, "name": "vote", "features": list(X_train.columns)}, path)
    return path


@pytest.fixture
def client(artifact_path, monkeypatch):
    monkeypatch.setenv("WINE_MODEL_PATH", str(artifact_path))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("WINE_MODEL_PATH", str(tmp_path / "absent.joblib"))
    with TestClient(app) as c:
        yield c


def test_health_reports_loaded_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "model": "vote"}


def test_predict_returns_consistent_decision(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability"] <= 1
    assert body["threshold"] == 0.35
    assert body["prediction"] == int(body["probability"] >= 0.35)
    assert body["label"] == ("bon" if body["prediction"] else "mauvais")


def test_predict_ranks_a_strong_wine_above_a_weak_one(client):
    weak = {**VALID, "alcohol": 8.5, "volatile_acidity": 0.8, "chlorides": 0.12, "density": 0.999}
    strong = client.post("/predict", json=VALID).json()["probability"]
    assert strong > client.post("/predict", json=weak).json()["probability"]


@pytest.mark.parametrize(
    "change",
    [
        {"alcohol": -1},
        {"alcohol": 80},
        {"pH": 0},
        {"density": 5},
        {"type": "rose"},
        {"fixed_acidity": "abc"},
    ],
)
def test_invalid_values_are_rejected(client, change):
    assert client.post("/predict", json={**VALID, **change}).status_code == 422


def test_missing_field_is_rejected(client):
    body = {k: v for k, v in VALID.items() if k != "sulphates"}
    assert client.post("/predict", json=body).status_code == 422


def test_service_unavailable_without_model(client_without_model):
    health = client_without_model.get("/health").json()
    assert health["model_loaded"] is False and health["status"] == "model_missing"
    assert client_without_model.post("/predict", json=VALID).status_code == 503


def test_web_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "/predict" in r.text
