import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.config import MODEL_PATH

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Nom de champ de l'API -> colonne du jeu de données
COLUMNS = {
    "fixed_acidity": "fixed acidity",
    "volatile_acidity": "volatile acidity",
    "citric_acid": "citric acid",
    "residual_sugar": "residual sugar",
    "chlorides": "chlorides",
    "free_sulfur_dioxide": "free sulfur dioxide",
    "total_sulfur_dioxide": "total sulfur dioxide",
    "density": "density",
    "pH": "pH",
    "sulphates": "sulphates",
    "alcohol": "alcohol",
    "type": "type",
}


class WineFeatures(BaseModel):
    """Mesures physico-chimiques d'un vin ; les bornes écartent les valeurs physiquement absurdes."""

    fixed_acidity: float = Field(ge=0, le=20)
    volatile_acidity: float = Field(ge=0, le=3)
    citric_acid: float = Field(ge=0, le=3)
    residual_sugar: float = Field(ge=0, le=100)
    chlorides: float = Field(ge=0, le=1)
    free_sulfur_dioxide: float = Field(ge=0, le=400)
    total_sulfur_dioxide: float = Field(ge=0, le=700)
    density: float = Field(ge=0.9, le=1.1)
    pH: float = Field(ge=2, le=5)
    sulphates: float = Field(ge=0, le=3)
    alcohol: float = Field(ge=5, le=20)
    type: Literal["red", "white"]

    model_config = {
        "json_schema_extra": {
            "example": {
                "fixed_acidity": 7.0, "volatile_acidity": 0.27, "citric_acid": 0.36, "residual_sugar": 20.7,
                "chlorides": 0.045, "free_sulfur_dioxide": 45.0, "total_sulfur_dioxide": 170.0, "density": 1.001,
                "pH": 3.0, "sulphates": 0.45, "alcohol": 8.8, "type": "white",
            }
        }
    }


class Prediction(BaseModel):
    prediction: Literal[0, 1]
    label: Literal["bon", "mauvais"]
    probability: float
    threshold: float
    model: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.getenv("WINE_MODEL_PATH", MODEL_PATH))
    # Artefact produit par `python -m src.train` : fichier de confiance, pas une entrée utilisateur
    app.state.artifact = joblib.load(path) if path.exists() else None
    yield


app = FastAPI(title="Wine Quality Classification", version="1.0.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health(request: Request):
    artifact = request.app.state.artifact
    return {"status": "ok" if artifact else "model_missing", "model_loaded": artifact is not None,
            "model": artifact["name"] if artifact else None}


@app.post("/predict", response_model=Prediction)
def predict(wine: WineFeatures, request: Request):
    artifact = request.app.state.artifact
    if artifact is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé : lancer `python -m src.train`.")
    row = {COLUMNS[k]: v for k, v in wine.model_dump().items()}
    X = pd.DataFrame([row])[artifact["features"]]
    probability = float(artifact["model"].predict_proba(X)[0, 1])
    good = probability >= artifact["threshold"]
    return Prediction(
        prediction=int(good),
        label="bon" if good else "mauvais",
        probability=round(probability, 4),
        threshold=round(artifact["threshold"], 4),
        model=artifact["name"],
    )
