from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Datasets"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "report" / "figures"
MODEL_PATH = MODELS_DIR / "wine_model.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
GOOD_QUALITY_THRESHOLD = 7

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

CATEGORICAL_FEATURES = ["type"]
TARGET = "good"

# Asymétrie > 1 constatée en EDA (01_eda.ipynb)
SKEWED_FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "sulphates",
]
