import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    CATEGORICAL_FEATURES,
    DATA_DIR,
    FEATURES,
    GOOD_QUALITY_THRESHOLD,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)


def load_wine_data() -> pd.DataFrame:
    """Fusionne les vins rouges et blancs et ajoute la colonne catégorielle `type`."""
    red = pd.read_csv(DATA_DIR / "winequality-red.csv", sep=";")
    white = pd.read_csv(DATA_DIR / "winequality-white.csv", sep=";")
    red["type"] = "red"
    white["type"] = "white"
    return pd.concat([red, white], ignore_index=True)


def add_binary_target(df: pd.DataFrame, threshold: int = GOOD_QUALITY_THRESHOLD) -> pd.DataFrame:
    """Ajoute `good` = 1 si quality >= threshold, 0 sinon."""
    out = df.copy()
    out["good"] = (out["quality"] >= threshold).astype(int)
    return out


def drop_duplicate_wines(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les doublons exacts (évite qu'un même vin soit dans le train et le test)."""
    subset = FEATURES + CATEGORICAL_FEATURES + ["quality"]
    return df.drop_duplicates(subset=subset).reset_index(drop=True)


def split_data(df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Découpage train/test stratifié sur (cible, type) ; retourne X_train, X_test, y_train, y_test."""
    X = df[FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    strata = y.astype(str) + "_" + df["type"]
    return train_test_split(X, y, test_size=test_size, stratify=strata, random_state=random_state)
