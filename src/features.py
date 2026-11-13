import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data

from .config import CATEGORICAL_FEATURES, FEATURES, RANDOM_STATE, SKEWED_FEATURES

SELECTION_METHODS = ("none", "vif", "l1", "rf", "mi")


class VIFFilter(BaseEstimator, TransformerMixin):
    """Retire itérativement la variable de plus fort VIF tant qu'il dépasse `threshold`."""

    def __init__(self, threshold: float = 10.0):
        self.threshold = threshold

    @staticmethod
    def _vif(X: np.ndarray) -> np.ndarray:
        # VIF_i = [R^{-1}]_ii où R est la matrice de corrélation
        corr = np.corrcoef(X, rowvar=False)
        return np.diag(np.linalg.pinv(np.atleast_2d(corr)))

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype=float)
        keep = list(range(X.shape[1]))
        # Les colonnes constantes ont une corrélation indéfinie : on les écarte d'emblée
        keep = [j for j in keep if X[:, j].std() > 0]
        while len(keep) > 1:
            vif = self._vif(X[:, keep])
            worst = int(np.argmax(vif))
            if vif[worst] <= self.threshold:
                break
            keep.pop(worst)
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[keep] = True
        return self

    def transform(self, X):
        check_is_fitted(self, "support_")
        X = validate_data(self, X, dtype=float, reset=False)
        return X[:, self.support_]

    def get_support(self) -> np.ndarray:
        check_is_fitted(self, "support_")
        return self.support_

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "support_")
        names = np.asarray(
            input_features if input_features is not None else self.feature_names_in_,
            dtype=object,
        )
        return names[self.support_]


def build_preprocessor(log_skewed: bool = False) -> ColumnTransformer:
    """Standardise les numériques (log1p optionnel sur les asymétriques) et encode `type` en 0/1."""
    skewed = [c for c in FEATURES if c in SKEWED_FEATURES] if log_skewed else []
    regular = [c for c in FEATURES if c not in skewed]

    transformers = [("num", StandardScaler(), regular)]
    if skewed:
        log_scale = Pipeline(
            [
                ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("skew", log_scale, skewed))
    transformers.append(("cat", OneHotEncoder(drop="if_binary", sparse_output=False), CATEGORICAL_FEATURES))

    pre = ColumnTransformer(transformers, verbose_feature_names_out=False)
    return pre.set_output(transform="pandas")


def build_selector(method: str = "none", **kwargs):
    """Retourne un sélecteur de variables (ou "passthrough"). `kwargs` surcharge les valeurs par défaut."""
    if method == "none":
        return "passthrough"
    if method == "vif":
        return VIFFilter(threshold=kwargs.get("threshold", 10.0))
    if method == "l1":
        est = LogisticRegression(
            l1_ratio=1.0,
            solver="liblinear",
            C=kwargs.get("C", 0.05),
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        return SelectFromModel(est, threshold=1e-8)
    if method == "rf":
        est = RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 300),
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        return SelectFromModel(est, threshold=kwargs.get("threshold", "mean"))

    if method == "mi":

        def score(X, y):
            return mutual_info_classif(X, y, random_state=RANDOM_STATE)

        return SelectKBest(score, k=kwargs.get("k", 6))
    raise ValueError(f"Méthode inconnue : {method!r} (choix : {SELECTION_METHODS})")


def build_feature_pipeline(selection: str = "none", log_skewed: bool = False, **selector_kwargs) -> Pipeline:
    """Prétraitement + sélection ; à compléter par un classifieur (le fit se fait uniquement sur le train)."""
    return Pipeline(
        [
            ("prep", build_preprocessor(log_skewed=log_skewed)),
            ("select", build_selector(selection, **selector_kwargs)),
        ]
    ).set_output(transform="pandas")


def selected_features(pipeline: Pipeline) -> list[str]:
    """Noms des variables conservées par un pipeline `build_feature_pipeline` déjà ajusté."""
    names = pipeline.named_steps["prep"].get_feature_names_out()
    selector = pipeline.named_steps["select"]
    if selector == "passthrough":
        return list(names)
    return list(selector.get_feature_names_out(names))


def selection_table(X_train: pd.DataFrame, y_train: pd.Series, log_skewed: bool = False, **kwargs) -> pd.DataFrame:
    """Tableau booléen (variables x méthodes) des variables retenues par chaque méthode, fit sur le train."""
    columns = {}
    for method in SELECTION_METHODS:
        pipe = build_feature_pipeline(method, log_skewed=log_skewed, **kwargs.get(method, {}))
        pipe.fit(X_train, y_train)
        columns[method] = pd.Series(True, index=selected_features(pipe))
    all_names = build_preprocessor(log_skewed).fit(X_train).get_feature_names_out()
    table = pd.DataFrame(columns).reindex(all_names).fillna(False).astype(bool)
    table["n_methods"] = table.drop(columns="none").sum(axis=1)
    return table.sort_values("n_methods", ascending=False)
