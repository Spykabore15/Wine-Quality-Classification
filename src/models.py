import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE
from .features import build_feature_pipeline

# Modèles de l'énoncé, puis modèles d'ensemble à titre de comparaison
CORE_MODELS = ("logreg", "knn", "tree", "nb", "svm")
ENSEMBLE_MODELS = ("rf", "gb")
MODEL_NAMES = ("dummy", *CORE_MODELS, *ENSEMBLE_MODELS)


def make_classifier(name: str, **params):
    """Classifieur sklearn avec des valeurs par défaut raisonnables ; `params` les surcharge."""
    # class_weight="balanced" compense le déséquilibre 80/20 là où c'est supporté
    factories = {
        "dummy": lambda: DummyClassifier(strategy="prior"),
        "logreg": lambda: LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
        "knn": lambda: KNeighborsClassifier(n_neighbors=5),
        "tree": lambda: DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        "nb": lambda: GaussianNB(),
        "svm": lambda: SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE),
        "rf": lambda: RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        ),
        "gb": lambda: GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    if name not in factories:
        raise ValueError(f"Modèle inconnu : {name!r} (choix : {MODEL_NAMES})")
    return factories[name]().set_params(**params)


def build_model(name: str, selection: str = "none", log_skewed: bool = False, **params) -> Pipeline:
    """Pipeline complet prétraitement -> sélection -> classifieur (étape `clf`)."""
    features = build_feature_pipeline(selection, log_skewed=log_skewed)
    return Pipeline([*features.steps, ("clf", make_classifier(name, **params))])


MODEL_RATIONALE = {
    "dummy": {
        "famille": "Référence",
        "frontière": "Aucune (classe majoritaire)",
        "échelle": "Indifférent",
        "atouts": "Plancher à battre ; montre le piège de l'accuracy (~81 %)",
        "limites": "Aucune capacité prédictive",
    },
    "logreg": {
        "famille": "Linéaire, discriminatif",
        "frontière": "Hyperplan",
        "échelle": "Sensible (standardisation + régularisation)",
        "atouts": "Coefficients interprétables, probabilités calibrées, rapide",
        "limites": "Frontière linéaire ; instable si variables colinéaires (density)",
    },
    "knn": {
        "famille": "Non paramétrique, basé sur la distance",
        "frontière": "Non linéaire, locale",
        "échelle": "Très sensible (distance euclidienne)",
        "atouts": "Aucune hypothèse de forme ; capte des structures locales",
        "limites": "Sensible au bruit et aux variables non pertinentes ; k à valider (biais/variance)",
    },
    "tree": {
        "famille": "Non linéaire, règles (Gini/entropie)",
        "frontière": "Rectangles alignés sur les axes",
        "échelle": "Indifférent",
        "atouts": "Interprétable, robuste aux aberrations, capte les interactions",
        "limites": "Variance élevée, sur-apprentissage si profondeur libre",
    },
    "nb": {
        "famille": "Probabiliste génératif (Bayes)",
        "frontière": "Quadratique (variances par classe)",
        "échelle": "Indifférent",
        "atouts": "Très rapide, peu de données nécessaires",
        "limites": "Suppose indépendance et gaussianité : violées ici (corrélations, asymétries)",
    },
    "svm": {
        "famille": "Marge maximale, à noyau",
        "frontière": "Non linéaire (RBF)",
        "échelle": "Très sensible",
        "atouts": "Bon en dimension modérée, marge souple robuste au bruit",
        "limites": "Hyperparamètres C et gamma à régler ; coût O(n^2) ; peu interprétable",
    },
    "rf": {
        "famille": "Ensemble : bagging d'arbres",
        "frontière": "Non linéaire",
        "échelle": "Indifférent",
        "atouts": "Réduit la variance des arbres ; importance des variables ; erreur OOB",
        "limites": "Moins interprétable ; modèle volumineux",
    },
    "gb": {
        "famille": "Ensemble : boosting séquentiel d'arbres",
        "frontière": "Non linéaire",
        "échelle": "Indifférent",
        "atouts": "Réduit le biais ; souvent le meilleur sur données tabulaires",
        "limites": "Sensible au bruit et aux hyperparamètres ; plus lent à entraîner",
    },
}


def model_summary() -> pd.DataFrame:
    """Tableau de justification théorique des modèles (propriétés, adéquation aux données)."""
    return pd.DataFrame(MODEL_RATIONALE).T.loc[list(MODEL_NAMES)]
