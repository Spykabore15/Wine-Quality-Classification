from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier

# Modèles aux biais différents : bagging d'arbres, noyau, boosting, voisins
ENSEMBLE_MEMBERS = ("rf", "svm", "gb", "knn")


def build_voting_ensemble(members: dict, weights=None) -> VotingClassifier:
    """Vote souple (moyenne des probabilités) de pipelines déjà optimisés ; `members` : nom -> pipeline."""
    estimators = []
    for name, pipeline in members.items():
        est = clone(pipeline)
        # SVC n'a pas de probabilités : on les obtient par calibration sigmoïde (Platt) en CV interne
        final = est.named_steps["clf"]
        if not hasattr(final, "predict_proba"):
            est.set_params(clf=CalibratedClassifierCV(final, method="sigmoid", cv=5, ensemble=False))
        estimators.append((name, est))
    return VotingClassifier(estimators, voting="soft", weights=weights, n_jobs=-1)
