from typing import Tuple

import numpy as np
from sklearn import metrics


def specific_sensitivity(
    y_true: np.ndarray, y_score: np.ndarray, specificity: float
) -> Tuple[float, float]:
    fpr, tpr, thresholds = metrics.roc_curve(
        y_true=y_true, y_score=y_score, drop_intermediate=False
    )

    # Find index were we have required specificity
    idx = np.argwhere(1 - fpr >= specificity)[-1]

    # Return value is the sensitivity at at least specificity.
    return tpr[idx], thresholds[idx]


def specific_sensitivity_fn(estimator, X, y, specificity):
    y_score = estimator.predict_proba(X)

    if y_score.ndim > 1:
        y_score = y_score[:, 1]

    sensitivity, threshold = specific_sensitivity(
        y_true=y, y_score=y_score, specificity=specificity
    )
    return sensitivity
