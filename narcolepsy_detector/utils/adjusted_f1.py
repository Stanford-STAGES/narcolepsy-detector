from typing import Optional, Union

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from narcolepsy_detector.utils.specific_sensitivity import specific_sensitivity


def adjusted_f1(
    y_pred: Union[list, np.ndarray],
    y_true: Union[list, np.ndarray],
    data_prevalence: float,
    prevalence: float = 0.1,
    n_classes: Optional[int] = 2,
) -> np.ndarray:
    assert n_classes == 2, f"Only binary classification is currently supported, received n_classes={n_classes}."

    tp_fn_adjustment = prevalence / data_prevalence
    tn_fp_adjustment = (1 - prevalence) / (1 - data_prevalence)
    adjustment_factor = np.array([[tn_fp_adjustment], [tp_fn_adjustment]])

    conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=range(n_classes), normalize=None)
    adjusted_confmat = adjustment_factor * conf_mat

    tn, fp, fn, tp = adjusted_confmat.ravel()

    return 2 * tp / (2 * tp + fp + fn)


def adjusted_f1_fn(estimator, X, y, prevalence, data_prevalence, specificity=0.90, *args, **kwargs):
    y_score = estimator.predict_proba(X)

    if y_score.ndim > 1:
        y_score = y_score[:, 1]

    _, threshold = specific_sensitivity(y_true=y, y_score=y_score, specificity=specificity)

    y_pred = (y_score >= threshold).astype(int)

    adj_f1 = adjusted_f1(y_pred=y_pred, y_true=y, prevalence=prevalence, data_prevalence=data_prevalence)

    return adj_f1
