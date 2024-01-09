from typing import Optional, Union

import numpy as np
from sklearn.metrics import confusion_matrix


def adjusted_aucpr(
    y_pred: Union[list, np.ndarray],
    y_true: Union[list, np.ndarray],
    prevalence: float = 0.1,
    threshold: float = 0.50,
    n_classes: Optional[int] = 2,
) -> np.ndarray:

    assert n_classes == 2, f"Only binary classification is currently supported, received n_classes={n_classes}."

    conf_mat = confusion_matrix(
        y_true=y_true, y_pred=y_pred, labels=range(n_classes), normalize=None
    )
    adjusted_confmat = 

    return None
