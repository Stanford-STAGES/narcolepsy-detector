import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin


class GaussianProcessEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self):
        ...

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseEstimator:

        self.classes_, y = np.unique(y, return_inverse=True)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:

        return X

    def predict(self, X: np.ndarray) -> np.ndarray:

        p = self.predict_proba(X)

        return self.classes_[np.argmax(p, axis=1)]
