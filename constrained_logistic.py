import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize

class ConstrainedLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0):
        self.C = C
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape


        def loss(params):
            w = params[:-1]
            b = params[-1]
            logits = np.dot(X, w) + b
            probs = 1 / (1 + np.exp(-logits))

            probs = np.clip(probs, 1e-15, 1 - 1e-15)
            ll = -np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))
            reg = (0.5 / self.C) * np.sum(w**2)
            return ll + reg


        def grad(params):
            w = params[:-1]
            b = params[-1]
            logits = np.dot(X, w) + b
            probs = 1 / (1 + np.exp(-logits))
            error = probs - y

            grad_w = np.dot(X.T, error) + (1 / self.C) * w
            grad_b = np.sum(error)
            return np.append(grad_w, grad_b)


        initial_params = np.zeros(n_features + 1)


        bounds = [(0, None)] * n_features + [(None, None)]

        res = minimize(loss, initial_params, method='L-BFGS-B', jac=grad, bounds=bounds)

        self.coef_ = np.array([res.x[:-1]])
        self.intercept_ = np.array([res.x[-1]])
        return self

    def predict_proba(self, X):
        z = np.dot(X, self.coef_[0]) + self.intercept_[0]
        p = 1 / (1 + np.exp(-z))
        return np.vstack([1-p, p]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
