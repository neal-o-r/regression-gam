import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from bspline import B_spline


class RegressionGAMGCV(BaseEstimator, ClassifierMixin):
    """
        Linear (Identity Link) Regression GAM with GCV to select
        the optimal smoothing parameter
        """

    def __init__(self, ls=20.0, knots=10, maxiter=100):
        self.lambdas = np.logspace(-5, 2, ls)
        self.knots = knots
        self.maxiter = maxiter

    def fit(self, X, Y):
        self.gcvs = np.array([self._fit_model(X, Y, l) for l in self.lambdas])

        self.a, _ = min(self.gcvs, key=lambda t: t[1])

    def predict(self, X):

        return self.a.dot(B_spline(X).T).flatten()

    def _fit_model(self, X, y, l):
        """
                Fit model by augmented design matrix
                """

        B = B_spline(X, n=self.knots)
        n, f = B.shape

        D = np.diff(np.eye(f), 2)

        B_aug = np.vstack((B, np.sqrt(l) * D.T))
        y_aug = np.hstack((y.reshape(1, -1), np.zeros((1, f - 2))))

        a = np.linalg.lstsq(B_aug, y_aug.T)[0].T
        yhat = a.dot(B.T)

        Q = np.linalg.pinv(B.T.dot(B) + l * D.dot(D.T))

        s = np.linalg.norm((yhat - y))
        t = np.sum(np.diag(Q * B.T.dot(B)))

        gcv = s / (n - t) ** 2

        return a, gcv
