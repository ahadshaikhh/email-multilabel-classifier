from sklearn.linear_model import LogisticRegression
import numpy as np

from sklearn.linear_model import LogisticRegression

class ChainedModel:
    def __init__(self):
        self.model2 = LogisticRegression(max_iter=500, solver='lbfgs')
        self.model3 = LogisticRegression(max_iter=500, solver='lbfgs')
        self.model4 = LogisticRegression(max_iter=500, solver='lbfgs')

    def fit(self, X_vec, y2, y3, y4):
        self.model2.fit(X_vec, y2)
        y2_pred = self.model2.predict(X_vec)

        X3 = self._combine_features(X_vec, y2_pred)
        self.model3.fit(X3, y3)
        y3_pred = self.model3.predict(X3)

        X4 = self._combine_features(X3, y3_pred)
        self.model4.fit(X4, y4)

    def predict(self, X_vec):
        y2_pred = self.model2.predict(X_vec)
        X3 = self._combine_features(X_vec, y2_pred)
        y3_pred = self.model3.predict(X3)
        X4 = self._combine_features(X3, y3_pred)
        y4_pred = self.model4.predict(X4)
        return y2_pred, y3_pred, y4_pred

    def _combine_features(self, X, y_pred):
        y_pred = y_pred.reshape(-1, 1) if len(y_pred.shape) == 1 else y_pred

        # If X is a sparse matrix, convert to dense
        if hasattr(X, 'toarray'):
            X = X.toarray()

        return np.hstack((X, y_pred))

