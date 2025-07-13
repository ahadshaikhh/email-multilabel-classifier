from sklearn.linear_model import LogisticRegression
import numpy as np

from sklearn.linear_model import LogisticRegression

class ChainedModel:
    def __init__(self):
        self.model2 = LogisticRegression(max_iter=1000)
        self.model3 = LogisticRegression(max_iter=1000)
        self.model4 = LogisticRegression(max_iter=1000)

    def fit(self, X_vec, y2, y3, y4):
        print("[Stage 1] Training Type 2 Model on X...")
        self.model2.fit(X_vec, y2)
        y2_pred = self.model2.predict(X_vec)

        print("X shape:", X_vec.shape)  # 👈 Shape before chaining

        print("[Stage 2] Training Type 3 Model on [X + y2_pred]...")
        X3 = self._combine_features(X_vec, y2_pred)
        print("X3 shape (with y2_pred):", X3.shape)  # 👈 Shape after 1st chain
        self.model3.fit(X3, y3)
        y3_pred = self.model3.predict(X3)

        print("[Stage 3] Training Type 4 Model on [X + y2_pred + y3_pred]...")
        X4 = self._combine_features(X3, y3_pred)
        print("X4 shape (with y2_pred + y3_pred):", X4.shape)  # 👈 Shape after 2nd chain
        self.model4.fit(X4, y4)


    def predict(self, X_vec):
        print("[Predict] Type 2")
        y2_pred = self.model2.predict(X_vec)

        print("[Predict] Type 3 using [X + y2_pred]")
        X3 = self._combine_features(X_vec, y2_pred)
        y3_pred = self.model3.predict(X3)

        print("[Predict] Type 4 using [X + y2_pred + y3_pred]")
        X4 = self._combine_features(X3, y3_pred)
        y4_pred = self.model4.predict(X4)

        return y2_pred, y3_pred, y4_pred

    def _combine_features(self, X, y_pred):
        y_pred = y_pred.reshape(-1, 1) if len(y_pred.shape) == 1 else y_pred
        if hasattr(X, 'toarray'):
            X_array = X.toarray()
        else:
            X_array = X
        return np.hstack((X_array, y_pred))


