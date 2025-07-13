from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib

class ChainedModel:
    def _init_(self):
        self.model2 = RandomForestClassifier()
        self.model3 = RandomForestClassifier()
        self.model4 = RandomForestClassifier()

    def fit(self, X_vec, y2, y3, y4):
        # Train on Type 2
        self.model2.fit(X_vec, y2)
        y2_pred = self.model2.predict(X_vec)

        # Train on Type 3 using X + y2_pred
        X3 = np.hstack([X_vec.toarray(), y2_pred.reshape(-1, 1)])
        self.model3.fit(X3, y3)
        y3_pred = self.model3.predict(X3)

        # Train on Type 4 using X + y2_pred + y3_pred
        X4 = np.hstack([X3, y3_pred.reshape(-1, 1)])
        self.model4.fit(X4, y4)

    def predict(self, X_vec):
        y2_pred = self.model2.predict(X_vec)
        X3 = np.hstack([X_vec.toarray(), y2_pred.reshape(-1, 1)])
        y3_pred = self.model3.predict(X3)
        X4 = np.hstack([X3, y3_pred.reshape(-1, 1)])
        y4_pred = self.model4.predict(X4)
        return y2_pred, y3_pred, y4_pred

    def save_models(self, path="outputs/"):
        joblib.dump(self.model2, path + "model2.joblib")
        joblib.dump(self.model3, path + "model3.joblib")
        joblib.dump(self.model4, path + "model4.joblib")