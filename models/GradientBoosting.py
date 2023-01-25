import xgboost as xgb
import streamlit as st


class GradientB:

    def __init__(self, learning_rate=0.1, max_depth=3, n_estimators=100, random_state=123,
                   objective="reg:squarederror"):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.objective = objective

    def train(self, X_train, X_val, y_train, y_val):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = {
            "objective": self.objective,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
        }
        self.model = xgb.train(params, dtrain, evals=[(dval, "Validation")], early_stopping_rounds=100,
                               verbose_eval=True)

    def predict(self, X_test, y_test):
        dtest = xgb.DMatrix(X_test, label=y_test)
        ### Effectuons des prédictions sur les données de test
        y_pred = self.model.predict(dtest)
        ### Affichons les résultats
        print("Les prédictions du modèle sont :", y_pred)
        st.write("Les prédictions du modèle sont :", y_pred)

