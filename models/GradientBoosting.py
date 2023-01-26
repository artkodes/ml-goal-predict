import pickle

import pandas as pd
import xgboost as xgb
import streamlit as st
from sklearn.metrics import classification_report


class GradientB:

    def __init__(self, learning_rate=0.1, max_depth=3, n_estimators=100, random_state=123,
                 objective="reg:squarederror"):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.objective = objective

    def train(self, X_train, X_val, y_train, y_val):
        params = {
            "objective": self.objective,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
        }
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_train, y_train)
        # save the model to disk
        filename = 'models/GradientBoosting.sav'
        pickle.dump(self.model, open(filename, 'wb'))

    def load(self):
        filename = 'models/GradientBoosting.sav'
        self.model = pickle.load(open(filename, 'rb'))

    def predict(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        print("Les prédictions du modèle sont :", y_pred)
        st.write("Score :", self.model.score(X_test, y_test))
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()
        st.write(df)
