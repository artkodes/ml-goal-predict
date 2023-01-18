import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import streamlit as st


class Rf:
    def __init__(self, n_estimators=100, max_depth=3, random_state=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        # self.model = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state)
        self.model = RandomForestClassifier(random_state=self.random_state)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_train, y_train, X_test, y_test, y_pred):
        st.write("Le score sur les données d'entraînement est :", self.model.score(X_train, y_train))
        st.write("Le score sur les données d'évaluation est :", self.model.score(X_test, y_test))
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()
        st.write(df)
