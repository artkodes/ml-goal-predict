import pickle
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st


class KNN:
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        # save the model to disk
        filename = 'models/KNN.sav'
        pickle.dump(self.model, open(filename, 'wb'))

    def load(self):
        filename = 'models/KNN.sav'
        self.model = pickle.load(open(filename, 'rb'))

    def predict(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = self.model.score(X_test, y_test)
        st.write("Précision : ", accuracy)
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()
        st.write(df)
