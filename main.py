import streamlit as st
from sklearn.metrics import classification_report
from models.Rf import Rf
from process import EventProcessor, DataEncoder, DataSplitter


def rf_model(splitter):
    # Random Forest
    rf = Rf(random_state=123)
    rf.fit(X_train=splitter.X_train, y_train=splitter.y_train)
    y_pred = rf.predict(X_test=splitter.X_test)
    rf.evaluate(X_train=splitter.X_train, y_train=splitter.y_train, X_test=splitter.X_test,
                y_test=splitter.y_test, y_pred=y_pred)


if __name__ == '__main__':
    file = st.sidebar.file_uploader('Upload a CSV file')

    if file is not None:
        info_file = 'data/ginf.csv'
        processor = EventProcessor(events_file=file, info_file=info_file)
        processor.process()

        encodeur = DataEncoder(processor.shots)
        encodeur.encode_categorical_variables()

        # predicted value
        target = encodeur.encoded_data['is_goal']
        # features
        data = encodeur.encoded_data.drop(['is_goal'], axis=1)

        splitter = DataSplitter(
            data=data,
            target=target, random_state=123
        )
        splitter.split()

        selected_model = st.sidebar.selectbox('Select a model', ['Random Forest'])

        if selected_model == 'Random Forest':
            rf_model(splitter)
