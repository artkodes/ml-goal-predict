import streamlit as st

from process import EventProcessor, DataEncoder, DataSplitter

if __name__ == '__main__':
    st.title('Uber pickups in NYC')
    file = st.file_uploader('Upload a CSV file')

    if file is not None:
        info_file = 'data/ginf.csv'
        processor = EventProcessor(events_file=file, info_file=info_file)
        processor.process()

        encodeur = DataEncoder(processor.shots)
        encodeur.encode_categorical_variables()

        splitter = DataSplitter(
            data=encodeur.encoded_data.iloc[:, :-1],
            target=encodeur.encoded_data.iloc[:, -1]
        )
        splitter.split()

        #st.write(processor.shots.head(100))

        print(processor.shots.isna().sum())