import streamlit as st
import joblib
from nltk.stem import PorterStemmer
import re

load_pre = joblib.load("preprocessing.joblib")

bow_vectorizer = joblib.load("bow_vectorizer.joblib")

bow_model = joblib.load("bow_model.joblib")

st.title("Sentiment Analysis")
st.write("Enter a tweet to analyze its sentiment:")

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"[^a-zA-Z]", " ", text)

    words = text.split()

    words = [stemmer.stem(word) for word in words if word not in stop_words]


    return " ".join(words)


input_text = st.text_input("")

if st.button("Predict Tweet"):

    stop_words = load_pre['stopwords']
    stemmer = PorterStemmer()
    clean_tweet = clean_text(input_text)
    st.write(f"cleaned Text : {clean_tweet}")

    vector = bow_vectorizer.transform([clean_tweet])

    prediction = bow_model.predict(vector)

    st.write("model prediction", prediction)

    st.write("Sentiment:", load_pre['reverse_sentiment_map'][prediction[0]])






