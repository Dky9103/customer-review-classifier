import os

# Fix Hugging Face permission issues
os.environ["STREAMLIT_USAGE_STATS"] = "0"
os.environ["HF_HOME"] = "/tmp"
os.environ["XDG_CACHE_HOME"] = "/tmp"

import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")

# UI
st.title("📝 Customer Review Sentiment Classifier")
st.write("This app predicts whether a review is *Positive* or *Negative*.")

review = st.text_area("✍ Enter your customer review here:")

if st.button("Predict Sentiment"):
    if not review.strip():
        st.warning("Please enter some text.")
    else:
        vector = tfidf.transform([review])
        pred = model.predict(vector)[0]
        if pred == 1:
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")