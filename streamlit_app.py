import streamlit as st
import joblib
import re
import string

# Define clean_text
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Define prediction function
def predict_news(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = rf_model.predict(vec)[0]
    return "❌ FAKE NEWS" if pred == 1 else "✅ REAL NEWS"

# Load model and vectorizer
rf_model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# UI
st.title("Fake News Detector")
user_input = st.text_area("Enter News Text:")
if st.button("Predict"):
    result = predict_news(user_input)
    st.subheader(f"The news is: {result}")
