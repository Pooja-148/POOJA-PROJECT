# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import re
import string

# load dataset
df = pd.read_csv(r'C:\Users\ADMIN\Downloads\fake_news_dataset (1).csv')
print(df.head())
print(df.info())

# Check for nulls
df = df.dropna()

# Combine title and text if available
if 'title' in df.columns and 'text' in df.columns:
    df['content'] = df['title'] + " " + df['text']
else:
    df['content'] = df[df.columns[0]]  # fallback

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

df['content'] = df['content'].apply(clean_text)

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(df['content'])

y = df['label']  # Assumes 'label' is 0 (real) and 1 (fake)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

def evaluate(model, name):
    y_pred = model.predict(X_test)
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.show()

evaluate(lr_model, "Logistic Regression")
evaluate(rf_model, "Random Forest")

def plot_roc(model, name):
    y_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{name} AUC: {auc(fpr, tpr):.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

plot_roc(lr_model, "Logistic Regression")
plot_roc(rf_model, "Random Forest")
plt.show()

# Save this as app.py
# streamlit run app.py
import streamlit as st

def predict_news(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = rf_model.predict(vec)[0]
    return "Fake" if pred == 1 else "Real"

st.title("Fake News Detector")
user_input = st.text_area("Enter News Text:")
if st.button("Predict"):
    result = predict_news(user_input)
    st.subheader(f"The news is: {result}")
