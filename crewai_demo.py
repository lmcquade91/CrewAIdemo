import streamlit as st
import joblib
import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define custom transformer (must match your original pipeline)
class RobertaSentimentScorer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='cardiffnlp/twitter-roberta-base-sentiment-latest'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        scores = []
        for text in X:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                sentiment_score = probs[2].item() - probs[0].item()
            scores.append([sentiment_score])
        return np.array(scores)

# Load pipeline
pipeline = joblib.load("/content/drive/MyDrive/Data /sentiment_pipeline.pkl'")  # make sure this path is correct in your app

# Email simulation function
def generate_email(name, email, sentiment):
    if sentiment == 'negative':
        return f"""
        To: {email}
        Subject: We're sorry, {name}

        Hi {name},

        We're really sorry to hear that you had a negative experience.
        Your feedback is important, and we're working hard to improve.

        Our team will follow up with you shortly.

        Regards,
        Customer Experience Team
        """
    else:
        return f"""
        To: {email}
        Subject: Thank you, {name}!

        Hi {name},

        Thank you for your positive feedback!
        We're so glad you enjoyed your stay.

        Hope to welcome you again soon!

        Regards,
        Customer Experience Team
        """

# Streamlit UI
st.title("💬 Review Sentiment & Auto Email Demo")

name = st.text_input("Your Name")
email = st.text_input("Your Email")
review = st.text_area("Your Review")

if st.button("Submit Review"):
    if name and email and review:
        prediction = pipeline.predict([review])[0]
        sentiment = "positive" if prediction == 2 else "negative"
        st.success(f"Sentiment detected: **{sentiment.upper()}**")

        email_text = generate_email(name, email, sentiment)
        st.markdown("### 📧 Email Preview:")
        st.code(email_text)
    else:
        st.warning("Please fill in all fields.")

