import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# ===== Custom RoBERTa Transformer =====
class RobertaSentimentScorer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='cardiffnlp/twitter-roberta-base-sentiment-latest'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

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

# ===== Sample Training for Classifier Logic (simulated) =====
def train_demo_pipeline():
    X = np.array([[-0.8], [-0.5], [-0.2], [0.0], [0.4], [0.6], [0.9]])  # Simulated sentiment scores
    y = np.array([0, 0, 0, 1, 1, 1, 1])  # 0 = negative, 1 = positive

    pipe = Pipeline([
        ('pca', PCA(n_components=1)),
        ('clf', LogisticRegression())
    ])
    pipe.fit(X, y)
    return pipe

# ===== Load models once =====
@st.cache_resource
def load_sentiment_components():
    scorer = RobertaSentimentScorer()
    classifier = train_demo_pipeline()
    return scorer, classifier

scorer, classifier = load_sentiment_components()

# ===== Streamlit UI =====
st.title("💬 Sentiment-Driven Auto Email Demo (RoBERTa + LogisticRegression)")

name = st.text_input("Guest Name")
email = st.text_input("Guest Email")
review = st.text_area("Guest Review")

if st.button("Analyze and Generate Email"):
    if name and email and review:
        # Transform using RoBERTa
        score = scorer.transform([review])[0][0]
        sentiment_class = classifier.predict([[score]])[0]
        sentiment = "positive" if sentiment_class == 1 else "negative"

        st.success(f"RoBERTa score: `{score:.3f}` → **{sentiment.upper()}**")

        # Generate email
        if sentiment == "negative":
            email_text = f"""
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
            email_text = f"""
            To: {email}
            Subject: Thank you, {name}!

            Hi {name},

            Thank you for your positive feedback!
            We're so glad you enjoyed your stay.

            Hope to welcome you again soon!

            Regards,
            Customer Experience Team
            """

        st.markdown("### 📧 Generated Email")
        st.code(email_text)
    else:
        st.warning("Please complete all fields.")
