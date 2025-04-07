import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ===== Load VADER once =====
@st.cache_resource
def load_analyzer():
    return SentimentIntensityAnalyzer()

analyzer = load_analyzer()

# ===== Use VADER to get sentiment score =====
def get_sentiment_score(review):
    scores = analyzer.polarity_scores(review)
    return scores['compound']

# ===== Email generator =====
def generate_email(name, email, sentiment):
    if sentiment == 'negative':
        return f"""To: {email}
Subject: We're sorry, {name}

Hi {name},

We're really sorry to hear that you had a negative experience.
Your feedback is important, and we're working hard to improve.

Regards,
Customer Experience Team
"""
    else:
        return f"""To: {email}
Subject: Thank you, {name}!

Hi {name},

Thank you for your positive feedback!
We’re so glad you enjoyed your stay.

Hope to welcome you again soon!

Regards,
Customer Experience Team
"""

# ===== Streamlit UI =====
st.title("💬 AI-Driven Email Generator (No Download Needed)")

name = st.text_input("Guest Name")
email = st.text_input("Guest Email")
review = st.text_area("Guest Review")

if st.button("Analyze & Generate Email"):
    if name and email and review:
        score = get_sentiment_score(review)
        sentiment = "positive" if score > 0 else "negative"
        st.success(f"Sentiment Score: `{score:.3f}` → Detected as **{sentiment.upper()}**")

        email_preview = generate_email(name, email, sentiment)
        st.markdown("### 📧 Auto-Generated Email")
        st.code(email_preview)
    else:
        st.warning("Please complete all fields.")
