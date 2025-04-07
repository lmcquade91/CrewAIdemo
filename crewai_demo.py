import dill
import streamlit as st

# Load the pipeline once
@st.cache_resource
def load_pipeline():
    with open("sentiment_pipeline.pkl", "rb") as f:
        pipeline = dill.load(f)
    return pipeline

sentiment_pipeline = load_pipeline()

# ===== Use the pipeline =====
def get_sentiment_score(review):
    score = sentiment_pipeline.named_steps["roberta"].transform([review])[0][0]
    return score


# ===== Email generator =====
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

# ===== Streamlit UI =====
st.title("💬 Review Sentiment & Auto Email Demo")

name = st.text_input("Your Name")
email = st.text_input("Your Email")
review = st.text_area("Your Review")

if st.button("Submit Review"):
    if name and email and review:
        score = get_sentiment_score(review)
        sentiment = "positive" if score > 0 else "negative"
        st.success(f"Sentiment score: `{score:.3f}` → Detected as **{sentiment.upper()}**")

        email_text = generate_email(name, email, sentiment)
        st.markdown("### 📧 Email Preview:")
        st.code(email_text)
    else:
        st.warning("Please fill in all fields.")
