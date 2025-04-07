import streamlit as st
import os
import gdown
import zipfile
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@st.cache_resource
def load_roberta():
    model_path = "./models/roberta"

    if not os.path.exists(model_path):
        st.info("📥 Downloading RoBERTa model...")
        zip_path = "roberta.zip"
        
        # Download from your Google Drive
        url = "https://drive.google.com/uc?id=1L8NzdLZGLBL5EEGXXYNEqlxXnLMRqHwp"
gdown.download(url, output="roberta.zip", quiet=False)

        st.info("🗂️ Unzipping model files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("models")  # Unzips to ./models/roberta

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

# ✅ LOAD THE MODEL
tokenizer, model = load_roberta()

# ===== Predict sentiment score =====
def get_sentiment_score(review):
    inputs = tokenizer(review, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]
        sentiment_score = probs[2].item() - probs[0].item()  # pos - neg
    return sentiment_score

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
