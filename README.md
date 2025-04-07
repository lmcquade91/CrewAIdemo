🤖 AI-Driven Email Generator – Project Summary
For the purpose of this exercise, we’ve built an AI-powered application that analyzes guest reviews and automatically generates personalized email responses based on the detected sentiment.

Originally, the system was designed to use a fine-tuned RoBERTa sentiment analysis model with a Logistic Regression classifier and PCA for dimensionality reduction. However, due to file size and runtime limitations in the deployment environment (Streamlit Cloud), we were unable to load the RoBERTa model directly.

As a result, we’ve adapted the solution using the lightweight VADER sentiment analysis model, which runs locally with no need for external downloads. This ensures that the application remains fully functional and still demonstrates the core concept:

an AI agent that reads and understands review text, determines sentiment, and generates an appropriate response email.

The logic and interface remain the same — once a user enters a review, the app:

Analyzes the sentiment score

Classifies it as positive or negative

Auto-generates a polite and professional email response based on the result

This solution showcases how even with resource constraints, generative AI workflows can still be implemented effectively.
