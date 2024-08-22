import streamlit as st
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
import string

# Download stopwords if they are not already available
nltk.download('stopwords', quiet=True)

# Load stopwords
stopwords_list = stopwords.words("english")

def preprocess_text(text):
    # Remove punctuation and convert to lowercase
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])

    # Tokenization and remove stopwords
    words = text.split()
    words = [word for word in words if word not in stopwords_list]

    return " ".join(words)

def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity == 0:
        return "Neutral"
    else:
        return "Negative"

# Streamlit app code
st.title("Sentiment Analysis App")

st.write("Enter the text below to analyze its sentiment:")

user_input = st.text_area("Text Input", "")

if st.button("Analyze"):
    if user_input:
        preprocessed_text = preprocess_text(user_input)
        sentiment = analyze_sentiment(preprocessed_text)
        st.write(f"**Preprocessed Text:** {preprocessed_text}")
        st.write(f"**Sentiment:** {sentiment}")
    else:
        st.write("Please enter some text to analyze.")
