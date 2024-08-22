import streamlit as st
import numpy as np
import nltk
from nltk.corpus import stopwords, twitter_samples
import re, string

# Load NLTK resources
nltk.download('stopwords')
nltk.download('twitter_samples')

# Define your functions (e.g., process_tweet, build_freqs, etc.)
def process_tweet(tweet):
    stemmer = nltk.PorterStemmer()
    stopwords_english = stopwords.words('english')
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tokenizer = nltk.TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and
                word not in string.punctuation):
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)

    return tweets_clean

def build_freqs(tweets, ys):
    yslist = np.squeeze(ys).tolist()
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
    return freqs

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradientDescent(x, y, theta, alpha, num_iters):
    m = x.shape[0]
    for i in range(num_iters):
        z = np.dot(x, theta)
        h = sigmoid(z)
        cost = -1. / m * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))
        theta = theta - (alpha / m) * np.dot(x.T, (h - y))
    return float(cost), theta

def extract_features(tweet, freqs):
    word_l = process_tweet(tweet)
    x = np.zeros((1, 3))
    x[0, 0] = 1
    for word in word_l:
        x[0, 1] += freqs.get((word, 1.0), 0)
        x[0, 2] += freqs.get((word, 0.0), 0)
    return x

def predict_tweet(tweet, freqs, theta):
    x = extract_features(tweet, freqs)
    y_pred = sigmoid(np.dot(x, theta))
    return float(y_pred)  # Ensure it's a float

def test_logistic_regression(test_x, test_y, freqs, theta):
    y_hat = []
    for tweet in test_x:
        y_pred = predict_tweet(tweet, freqs, theta)
        y_hat.append(1 if y_pred > 0.5 else 0)
    accuracy = (y_hat == np.squeeze(test_y)).sum() / len(test_x)
    return accuracy

# Preparing the data
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

train_pos = all_positive_tweets[:4000]
train_neg = all_negative_tweets[:4000]
train_x = train_pos + train_neg
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)

freqs = build_freqs(train_x, train_y)
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :] = extract_features(train_x[i], freqs)
Y = train_y

J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)

# Streamlit App
st.title("Tweet Sentiment Analysis")

user_input = st.text_input("Enter a tweet for sentiment analysis:")

if user_input:
    sentiment = predict_tweet(user_input, freqs, theta)
    st.write(f"Sentiment score: {sentiment:.2f}")

    if sentiment > 0.6:
        st.write("The sentiment of the tweet is **Positive**.")
    elif sentiment < 0.4:
        st.write("The sentiment of the tweet is **Negative**.")
    else:
        st.write("The sentiment of the tweet is **Neutral**.")
