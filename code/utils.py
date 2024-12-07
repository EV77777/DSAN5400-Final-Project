# utils.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
import numpy as np


def load_datasets():
    tsla_data = pd.read_csv("data/cleaned_TSLA_stock_data.csv")
    reddit_data = pd.read_csv("data/cleaned_reddit_posts_fixed.csv")
    return tsla_data, reddit_data

def merge_datasets(tsla_data, reddit_data):
    reddit_data['date'] = pd.to_datetime(reddit_data['created_utc']).dt.date
    tsla_data['date'] = pd.to_datetime(tsla_data['Date']).dt.date
    tsla_data['price_movement'] = tsla_data['Adj Close'].diff().apply(lambda x: 1 if x > 0 else 0)
    return pd.merge(tsla_data, reddit_data, on='date', how='inner')

def split_features_target(data):
    data = data.dropna()
    X = data['cleaned_text']
    y = data['price_movement']  # Assuming binary classification (up/down)
    return X, y

def vectorize_text(X):
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(X), vectorizer

def split_train_test(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_naive_bayes(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def train_knn(X_train, y_train, n_neighbors=3):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    return accuracy, precision, recall, f1, predictions

def sentiment_analysis_bert(X_test):
    sentiment_pipeline = pipeline("sentiment-analysis", device=0)  # Use GPU if available
    results = []
    
    for post in X_test[:100]:  # Process in smaller chunks
        if isinstance(post, str):  # Ensure input is a valid string
            try:
                result = sentiment_pipeline(post)[0]
                results.append(result['label'])
            except Exception as e:
                print(f"Error processing post: {post}, Error: {e}")
                results.append('Error')  # Handle gracefully
    return results

def sentiment_analysis_finbert(X_test):
    finbert_tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    finbert_model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    finbert_pipeline = pipeline("sentiment-analysis", model=finbert_model, tokenizer=finbert_tokenizer, device=0)
    results = []
    for post in X_test[:100]:  # Limiting to first 5 for simplicity
        result = finbert_pipeline(post)[0]
        results.append(result['label'])
    return results

def combine_sentiment_with_stock_data(sentiment_results, stock_data):
    # Assume we have a numerical representation of sentiment (e.g., Positive = 1, Negative = -1)
    sentiment_numeric = [1 if sentiment == 'Positive' else -1 for sentiment in sentiment_results]
    stock_data = stock_data.head(len(sentiment_results))
    stock_data['sentiment'] = sentiment_numeric
    return stock_data

def train_model_with_time_windows(data, time_window):
    # Create features based on the given time window (e.g., 1 day, 7 days)
    features = []
    labels = []
    for i in range(len(data) - time_window - 1):
        window_data = data.iloc[i:i+time_window]
        features.append(window_data[['sentiment', 'Adj Close']].values.flatten())
        labels.append(data.iloc[i+time_window]['price_movement'])
    X = np.array(features)
    y = np.array(labels)
    return train_test_split(X, y, test_size=0.2, random_state=42)