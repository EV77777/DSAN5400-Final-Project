# main.py
from utils import load_datasets, merge_datasets, split_features_target, vectorize_text, split_train_test, train_naive_bayes, train_knn, train_logistic_regression, evaluate_model, sentiment_analysis_bert, sentiment_analysis_finbert, combine_sentiment_with_stock_data, train_model_with_time_windows
import numpy as np

# Load data
tsla_data, reddit_data = load_datasets()

# Preprocess data
data = merge_datasets(tsla_data, reddit_data)

X, y = split_features_target(data)

# Vectorize text
X_vectorized, vectorizer = vectorize_text(X)

# Split data
X_train, X_test, y_train, y_test = split_train_test(X_vectorized, y)
# Train Naive Bayes model
nb_model = train_naive_bayes(X_train, y_train)
nb_accuracy, nb_precision, nb_recall, nb_f1, _ = evaluate_model(nb_model, X_test, y_test)
print("Naive Bayes Model Evaluation:")
print(f"Accuracy: {nb_accuracy:.2f}")
print(f"Precision: {nb_precision:.2f}")
print(f"Recall: {nb_recall:.2f}")
print(f"F1 Score: {nb_f1:.2f}\n")


# Train KNN model
knn_model = train_knn(X_train, y_train)
knn_accuracy, knn_precision, knn_recall, knn_f1, _ = evaluate_model(knn_model, X_test, y_test)
print("K-Nearest Neighbors Model Evaluation:")
print(f"Accuracy: {knn_accuracy:.2f}")
print(f"Precision: {knn_precision:.2f}")
print(f"Recall: {knn_recall:.2f}")
print(f"F1 Score: {knn_f1:.2f}\n")

# Sentiment Analysis using BERT and FinBERT
X_train, X_test, y_train, y_test = split_train_test(X, y)
sentiment_results_bert = sentiment_analysis_bert(X_test)
# Output a few sentiment analysis results for reference
print("Sample Sentiment Analysis with BERT (First 3 Results):")
for post, sentiment in zip(X_test[:3], sentiment_results_bert[:3]):
    print(f"Text: {post} Sentiment: {sentiment}")

X_train, X_test, y_train, y_test = split_train_test(X, y)
sentiment_results_finbert = sentiment_analysis_finbert(X_test)
# Output a few sentiment analysis results for reference
print("Sample Sentiment Analysis with FinBERT (First 3 Results):")
for post, sentiment in zip(X_test[:3], sentiment_results_finbert[:3]):
    print(f"Text: {post} Sentiment: {sentiment}")

# Combine sentiment analysis results with stock price data
combined_data_bert = combine_sentiment_with_stock_data(sentiment_results_bert, tsla_data)
combined_data_finbert = combine_sentiment_with_stock_data(sentiment_results_finbert, tsla_data)

# Train model with different time windows (e.g., 1 day, 7 days) and predict price direction
for time_window in [1,7,15,60]:
    print(f"Training with time window: {time_window} day(s)")
    X_train, X_test, y_train, y_test = train_model_with_time_windows(combined_data_bert, time_window)
    lr_model = train_logistic_regression(X_train, y_train)
    lr_accuracy, lr_precision, lr_recall, lr_f1, predictions = evaluate_model(lr_model, X_test, y_test)
    print("Logistic Regression Model Evaluation with Sentiment and Time Window:")
    print(f"Accuracy: {lr_accuracy:.2f}")
    print(f"Precision: {lr_precision:.2f}")
    print(f"Recall: {lr_recall:.2f}")
    print(f"F1 Score: {lr_f1:.2f}\n")

    # Predict the direction of price fluctuations
    positive_predictions = np.sum(predictions == 1)
    negative_predictions = np.sum(predictions == 0)
    print(np.unique(predictions))
    total_predictions = len(predictions)
    increase_percent = (positive_predictions / total_predictions) * 100
    decrease_percent = (negative_predictions / total_predictions) * 100

    print(f"Prediction: The stock will increase with a probability of {increase_percent:.2f}%")
    print(f"Prediction: The stock will decrease with a probability of {decrease_percent:.2f}%\n")
    
    if (increase_percent > decrease_percent):
        print("Prediction: The stock will likely increase")
    else:
        print("Prediction: The stock will likely  decrease")