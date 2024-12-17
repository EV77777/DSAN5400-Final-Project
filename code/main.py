from utils.loader import load_datasets, merge_datasets
from utils.preprocessing import split_features_target, vectorize_text, split_train_test
from utils.models import train_naive_bayes, train_knn, train_logistic_regression, evaluate_model
from utils.sentiment import sentiment_analysis_bert, sentiment_analysis_finbert, combine_sentiment_with_stock_data
from utils.train_windows import train_model_with_time_windows
import numpy as np
def main():
    # Load data
    tsla_data, reddit_data = load_datasets()
    data = merge_datasets(tsla_data, reddit_data)

    # Preprocess data
    X, y = split_features_target(data)
    X_vectorized, vectorizer = vectorize_text(X)
    X_train, X_test, y_train, y_test = split_train_test(X_vectorized, y)

    # Train Naive Bayes
    nb_model = train_naive_bayes(X_train, y_train)
    nb_accuracy, nb_precision, nb_recall, nb_f1, _ = evaluate_model(nb_model, X_test, y_test)
    print("Naive Bayes Model Evaluation:")
    print(f"Accuracy: {nb_accuracy:.2f}")
    print(f"Precision: {nb_precision:.2f}")
    print(f"Recall: {nb_recall:.2f}")
    print(f"F1 Score: {nb_f1:.2f}\n")

    knn_model = train_knn(X_train, y_train)
    knn_accuracy, knn_precision, knn_recall, knn_f1, _ = evaluate_model(knn_model, X_test, y_test)
    print("K-Nearest Neighbors Model Evaluation:")
    print(f"Accuracy: {knn_accuracy:.2f}")
    print(f"Precision: {knn_precision:.2f}")
    print(f"Recall: {knn_recall:.2f}")
    print(f"F1 Score: {knn_f1:.2f}\n")

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

if __name__ == "__main__":
    main()