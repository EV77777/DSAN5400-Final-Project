from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def train_naive_bayes(X_train, y_train):
    """
    Train a Naive Bayes classifier.
    Args:
        X_train: Training features.
        y_train: Training labels.
    Returns:
        model: Trained Naive Bayes model.
    """
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def train_knn(X_train, y_train, n_neighbors=3):
    """
    Train a K-Nearest Neighbors classifier.
    Args:
        X_train: Training features.
        y_train: Training labels.
        n_neighbors (int): Number of neighbors to consider.
    Returns:
        model: Trained KNN model.
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression classifier.
    Args:
        X_train: Training features.
        y_train: Training labels.
    Returns:
        model: Trained Logistic Regression model.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model using accuracy, precision, recall, and F1 score.
    Args:
        model: Trained model to evaluate.
        X_test: Test features.
        y_test: Test labels.
    Returns:
        accuracy, precision, recall, f1, predictions: Evaluation metrics and predictions.
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    return accuracy, precision, recall, f1, predictions