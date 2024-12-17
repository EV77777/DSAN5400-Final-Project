from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def split_features_target(data):
    """
    Split data into features (cleaned_text) and target (price_movement).
    Args:
        data (DataFrame): Merged dataset.
    Returns:
        X (Series): Cleaned text data.
        y (Series): Price movement labels (binary classification).
    """
    data = data.dropna()
    X = data['cleaned_text']
    y = data['price_movement']  # Assuming binary classification (up/down)
    return X, y

def vectorize_text(X):
    """
    Convert text data into numerical representations using CountVectorizer.
    Args:
        X (Series): Text data.
    Returns:
        X_vectorized (sparse matrix): Vectorized text data.
        vectorizer (CountVectorizer): Fitted CountVectorizer instance.
    """
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(X), vectorizer

def split_train_test(X, y):
    """
    Split features and labels into training and test datasets.
    Args:
        X: Feature data (e.g., vectorized text).
        y: Target labels.
    Returns:
        X_train,
    """
    return train_test_split(X, y, test_size=0.2, random_state=42)