import numpy as np
from sklearn.model_selection import train_test_split
def train_model_with_time_windows(data, time_window):
    """
    Create features and labels based on sliding time windows.
    Args:
        data (DataFrame): Combined data with sentiment and stock features.
        time_window (int): Number of days in the time window.
    Returns:
        X_train, X_test, y_train, y_test: Training and test splits based on time windows.
    """
    features = []
    labels = []
    for i in range(len(data) - time_window - 1):
        window_data = data.iloc[i:i+time_window]
        features.append(window_data[['sentiment', 'Adj Close']].values.flatten())
        labels.append(data.iloc[i+time_window]['price_movement'])
    X = np.array(features)
    y = np.array(labels)
    return train_test_split(X, y, test_size=0.2, random_state=42)