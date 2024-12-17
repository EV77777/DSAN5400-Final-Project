import pandas as pd

def load_datasets():
    """
    Load Tesla stock data and Reddit posts data from CSV files.
    Returns:
        tsla_data (DataFrame): Tesla stock data.
        reddit_data (DataFrame): Reddit posts data.
    """
    tsla_data = pd.read_csv("data/cleaned_TSLA_stock_data.csv")
    reddit_data = pd.read_csv("data/cleaned_reddit_posts_fixed.csv")
    return tsla_data, reddit_data

def merge_datasets(tsla_data, reddit_data):
    """
    Merge Tesla stock data with Reddit posts data on the date column.
    Args:
        tsla_data (DataFrame): Tesla stock data.
        reddit_data (DataFrame): Reddit posts data.
    Returns:
        merged_data (DataFrame): Merged dataset.
    """
    reddit_data['date'] = pd.to_datetime(reddit_data['created_utc']).dt.date
    tsla_data['date'] = pd.to_datetime(tsla_data['Date']).dt.date
    tsla_data['price_movement'] = tsla_data['Adj Close'].diff().apply(lambda x: 1 if x > 0 else 0)
    return pd.merge(tsla_data, reddit_data, on='date', how='inner')
