from transformers import pipeline, BertTokenizer, BertForSequenceClassification

def sentiment_analysis_bert(X_test):
    """
    Perform sentiment analysis using a BERT-based model.
    Args:
        X_test (Series): Text data to analyze.
    Returns:
        results (list): Sentiment labels (e.g., Positive, Negative).
    """
    sentiment_pipeline = pipeline("sentiment-analysis", device=0)
    results = []
    for post in X_test[:100]:  # Process in smaller chunks
        if isinstance(post, str):
            try:
                result = sentiment_pipeline(post)[0]
                results.append(result['label'])
            except Exception as e:
                print(f"Error processing post: {post}, Error: {e}")
                results.append('Error')  # Handle gracefully
    return results

def sentiment_analysis_finbert(X_test):
    """
    Perform sentiment analysis using the FinBERT model.
    Args:
        X_test (Series): Text data to analyze.
    Returns:
        results (list): Sentiment labels (e.g., Positive, Negative, Neutral).
    """
    finbert_tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    finbert_model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    finbert_pipeline = pipeline("sentiment-analysis", model=finbert_model, tokenizer=finbert_tokenizer, device=0)
    results = []
    for post in X_test[:100]:
        result = finbert_pipeline(post)[0]
        results.append(result['label'])
    return results

def combine_sentiment_with_stock_data(sentiment_results, stock_data):
    """
    Combine sentiment results with stock data.
    Args:
        sentiment_results (list): Sentiment analysis results.
        stock_data (DataFrame): Tesla stock data.
    Returns:
        stock_data (DataFrame): Updated stock data with sentiment scores.
    """
    sentiment_numeric = [1 if sentiment == 'Positive' else -1 for sentiment in sentiment_results]
    stock_data = stock_data.head(len(sentiment_results))
    stock_data['sentiment'] = sentiment_numeric
    return stock_data
