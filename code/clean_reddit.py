import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Step 1: Download necessary resources
try:
    nltk.download('stopwords')
    nltk.download('punkt')
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")

# Step 2: Load the data
file_path = '../data/one-year-of-tsla-on-reddit-posts.csv'
social_data = pd.read_csv(file_path)

# Combine 'title' and 'selftext' to form the text data
social_data['text'] = social_data['title'].fillna('') + ' ' + social_data['selftext'].fillna('')

# Retain only relevant columns: 'created_utc' and 'text'
social_data = social_data[['created_utc', 'text']]

# Convert 'created_utc' to datetime format
social_data['created_utc'] = pd.to_datetime(social_data['created_utc'], unit='s')

# Drop rows with missing or empty 'text'
social_data = social_data[social_data['text'].str.strip() != '']

# Use NLTK's English stopwords list
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    stop_words = set(["a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
                      "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
                      "to", "was", "were", "will", "with"])

# Function to clean text
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize text
    tokens = text.split()  # Fallback tokenization
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply text cleaning function
social_data['cleaned_text'] = social_data['text'].apply(clean_text)

# Save the cleaned data
output_path = '../data/cleaned_reddit_posts_fixed.csv'
social_data[['created_utc', 'cleaned_text']].to_csv(output_path, index=False)

# Display cleaned data preview and save path
cleaned_preview = social_data[['created_utc', 'cleaned_text']].head()
cleaned_preview, output_path

