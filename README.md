# DSAN5400-Final-Project

## Labor Division

Yiwei Qi: Data Cleaning

Samantha Moon: EDA

Yubo Xing & Jiachen Gao: Modeling

## Milestone

12.1-12.2 Data Cleaning

12.2-12.6 EDA & Modeling

12.6-12.8 Prepare for Presentation

12.8-12.9 Review

## File Tree Structure

project/
│   ├── code/
│   │   ├── fetch_reddit.py     # Script for fetching Reddit posts on TSLA
│   │   ├── fetch_prices.py     # Script for fetching stock price data
│   │   ├── clean_reddit.py     # Script for cleaning Reddit posts on TSLA
│   │   └── clean_prices.py     # Script for cleaning stock price data
│   ├── data/
│   │   ├── TSLA_stock_data.csv # Raw TSLA stock price data
│   │   ├── one-year-of-tsla-on-reddit-posts.csv              # Raw TSLA Reddit posts data
│   │   ├── one-year-of-tsla-on-reddit-comments.csv           # Raw TSLA Reddit comments data (We don't use this dataset)
│   │   ├── cleaned_TSLA_stock_data.csv                       # Cleaned TSLA stock price data
│   │   └── cleaned_reddit_posts_fixed.csv                    # Cleaned TSLA Reddit posts data
│   ├── utils/                  # Utility functions and helper scripts
│   └── main.py                 # Main entry point for running the project
├── tests/                      # Directory for unit tests
├── pyproject.toml              # Project configuration file (modern Python packaging)
├── README.md                   # Project documentation and instructions
└── environment.yml             # Environment configuration file (dependencies and libraries)

## Data Preparation and Cleaning (Just script, not final presentation)

### Data Source

This study constructs a multi-dimensional data set by collecting financial news, social media content, and stock price data to analyze the relationship between sentiment and stock price movements. The following are the specific data sources and how to obtain them:

- Social media data:  Kaggle dataset used: pavellexyr/one-year-of-tsla-on-reddit, containing Reddit posts and comments from July 5, 2021 to July 4, 2022. The data is downloaded and extracted through the Kaggle API to the ../data directory, including the titles and content of all posts mentioning "TSLA", covering multiple investment-related subreddits. (https://www.kaggle.com/datasets/pavellexyr/one-year-of-tsla-on-reddit)
- Stock price data:  Using the Yahoo Finance API, the daily opening, closing, high, and low prices of Tesla (TSLA) between July 5, 2021, and July 4, 2022 were obtained. Download via script and save as ../data/TSLA_stock_data.csv.

### Data Cleaning (https://drive.google.com/drive/folders/16XQv69wSSzYjwO8pb0owF99VqtSgqHUu?usp=drive_link)

The data processing process mainly includes the following steps: 

First, download the Reddit data set through the Kaggle API, and use the Yahoo Finance API to obtain the daily price data of Tesla stock; 

Second, clean the data, including merging the title and body of the Reddit post, and deleting irrelevant content (such as advertisements, invalid links), handle missing values, and convert the timestamp to a standard date format; then, normalize the text data, remove punctuation and special characters, and convert the text to lowercase; 

Finally, use NLTK The tool segments the text and removes common stop words (such as "the" and "and") to generate clean text data to support subsequent analysis. The processed data are stored as cleaned Reddit data and stock price data respectively for further modeling and analysis.

## MODEL DEVELOPMENT AND EXPERIMENTS Sentiment Analysis: 
1. Sentiment classification of financial texts using BERT and FinBERT.
2. Naive Bayes and KNN as baseline models.
3. Evaluation metrics: Accuracy, Precision, Recall, F1 Score.
Stock Price Prediction:
1. Use the output of sentiment analysis, combined with stock price data, to predict the direction of price fluctuations.
2. Experiment with the impact of different time windows (e.g., 1 day, 7 days) on the prediction.

## Software Testing 
- Write unit tests using pytest to ensure that every function and module works properly. 
- Format the code using black and follow PEP 8 specifications.

## Presentation

Content outline:
Research background and significance.
Project objectives and problem statement.
Data and methods.
Model and results:
Show model performance comparison (table or graph).
Demonstrate some prediction results.
Conclusions and future work.
tool:
Use PowerPoint or Google Slides.
Draw flow charts (such as model architecture diagrams, data processing flow charts)
