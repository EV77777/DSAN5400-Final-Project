# DSAN5400-Final-Project

## Labor Division

Yiwei Qi & Samantha Moon: Data Cleaning and EDA

Yubo Xing & Jiachen Gao: Modeling

## Milestone

12.1-12.2 Data Cleaning

12.2-12.5 EDA & Modeling

12.6-12.7 Prepare for Presentation

12.8-12.9 Review

## File Tree Structure

project/
├── src/
│   ├── data/
│   │   ├── fetch_news.py       # Script for fetching financial news articles
│   │   ├── preprocess.py       # Script for preprocessing text and data
│   │   └── fetch_prices.py     # Script for fetching stock price data
│   ├── models/
│   │   ├── naive_bayes.py      # Implementation of Naive Bayes model
│   │   ├── knn.py              # Implementation of K-Nearest Neighbors (KNN) model
│   │   ├── bert.py             # Implementation of BERT model for sentiment analysis
│   │   └── finbert.py          # Implementation of FinBERT model for financial sentiment analysis
│   ├── utils/                  # Utility functions and helper scripts
│   └── main.py                 # Main entry point for running the project
├── tests/                      # Directory for unit tests
├── pyproject.toml              # Project configuration file (modern Python packaging)
├── README.md                   # Project documentation and instructions
├── environment.yml             # Environment configuration file (dependencies and libraries)
└── architecture.drawio         # Diagram showing the architecture of the project

## 数据准备
数据来源：
金融新闻：通过 Reuters API、Bloomberg API。
社交媒体：使用 Twitter API 和 Reddit API。
股票价格：Yahoo Finance 或 Alpha Vantage API。
数据处理：
编写 Python 脚本获取和清理数据。
清理社交媒体数据中的噪音（如广告、非相关内容）。
对文本进行预处理：
标准化（去标点、小写化等）。
分词。
停用词去除。

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

内容大纲：
研究背景与意义。
项目目标与问题陈述。
数据与方法。
模型与结果：
展示模型性能对比（表格或图形）。
演示部分预测结果。
结论与未来工作。
工具：
使用 PowerPoint 或 Google Slides。
绘制流程图（如模型架构图、数据处理流程图）
