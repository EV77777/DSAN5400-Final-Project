import yfinance as yf
import pandas as pd

# Define the companies and their corresponding stock ticker symbols
companies = {
    "Apple Inc.": "AAPL",
    "Microsoft Corporation": "MSFT",
    "Amazon.com, Inc.": "AMZN",
    "Alphabet Inc. (GOOG)": "GOOG",
    "Meta Platforms, Inc.": "META"
}

# Define the date range for the data
start_date = "2023-01-01"
end_date = "2024-01-01"

def fetch_and_save_stock_data(companies, start_date, end_date):
    """
    Fetch daily stock price data for a list of companies from Yahoo Finance and save to CSV files.
    
    Parameters:
        companies (dict): A dictionary where keys are company names and values are stock ticker symbols.
        start_date (str): The start date for fetching data (format: YYYY-MM-DD).
        end_date (str): The end date for fetching data (format: YYYY-MM-DD).
    """
    for company, ticker in companies.items():
        print(f"Fetching data for {company} ({ticker})...")
        try:
            # Fetch the stock data
            data = yf.download(ticker, start=start_date, end=end_date)
            
            # Save the data to a CSV file
            filename = f"{ticker}_stock_data.csv"
            data.to_csv(filename)
            print(f"Data for {company} saved to {filename}.")
        except Exception as e:
            print(f"Error fetching data for {company}: {e}")

if __name__ == "__main__":
    fetch_and_save_stock_data(companies, start_date, end_date)
