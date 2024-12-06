import yfinance as yf
import pandas as pd
import os

# Define the companies and their corresponding stock ticker symbols
companies = {
    "Tesla Inc.": "TSLA"  # Update the company and ticker
}

# Define the date range for the data
start_date = "2021-07-05"  # Start date updated
end_date = "2022-07-04"    # End date updated

def fetch_and_save_stock_data(companies, start_date, end_date):
    """
    Fetch daily stock price data for a list of companies from Yahoo Finance and save to CSV files.
    
    Parameters:
        companies (dict): A dictionary where keys are company names and values are stock ticker symbols.
        start_date (str): The start date for fetching data (format: YYYY-MM-DD).
        end_date (str): The end date for fetching data (format: YYYY-MM-DD).
    """
    # Define the path to the "data" folder in the parent directory
    save_path = os.path.join(os.path.dirname(os.getcwd()), "data")
    os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist

    for company, ticker in companies.items():
        print(f"Fetching data for {company} ({ticker})...")
        try:
            # Fetch the stock data
            data = yf.download(ticker, start=start_date, end=end_date)
            
            # Save the data to a CSV file in the "data" folder
            filename = os.path.join(save_path, f"{ticker}_stock_data.csv")
            data.to_csv(filename)
            print(f"Data for {company} saved to {filename}.")
        except Exception as e:
            print(f"Error fetching data for {company}: {e}")

if __name__ == "__main__":
    fetch_and_save_stock_data(companies, start_date, end_date)
