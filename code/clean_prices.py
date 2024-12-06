import pandas as pd

# Load the uploaded file
file_path = '../data/TSLA_stock_data.csv'
data = pd.read_csv(file_path)

# Remove the first two rows and reset the index
cleaned_data = data.iloc[2:].reset_index(drop=True)

# Rename columns for clarity
cleaned_data.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']

# Convert 'Date' to datetime
cleaned_data['Date'] = pd.to_datetime(cleaned_data['Date'])

# Ensure numerical columns are in the correct data type
numerical_columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
cleaned_data[numerical_columns] = cleaned_data[numerical_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with any missing values (optional, depending on data quality needs)
cleaned_data = cleaned_data.dropna()

# Save cleaned data to a new CSV file
output_path = '../data/cleaned_TSLA_stock_data.csv'
cleaned_data.to_csv(output_path, index=False)

# Return cleaned data preview and save path
cleaned_data.head(), output_path

# Remove the first two rows and reset the index
cleaned_data = data.iloc[2:].reset_index(drop=True)

# Rename columns for clarity
cleaned_data.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']

# Convert 'Date' to datetime
cleaned_data['Date'] = pd.to_datetime(cleaned_data['Date'], errors='coerce')

# Ensure numerical columns are in the correct data type
numerical_columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
cleaned_data[numerical_columns] = cleaned_data[numerical_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with any missing values
cleaned_data = cleaned_data.dropna()

# Save cleaned data to a new CSV file
output_path = '../data/cleaned_TSLA_stock_data.csv'
cleaned_data.to_csv(output_path, index=False)

# Return cleaned data preview and save path
cleaned_data.head(), output_path
