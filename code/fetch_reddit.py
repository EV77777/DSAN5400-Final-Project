import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Explicitly set KAGGLE_CONFIG_DIR to the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['KAGGLE_CONFIG_DIR'] = current_dir  # Set to the script's directory

# Print the current directory for debugging
print(f"KAGGLE_CONFIG_DIR set to: {os.environ['KAGGLE_CONFIG_DIR']}")

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Set the download path for the dataset
download_path = os.path.join(current_dir, "../data")
os.makedirs(download_path, exist_ok=True)

# Specify the Kaggle dataset to download
dataset = "pavellexyr/one-year-of-tsla-on-reddit"
api.dataset_download_files(dataset, path=download_path, unzip=True)  # Download and unzip the dataset

# Print the path where the dataset was downloaded and extracted
print(f"Dataset downloaded and unzipped to: {os.path.abspath(download_path)}")
