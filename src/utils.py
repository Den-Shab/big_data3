import pandas as pd
import numpy as np
import requests
import sys
import os
import logging
from datetime import datetime, timedelta

def download_online_retail_data(output_dir="/app/data/raw/", url="https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"):
    os.makedirs(output_dir, exist_ok=True)
    excel_file_path = os.path.join(output_dir, "Online Retail.xlsx")
    csv_file_path = os.path.join(output_dir, "Online Retail.csv")

    if not os.path.exists(csv_file_path):
        print(f"Downloading data from {url} to {excel_file_path}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(excel_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")

            print(f"Converting Excel to CSV: {excel_file_path} -> {csv_file_path}...")
            df = pd.read_excel(excel_file_path)
            df.to_csv(csv_file_path, index=False)
            print("Conversion to CSV complete.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading the file: {e}")
            if os.path.exists(excel_file_path):
                os.remove(excel_file_path)
            raise
        except Exception as e:
            print(f"Error processing the Excel file: {e}")
            raise
    else:
        print(f"Dataset already exists at {csv_file_path}.")
    return csv_file_path

def setup_logging(log_file_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger

def generate_dummy_data(output_csv_path, num_rows=1000):
    data = {
        'InvoiceNo': [f'I{i:06d}' for i in range(num_rows)],
        'StockCode': [f'S{np.random.randint(100, 999):03d}' for _ in range(num_rows)],
        'Description': [f'Product {np.random.randint(1, 50)}' for _ in range(num_rows)],
        'Quantity': np.random.randint(1, 100, num_rows),
        'InvoiceDate': [
            (datetime.now() - timedelta(days=np.random.randint(1, 365))).strftime('%d/%m/%Y %H:%M')
            for _ in range(num_rows)
        ],
        'UnitPrice': np.round(np.random.uniform(0.5, 50.0, num_rows), 2),
        'CustomerID': np.random.randint(10000, 20000, num_rows),
        'Country': np.random.choice(['United Kingdom', 'France', 'Germany', 'Spain'], num_rows)
    }
    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)
    print(f"Dummy data saved to {output_csv_path}")