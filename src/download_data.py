import os
import urllib.request
import zipfile

file_path = "data/raw/default of credit card clients.xls"
zip_path = "data/raw/credit_default.zip"
dataset_url = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"

if os.path.exists(file_path):
    print("Dataset already exists.")
else:
    print("Dataset not found. Downloading...")
    urllib.request.urlretrieve(dataset_url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("data/raw")

    os.remove(zip_path)
    print("Dataset downloaded and extracted successfully.")