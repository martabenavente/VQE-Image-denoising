import kagglehub
import os

# Set custom cache path
custom_path = "./_dev-data"
os.environ['KAGGLEHUB_CACHE'] = custom_path

# Download latest version
path = kagglehub.dataset_download("hojjatk/mnist-dataset")
print("Path to dataset files:", path)