import os
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from utils import load_config

"""
def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
"""

config = load_config()
data_dir = config['data']['raw_data_dir']

def download_data(url, dest_dir):
    """
    Download data from a URL and save it to the destination directory.

    :param url: The URL from which to download the file.
    :param dest_dir: The directory where the files will be extracted.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Get the file name from the URL
    filename = url.split('/')[-1]
    filepath = os.path.join(dest_dir, filename)

    # No need to download if the file already exists
    if not os.path.exists(filepath):
        print(f"Downloading: {url}")
        urllib.request.urlretrieve(url, filepath)
        print(f"Downloaded: {filepath}")

def extract_data(filepath, dest_dir):
    """
    Extract a gzip archive.

    :param filepath: Path to the archive.
    :param dest_dir: Directory where the files will be extracted.
    """
    # Extract gzip archive
    if filepath.endswith(".gz"):
        print(f"Extracting: {filepath}")
        with tarfile.open(filepath, "r:gz") as tar:
            tar.extractall(path=dest_dir)
        print(f"Extracted: {dest_dir}")

        # Optional: Remove the gzip file
        # os.remove(filepath)

    """
    # Extract tar archive
    if filepath.endswith(".tar"):
        print(f"Extracting: {filepath}")
        with tarfile.open(filepath, "r") as tar:
            tar.extractall(path=dest_dir)
        print(f"Extracted: {dest_dir}")

        # Remove the tar file
        os.remove(filepath)
    """

def main():
    # Training and validation datasets
    datasets = {
        #"validation_set": "https://uofi.account.box.com/shared/static/m22q747nawtxq8e5iihjulpapwlvucr5.gz",
        #"training_set": "https://uofi.box.com/shared/static/svlkblkh5o4a3q3qwu7iks6r21cmmu64.gz",
        "test_set": "https://uofi.box.com/shared/static/bmtkjrj9g832w9qybjd1yc4l6cyqx6cs.gz",
        "real_set": "https://uofi.box.com/shared/static/7cy1yuahmaiucq857wgo3exln8wvc825.gz"
    }

    # Use ThreadPoolExecutor for downloading
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(download_data, url, data_dir) for url in datasets.values()]
        for future in futures:
            future.result()  # Wait for all downloads to complete

    # Use ProcessPoolExecutor for extraction
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(extract_data, os.path.join(data_dir, url.split('/')[-1]), data_dir) for url in datasets.values()]
        for future in futures:
            future.result()  # Wait for all extractions to complete

if __name__ == "__main__":
    main()
