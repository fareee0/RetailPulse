"""
RetailPulse - Dataset Acquisition Script
Downloads the UCI Online Retail II dataset (or provides Kaggle download instructions).
"""

import os
import sys
import zipfile
import argparse
import urllib.request
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger("download_data")

def download_uci(raw_dir: str, uci_url: str) -> str:
    """
    Downloads and extracts the Online Retail II dataset from the official UCI Machine Learning Repository.
    """
    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, "online_retail_ii.zip")
    excel_path = os.path.join(raw_dir, "online_retail_II.xlsx")

    if os.path.exists(excel_path):
        logger.info(f"Target dataset already exists at: {excel_path} (size: {os.path.getsize(excel_path):,} bytes). Skipping download.")
        return excel_path

    logger.info(f"Connecting to UCI Repository: {uci_url}")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(uci_url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=60) as response, open(zip_path, "wb") as out_file:
            total_size = response.headers.get("Content-Length")
            total_size = int(total_size) if total_size else None
            downloaded = 0
            block_size = 1024 * 1024 # 1MB

            logger.info("Starting streaming download...")
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    pct = (downloaded / total_size) * 100
                    print(f"\r  Downloaded: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%)", end="", flush=True)
                else:
                    print(f"\r  Downloaded: {downloaded / (1024*1024):.1f} MB", end="", flush=True)

        print("")
        logger.info(f"Download complete: {zip_path} ({os.path.getsize(zip_path):,} bytes)")

        # Unzip
        logger.info("Extracting ZIP archive...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(raw_dir)

        # Confirm extraction
        extracted_files = os.listdir(raw_dir)
        logger.info(f"Extracted files in {raw_dir}: {extracted_files}")
        
        # In case the file name inside zip is slightly different, check for .xlsx
        for f in extracted_files:
            if f.lower().endswith(".xlsx") and "retail" in f.lower():
                final_path = os.path.join(raw_dir, f)
                logger.info(f"Successfully unpacked raw dataset to: {final_path}")
                return final_path

        return excel_path

    except Exception as e:
        logger.error(f"Failed to download or extract UCI dataset: {e}", exc_info=True)
        raise

def print_kaggle_instructions():
    """Prints instructions for acquiring the dataset via Kaggle API."""
    docs = """
========================================================================
KAGGLE API SETUP & DOWNLOAD INSTRUCTIONS:
========================================================================
1. Log in to your Kaggle account and navigate to:
   https://www.kaggle.com/settings
2. Under the 'API' section, click 'Create New Token'. This downloads 'kaggle.json'.
3. Place 'kaggle.json' in:
   - Windows: %USERPROFILE%\\.kaggle\\kaggle.json
   - Linux/macOS: ~/.kaggle/kaggle.json
4. Secure the file permissions (Linux/macOS):
   chmod 600 ~/.kaggle/kaggle.json
5. Run the Kaggle CLI command to download the Online Retail II dataset:
   kaggle datasets download -d mashlyn/online-retail-ii-uci -p data/raw --unzip
========================================================================
"""
    print(docs)

def main():
    parser = argparse.ArgumentParser(description="Download RetailPulse raw dataset")
    parser.add_argument("--source", choices=["uci", "kaggle"], default="uci", help="Data source provider")
    args = parser.parse_args()

    cfg = load_config()
    raw_dir = cfg["paths"]["raw_data_dir"]
    uci_url = cfg["data"]["uci_url"]

    if args.source == "uci":
        download_uci(raw_dir, uci_url)
    elif args.source == "kaggle":
        print_kaggle_instructions()

if __name__ == "__main__":
    main()
