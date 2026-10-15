"""
RetailPulse - Raw Data Loader
Loads the raw 2-year Online Retail II dataset across sheets with validation and logging.
"""

import os
from typing import Optional, List
import pandas as pd
from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger("data_loader")

def load_raw_data(filepath: Optional[str] = None, sheet_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Loads raw transaction data from Excel or CSV files.
    For UCI Online Retail II, it reads and concatenates both yearly sheets.
    
    Args:
        filepath: Path to the raw data file (defaults to config path).
        sheet_names: Specific list of sheets to load (defaults to config sheets).
        
    Returns:
        pd.DataFrame: Combined raw transactions dataframe.
    """
    cfg = load_config()
    
    if filepath is None:
        raw_dir = cfg["paths"]["raw_data_dir"]
        raw_filename = cfg["data"]["raw_filename"]
        filepath = os.path.join(raw_dir, raw_filename)
        
        # Fallback search if designated file doesn't exist yet
        if not os.path.exists(filepath):
            candidates = [
                os.path.join(raw_dir, f) for f in os.listdir(raw_dir) 
                if f.lower().endswith((".xlsx", ".csv"))
            ] if os.path.exists(raw_dir) else []
            if candidates:
                filepath = candidates[0]
            else:
                raise FileNotFoundError(
                    f"Raw dataset not found at '{filepath}'. Please run `python scripts/download_data.py` first."
                )

    logger.info(f"Loading raw dataset from: {filepath}")
    
    # Handle CSV format directly
    if filepath.lower().endswith(".csv"):
        df = pd.read_csv(filepath)
        logger.info(f"Successfully loaded CSV with {len(df):,} rows and {len(df.columns)} columns.")
        return df

    # Handle Excel format (Online Retail II multi-sheet)
    if sheet_names is None:
        sheet_names = cfg["data"].get("sheets", ["Year 2009-2010", "Year 2010-2011"])

    excel_file = pd.ExcelFile(filepath)
    available_sheets = excel_file.sheet_names
    logger.info(f"Available workbook sheets: {available_sheets}")

    dfs = []
    # If specified sheets exist, load and concatenate
    target_sheets = [s for s in sheet_names if s in available_sheets]
    if not target_sheets:
        target_sheets = [available_sheets[0]]

    for sheet in target_sheets:
        logger.info(f"Reading sheet: '{sheet}'...")
        sheet_df = pd.read_excel(excel_file, sheet_name=sheet)
        logger.info(f"  Sheet '{sheet}' loaded with {len(sheet_df):,} rows.")
        dfs.append(sheet_df)

    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined total raw records: {len(combined_df):,} rows and {len(combined_df.columns)} columns.")
    return combined_df
