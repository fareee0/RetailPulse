"""
RetailPulse - Phase 1 Pipeline Runner
Executes raw dataset loading, cleaning, validation, audit report generation, and persistence.
"""

import os
import sys
import json
from pathlib import Path

# Add root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.utils.config import load_config
from src.data.load import load_raw_data
from src.data.clean import clean_retail_data

logger = get_logger("run_clean")

def main():
    logger.info("Starting Phase 1 Execution: Raw Ingestion -> Clean -> Audit")
    cfg = load_config()

    # 1. Load Raw Data
    raw_df = load_raw_data()
    logger.info(f"Loaded Raw Dataset: {len(raw_df):,} rows.")

    # 2. Run Cleaning Pipeline
    clean_df, audit = clean_retail_data(raw_df, config=cfg, save_output=True)

    # 3. Save Audit Report
    proc_dir = cfg["paths"]["processed_data_dir"]
    audit_file = os.path.join(proc_dir, "cleaning_audit.json")
    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    logger.info(f"Audit log saved to: {audit_file}")
    logger.info(f"Cleaned dataset saved to: {cfg['data']['cleaned_parquet']} and {cfg['data']['cleaned_csv']}")
    
    print("\n" + "=" * 65)
    print("PHASE 1 - DATA LAYER COMPLETE")
    print("=" * 65)
    print(f"Raw Input Rows:          {audit['raw_rows']:,}")
    print(f"Cancellations Filtered:  {audit.get('cancellation_rows', 0):,}")
    print(f"Final Cleaned Rows:      {audit['final_rows']:,}")
    print(f"Retention Rate:          {audit['retention_rate_pct']}%")
    print(f"Distinct Customers:      {audit['unique_customers']:,}")
    print(f"Distinct SKUs:           {audit['unique_products']:,}")
    print(f"Distinct Countries:      {audit['unique_countries']:,}")
    print(f"Total Revenue:           GBP {audit['total_revenue']:,.2f}")
    print(f"Date Span:               {audit['date_min']} to {audit['date_max']}")
    print("=" * 65)

if __name__ == "__main__":
    main()
