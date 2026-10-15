import os
import sqlite3
import pandas as pd
import numpy as np

def run_etl():
    print("=" * 60)
    print("RetailPulse Automated ETL & Data Transformation Pipeline")
    print("=" * 60)
    
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    excel_path = "RetailPulse_Single_File.xlsx"
    if not os.path.exists(excel_path):
        excel_path = "RetailPulse_Data.xlsx"
    print(f"[*] Ingesting raw dataset from: {excel_path}...")
    
    # Read sheets
    print("[-] Reading RetailPulse_Master sheet...")
    df_master = pd.read_excel(excel_path, sheet_name="RetailPulse_Master")
    print(f"    Loaded {len(df_master):,} transaction rows and {len(df_master.columns)} columns.")
    
    print("[-] Reading Inventory sheet...")
    df_inventory = pd.read_excel(excel_path, sheet_name="Inventory")
    print(f"    Loaded {len(df_inventory):,} inventory snapshot rows.")
    
    print("[-] Reading Forecast sheet...")
    df_forecast = pd.read_excel(excel_path, sheet_name="Forecast")
    print(f"    Loaded {len(df_forecast):,} forecast rows.")
    
    # Data Cleaning and Type Casting
    df_master['OrderDate'] = pd.to_datetime(df_master['OrderDate'])
    df_master['SignupDate'] = pd.to_datetime(df_master['SignupDate'])
    
    # Fill numeric nulls or clean types if needed
    num_cols = ['Quantity', 'UnitPrice', 'DiscountPct', 'Revenue', 'Cost', 'Profit', 
                'RFM_Score', 'Recency', 'Frequency', 'Monetary', 'ChurnFlag', 'ChurnProbability']
    for c in num_cols:
        if c in df_master.columns:
            df_master[c] = pd.to_numeric(df_master[c], errors='coerce')
            
    # Margin calculation
    df_master['MarginPct'] = np.where(df_master['Revenue'] > 0, (df_master['Profit'] / df_master['Revenue']) * 100, 0.0)
    
    # Deduplicate Customer Analytics table for customer-level ML and segmentation
    customer_cols = ['CustomerID', 'CustomerName', 'Gender', 'AgeGroup', 'CustomerCity', 
                     'CustomerState', 'SignupDate', 'PreferredChannel', 'LoyaltyMember', 
                     'Segment', 'RFM_Score', 'Recency', 'Frequency', 'Monetary', 
                     'ChurnFlag', 'ChurnProbability', 'RiskBand']
    
    df_customers = df_master[customer_cols].drop_duplicates(subset=['CustomerID']).copy().reset_index(drop=True)
    # Calculate additional customer metrics from raw transactions:
    cust_agg = df_master.groupby('CustomerID').agg(
        TotalItems=('Quantity', 'sum'),
        AvgOrderValue=('Revenue', 'mean'),
        TotalProfit=('Profit', 'sum'),
        FirstOrderDate=('OrderDate', 'min'),
        LastOrderDate=('OrderDate', 'max')
    ).reset_index()
    
    df_customers = pd.merge(df_customers, cust_agg, on='CustomerID', how='left')
    print(f"[+] Extracted {len(df_customers):,} unique customers.")
    
    # Extract clean product catalog
    prod_cols = ['ProductID', 'ProductName', 'Category', 'SubCategory', 'Brand']
    df_products = df_master[prod_cols].drop_duplicates(subset=['ProductID']).copy().reset_index(drop=True)
    prod_agg = df_master.groupby('ProductID').agg(
        TotalSoldQty=('Quantity', 'sum'),
        TotalRevenue=('Revenue', 'sum'),
        TotalProfit=('Profit', 'sum'),
        AvgUnitPrice=('UnitPrice', 'mean'),
        AvgDiscount=('DiscountPct', 'mean')
    ).reset_index()
    df_products = pd.merge(df_products, prod_agg, on='ProductID', how='left')
    print(f"[+] Extracted {len(df_products):,} unique products.")
    
    # Extract clean store directory
    store_cols = ['StoreID', 'StoreName', 'StoreCity', 'StoreState', 'StoreRegion', 'StoreType']
    df_stores = df_master[store_cols].drop_duplicates(subset=['StoreID']).copy().reset_index(drop=True)
    store_agg = df_master.groupby('StoreID').agg(
        StoreOrders=('OrderID', 'nunique'),
        StoreRevenue=('Revenue', 'sum'),
        StoreProfit=('Profit', 'sum')
    ).reset_index()
    df_stores = pd.merge(df_stores, store_agg, on='StoreID', how='left')
    print(f"[+] Extracted {len(df_stores):,} unique retail store locations.")
    
    # Clean Inventory
    df_inventory['SnapshotDate'] = pd.to_datetime(df_inventory['SnapshotDate'])
    # Inventory KPIs: Stockout risk flags
    # Days of stock = StockOnHand / AvgDailyDemand90d
    df_inventory['DaysOfStock'] = np.where(
        df_inventory['AvgDailyDemand90d'] > 0, 
        df_inventory['StockOnHand'] / df_inventory['AvgDailyDemand90d'], 
        999.0
    ).round(1)
    
    # Stock Status Classification
    def calc_status(row):
        if row['StockOnHand'] <= 0:
            return 'Stockout'
        elif row['StockOnHand'] <= row['SafetyStock']:
            return 'Critical Stock'
        elif row['StockOnHand'] <= row['ReorderLevel']:
            return 'Reorder Required'
        elif row['StockOnHand'] >= row['MaxStockLevel'] * 0.9:
            return 'Overstock Risk'
        else:
            return 'Optimal'
            
    df_inventory['CalculatedStatus'] = df_inventory.apply(calc_status, axis=1)
    
    # Save optimized Parquet tables
    print("[*] Exporting processed datasets to high-performance Parquet format...")
    df_master.to_parquet("data/processed/master_sales.parquet", index=False)
    df_customers.to_parquet("data/processed/customers.parquet", index=False)
    df_products.to_parquet("data/processed/products.parquet", index=False)
    df_stores.to_parquet("data/processed/stores.parquet", index=False)
    df_inventory.to_parquet("data/processed/inventory.parquet", index=False)
    df_forecast.to_parquet("data/processed/forecast.parquet", index=False)
    
    # Also create SQLite database for fast analytical SQL querying
    print("[*] Creating SQLite database data/retailpulse.db for fast OLAP SQL queries...")
    conn = sqlite3.connect("data/retailpulse.db")
    df_master.to_sql("sales", conn, if_exists="replace", index=False)
    df_customers.to_sql("customers", conn, if_exists="replace", index=False)
    df_products.to_sql("products", conn, if_exists="replace", index=False)
    df_stores.to_sql("stores", conn, if_exists="replace", index=False)
    df_inventory.to_sql("inventory", conn, if_exists="replace", index=False)
    df_forecast.to_sql("forecast", conn, if_exists="replace", index=False)
    
    # Create indexes for sub-millisecond query performance
    with conn:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(OrderDate)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sales_cust ON sales(CustomerID)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sales_prod ON sales(ProductID)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sales_cat ON sales(Category)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cust_segment ON customers(Segment)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_inv_prod_store ON inventory(ProductID, StoreID)")
    conn.close()
    
    print("\n" + "=" * 60)
    print("ETL Pipeline Successfully Completed!")
    print(f"Sales Records:      {len(df_master):,}")
    print(f"Customer Profiles:  {len(df_customers):,}")
    print(f"Product SKUs:       {len(df_products):,}")
    print(f"Store Locations:    {len(df_stores):,}")
    print(f"Inventory Items:    {len(df_inventory):,}")
    print(f"Total Revenue:      INR {df_master['Revenue'].sum():,.2f}")
    print(f"Total Gross Profit: INR {df_master['Profit'].sum():,.2f}")
    print(f"Date Horizon:       {df_master['OrderDate'].min().strftime('%Y-%m-%d')} to {df_master['OrderDate'].max().strftime('%Y-%m-%d')}")
    print("=" * 60)

if __name__ == "__main__":
    run_etl()
