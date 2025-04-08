"""
Test script for the backtester package.
"""

import sys
import os
from backtester import get_price_data, get_vwap, relative_entropy_binned

print("Starting backtester test...")
print("Python version:", sys.version)
print("Current working directory:", os.getcwd())

# Check if data files exist
data_paths = [
    "../Prosperity 3 Data",
    "../../Prosperity 3 Data",
    "../../../Prosperity 3 Data",
    "Prosperity 3 Data"
]

for path in data_paths:
    print(f"Checking if {path} exists: {os.path.exists(path)}")

# Test get_price_data
print("\nTesting get_price_data...")
try:
    prices = get_price_data('SQUID_INK', 1)
    print(f"Successfully got price data with {len(prices)} rows")
    print(f"Columns: {prices.columns.tolist()}")
    print(f"First few rows:\n{prices.head()}")
except Exception as e:
    print(f"Error in get_price_data: {e}")

print("\nTest completed.")
