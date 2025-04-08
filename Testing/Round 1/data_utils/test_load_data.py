"""
Test script to load Squid_Ink data
"""

import os
import sys
import traceback

print("Starting test script...")

# Check if data directory exists
data_path = os.path.abspath("../../Prosperity 3 Data/")
print(f"Data path: {data_path}")
print(f"Data path exists: {os.path.exists(data_path)}")

if os.path.exists(data_path):
    round_path = os.path.join(data_path, "Round 1")
    print(f"Round path: {round_path}")
    print(f"Round path exists: {os.path.exists(round_path)}")
    
    if os.path.exists(round_path):
        files = os.listdir(round_path)
        print(f"Files in Round 1 directory: {files}")

# Try to import util_minimal
try:
    sys.path.append(os.path.abspath('.'))
    import util_minimal
    print("Successfully imported util_minimal")
    
    # Try to load data
    try:
        print("Trying to load price data...")
        prices = util_minimal.load_all_price_data(1)
        print(f"Successfully loaded price data with {len(prices)} rows")
        
        # Try to calculate VWAP
        try:
            print("Trying to calculate VWAP for SQUID_INK...")
            squid_vwap = util_minimal.get_vwap(prices, 'SQUID_INK')
            print(f"Successfully calculated VWAP with {len(squid_vwap)} data points")
            print(f"VWAP range: {squid_vwap.min()} to {squid_vwap.max()}")
        except Exception as e:
            print(f"Error calculating VWAP: {e}")
            traceback.print_exc()
    except Exception as e:
        print(f"Error loading price data: {e}")
        traceback.print_exc()
except Exception as e:
    print(f"Error importing util_minimal: {e}")
    traceback.print_exc()

print("Test script completed")
