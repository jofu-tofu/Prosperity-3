# Data Utilities for Squid_Ink Analysis

This directory contains utility functions and scripts for loading and processing Squid_Ink data.

## Files

- `util_minimal.py`: A minimal version of the util module without dependencies on external libraries like seaborn
- `test_load_data.py`: A test script for loading Squid_Ink data and verifying data access

## Key Functions

### Loading Price Data

```python
def load_price_data(round, day):
    """
    Load the data for a specific round and day.
    The data is expected to be in CSV files named 'r{round}_d{day}.csv'.
    """
    filename = f"Round {round}\prices_round_{round}_day_{day}.csv"
    filepath = os.path.join(datapath, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file {filename} not found in {datapath}")
    
    # Load the data into a pandas DataFrame
    data = pd.read_csv(filepath, sep=';')
    
    return data
```

### Loading All Price Data

```python
def load_all_price_data(round):
    """
    Load all price data for a specific round.
    The data is expected to be in CSV files named 'r{round}_d{day}.csv'.
    """
    all_data = pd.DataFrame()
    
    for day in range(-2, 1):
        try:
            data = load_price_data(round, day)
            data['timestamp'] += np.power(10, 6) * (day+2)
            all_data = pd.concat([all_data, data])
        except FileNotFoundError:
            print(f"Data for Round {round}, Day {day} not found. Skipping.")
    
    return all_data
```

### Calculating VWAP

```python
def get_vwap(raw_data, product, min_vol=0):
    """
    Calculate the Volume-Weighted Average Price (VWAP) for a specific product.
    
    Parameters:
        raw_data (pd.DataFrame): DataFrame containing price and volume data
        product (str): Product name
        min_vol (int): Minimum volume threshold
        
    Returns:
        pd.Series: VWAP series
    """
    raw_data = raw_data[raw_data['product'] == product].copy()
    
    if 'timestamp' in raw_data.columns:
        raw_data.set_index('timestamp', inplace=True)
    
    order_vol = raw_data.filter(['ask_volume_1', 'ask_volume_2', 'ask_volume_3', 
                                 'bid_volume_1', 'bid_volume_2', 'bid_volume_3'])
    order_vol = order_vol.map(lambda x: x if x > min_vol else 0)
    total_vol = order_vol.sum(axis=1)
    
    for i in range(1, 4):
        raw_data.loc[:, f'ask_dolvol_{i}'] = raw_data[f'ask_price_{i}'].multiply(order_vol[f'ask_volume_{i}'], fill_value=0)
        raw_data.loc[:, f'bid_dolvol_{i}'] = raw_data[f'bid_price_{i}'].multiply(order_vol[f'bid_volume_{i}'], fill_value=0)
    
    dolvol = raw_data.filter([
        'ask_dolvol_1', 'ask_dolvol_2', 'ask_dolvol_3',
        'bid_dolvol_1', 'bid_dolvol_2', 'bid_dolvol_3'
    ]).sum(axis=1)
    
    return dolvol.divide(total_vol)
```

## Usage

To test data loading:

```powershell
python test_load_data.py
```

To use the utility functions in your own scripts:

```python
import sys
import os
sys.path.append(os.path.abspath('../data_utils'))
import util_minimal

# Load data
prices = util_minimal.load_all_price_data(1)

# Calculate VWAP
squid_vwap = util_minimal.get_vwap(prices, 'SQUID_INK')
```
