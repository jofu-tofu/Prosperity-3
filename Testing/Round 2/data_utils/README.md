# Data Utilities for Round 2 Analysis

This directory contains utility functions and scripts for loading and processing data for Round 2.

## Files

- `util_minimal.py`: A minimal version of the util module without dependencies on external libraries like seaborn
- `test_load_data.py`: A test script for loading Round 2 data and verifying data access

## Key Functions

### Loading Price Data

```python
def load_price_data(round, day):
    """
    Load the data for a specific round and day.
    The data is expected to be in CSV files named 'prices_round_{round}_day_{day}.csv'.
    """
    filename = f"Round {round}/prices_round_{round}_day_{day}.csv"
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
    The data is expected to be in CSV files named 'prices_round_{round}_day_{day}.csv'.
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
def get_vwap(data, product):
    """
    Calculate the volume-weighted average price (VWAP) for a specific product.
    
    Parameters:
        data (pd.DataFrame): DataFrame with price data
        product (str): Product name (e.g., 'SQUID_INK')
        
    Returns:
        pd.Series: Series with VWAP values indexed by timestamp
    """
    # Filter for the specific product
    product_data = data[data['product'] == product].copy()
    
    if len(product_data) == 0:
        raise ValueError(f"No data found for product {product}")
    
    # Calculate mid price
    product_data['mid_price'] = (product_data['ask_price_1'] + product_data['bid_price_1']) / 2
    
    # Calculate volume
    product_data['volume'] = (
        product_data['ask_volume_1'] + product_data['ask_volume_2'] + product_data['ask_volume_3'] +
        product_data['bid_volume_1'] + product_data['bid_volume_2'] + product_data['bid_volume_3']
    )
    
    # Calculate VWAP
    product_data['price_volume'] = product_data['mid_price'] * product_data['volume']
    vwap = product_data.groupby('timestamp')['price_volume'].sum() / product_data.groupby('timestamp')['volume'].sum()
    
    return vwap
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
prices = util_minimal.load_all_price_data(2)

# Calculate VWAP
squid_vwap = util_minimal.get_vwap(prices, 'SQUID_INK')
```
