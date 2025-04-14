"""
Utility functions for backtesting momentum strategies.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Path to data directory
datapath = "../Prosperity 3 Data/"

# Alternative paths to try if the main path doesn't work
alternative_datapaths = [
    "../../Prosperity 3 Data/",
    "../../../Prosperity 3 Data/",
    "Prosperity 3 Data/"
]

def load_price_data(round, day):
    """
    Load the data for a specific round and day.
    The data is expected to be in CSV files named 'prices_round_{round}_day_{day}.csv'.
    """
    # Try different filename formats
    filename_formats = [
        f"Round {round}/prices_round_{round}_day_{day}.csv",
        f"Round {round}\\prices_round_{round}_day_{day}.csv",
        f"r{round}_d{day}.csv"
    ]

    # Try the main datapath first
    for filename in filename_formats:
        filepath = os.path.join(datapath, filename)
        if os.path.exists(filepath):
            print(f"Found data file at {filepath}")
            data = pd.read_csv(filepath, sep=';')
            return data

    # If not found, try alternative paths
    for alt_path in alternative_datapaths:
        for filename in filename_formats:
            filepath = os.path.join(alt_path, filename)
            if os.path.exists(filepath):
                print(f"Found data file at {filepath}")
                data = pd.read_csv(filepath, sep=';')
                return data

    # If we get here, the file wasn't found
    raise FileNotFoundError(f"Data file for round {round}, day {day} not found in any of the search paths")

def load_all_price_data(round):
    """
    Load all price data for a specific round.
    The data is expected to be in CSV files named 'prices_round_{round}_day_{day}.csv'.
    """
    all_data = pd.DataFrame()
    count = 0 
    for day in range(-2, 5):
        
        try:
            data = load_price_data(round, day)
            count += 1
            data['timestamp'] += np.power(10, 6) * (count-1)
            all_data = pd.concat([all_data, data])
        except FileNotFoundError:
            print(f"Data for Round {round}, Day {day} not found. Skipping.")

    return all_data.set_index('timestamp')

def get_price_data(product, round):
    """
    Load price data for a specific product and round.

    Parameters:
        product (str): Product name (e.g., 'SQUID_INK')
        round (int): Round number

    Returns:
        pd.DataFrame: DataFrame with price data including a 'vwap' column
    """
    print(f"Loading real data for {product} from round {round}...")
    all_data = load_all_price_data(round)

    if len(all_data) == 0:
        raise ValueError(f"No data loaded for round {round}")

    # Filter for the specific product
    product_data = all_data[all_data['product'] == product].copy()

    if len(product_data) == 0:
        raise ValueError(f"No data found for product {product} in round {round}")

    # Calculate VWAP
    vwap = get_vwap(all_data, product)

    # Add VWAP to the product data
    # First, ensure the timestamps match
    if 'timestamp' in product_data.columns:
        product_data.set_index('timestamp', inplace=True)

    # Add VWAP as a column
    product_data['vwap'] = vwap

    print(f"Successfully loaded real data with {len(product_data)} rows")
    return product_data

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
    # Filter for the specific product
    raw_data = raw_data[raw_data['product'] == product].copy()

    if len(raw_data) == 0:
        raise ValueError(f"No data found for product {product}")

    # Set timestamp as index if available
    if 'timestamp' in raw_data.columns:
        raw_data.set_index('timestamp', inplace=True)

    # Calculate VWAP
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

def relative_entropy_binned(data, num_bins=10):
    """
    Calculate the relative entropy of a data series.

    Parameters:
        data (pd.Series): Data series
        num_bins (int): Number of bins for histogram

    Returns:
        float: Relative entropy
    """
    # Compute histogram: counts and bin edges
    counts, bin_edges = np.histogram(data, bins=num_bins)

    # Normalize counts to create a probability distribution (p)
    total = counts.sum()
    if total == 0:
        raise ValueError("The data series is empty or contains no values within the bins.")

    p = counts / total
    entropy = -np.sum(p * np.log(p + 1e-10))  # Adding a small value to avoid log(0)

    return entropy/np.log(num_bins)

def calculate_price_spikes(prices, window=20, absolute=False):
    """
    Calculate price spikes based on rolling standard deviation (similar to Bollinger Bands).
    Price spikes are calculated as the deviation from the rolling mean divided by rolling standard deviation.

    Parameters:
        prices (pd.Series): Series of prices
        window (int): Window size for rolling calculations
        absolute (bool): Whether to return absolute values of spikes

    Returns:
        pd.Series: Price spikes (z-scores)
    """
    # Calculate log returns
    log_returns = np.log(prices).diff().dropna()

    # Calculate rolling standard deviation
    rolling_std = log_returns.rolling(window=window).std()

    # Calculate price spikes (z-scores)
    price_spikes = log_returns / rolling_std

    # Take absolute value if requested
    if absolute:
        price_spikes = price_spikes.abs()

    return price_spikes

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """
    Calculate Bollinger Bands for a price series.

    Parameters:
        prices (pd.Series): Series of prices
        window (int): Window size for rolling mean and standard deviation
        num_std (float): Number of standard deviations for the bands

    Returns:
        tuple: (middle_band, upper_band, lower_band)
    """
    # Calculate rolling mean (middle band)
    middle_band = prices.rolling(window=window).mean()

    # Calculate rolling standard deviation
    rolling_std = prices.rolling(window=window).std()

    # Calculate upper and lower bands
    upper_band = middle_band + (rolling_std * num_std)
    lower_band = middle_band - (rolling_std * num_std)

    return middle_band, upper_band, lower_band
