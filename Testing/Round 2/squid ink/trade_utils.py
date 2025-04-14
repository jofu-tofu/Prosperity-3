"""
Utility functions for loading and analyzing trade history data.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_trade_data(round_num, day_num):
    """
    Load trade data for a specific round and day.

    Parameters:
        round_num (int): Round number
        day_num (int): Day number

    Returns:
        pd.DataFrame: DataFrame with trade data
    """
    # Path to data directory - try multiple possible locations
    possible_data_paths = [
        '../Prosperity 3 Data',
        '../../Prosperity 3 Data',
        '../../../Prosperity 3 Data',
        '../../../../Prosperity 3 Data',
        'Prosperity 3 Data'
    ]
    
    # Find the first valid data path
    data_path = None
    for path in possible_data_paths:
        if os.path.exists(path):
            data_path = path
            print(f"Found data directory at {path}")
            break
    
    if data_path is None:
        print("Could not find data directory")
        return pd.DataFrame()
    
    # Construct file path
    filename = f"Round {round_num}/trades_round_{round_num}_day_{day_num}.csv"
    filepath = os.path.join(data_path, filename)
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return pd.DataFrame()
    
    # Load the data into a pandas DataFrame
    print(f"Loading {filepath}...")
    data = pd.read_csv(filepath, sep=';')
    print(f"Loaded {len(data)} trade records")
    
    return data

def load_all_trade_data(round_num):
    """
    Load all trade data for a specific round.

    Parameters:
        round_num (int): Round number

    Returns:
        pd.DataFrame: DataFrame with all trade data
    """
    all_data = pd.DataFrame()
    count = 0
    
    for day in range(-1, 2):  # Days -1, 0, 1
        try:
            data = load_trade_data(round_num, day)
            if len(data) > 0:
                count += 1
                # Add day offset to timestamp for continuity
                data['timestamp'] += np.power(10, 6) * (count-1)
                all_data = pd.concat([all_data, data])
        except Exception as e:
            print(f"Error loading trade data for Round {round_num}, Day {day}: {e}")
    
    print(f"Total trade records loaded: {len(all_data)}")
    return all_data

def filter_product_trades(trades_df, product):
    """
    Filter trade data for a specific product.

    Parameters:
        trades_df (pd.DataFrame): DataFrame with trade data
        product (str): Product name (e.g., 'SQUID_INK')

    Returns:
        pd.DataFrame: DataFrame with filtered trade data
    """
    product_trades = trades_df[trades_df['symbol'] == product].copy()
    print(f"Total number of {product} trades: {len(product_trades)}")
    print(f"Percentage of all trades: {len(product_trades) / len(trades_df) * 100:.2f}%")
    
    return product_trades

def calculate_trade_volume(trades_df, time_window=None):
    """
    Calculate trade volume over time.

    Parameters:
        trades_df (pd.DataFrame): DataFrame with trade data
        time_window (str): Time window for resampling (e.g., '1min', '5min')

    Returns:
        pd.Series: Series with trade volume over time
    """
    # Set timestamp as index if not already
    if trades_df.index.name != 'timestamp':
        trades_df = trades_df.set_index('timestamp')
    
    # Calculate absolute quantity (volume)
    trades_df['abs_quantity'] = trades_df['quantity'].abs()
    
    # If time_window is provided, resample the data
    if time_window:
        volume = trades_df['abs_quantity'].resample(time_window).sum()
    else:
        volume = trades_df['abs_quantity']
    
    return volume

def calculate_trade_value(trades_df, time_window=None):
    """
    Calculate trade value (price * quantity) over time.

    Parameters:
        trades_df (pd.DataFrame): DataFrame with trade data
        time_window (str): Time window for resampling (e.g., '1min', '5min')

    Returns:
        pd.Series: Series with trade value over time
    """
    # Set timestamp as index if not already
    if trades_df.index.name != 'timestamp':
        trades_df = trades_df.set_index('timestamp')
    
    # Calculate value (price * absolute quantity)
    trades_df['value'] = trades_df['price'] * trades_df['quantity'].abs()
    
    # If time_window is provided, resample the data
    if time_window:
        value = trades_df['value'].resample(time_window).sum()
    else:
        value = trades_df['value']
    
    return value

def calculate_vwap_from_trades(trades_df, time_window=None):
    """
    Calculate Volume-Weighted Average Price (VWAP) from trade data.

    Parameters:
        trades_df (pd.DataFrame): DataFrame with trade data
        time_window (str): Time window for resampling (e.g., '1min', '5min')

    Returns:
        pd.Series: Series with VWAP over time
    """
    # Set timestamp as index if not already
    if trades_df.index.name != 'timestamp':
        trades_df = trades_df.set_index('timestamp')
    
    # Calculate value and volume
    trades_df['value'] = trades_df['price'] * trades_df['quantity'].abs()
    trades_df['volume'] = trades_df['quantity'].abs()
    
    # If time_window is provided, resample the data
    if time_window:
        value_sum = trades_df['value'].resample(time_window).sum()
        volume_sum = trades_df['volume'].resample(time_window).sum()
        vwap = value_sum / volume_sum
    else:
        # Calculate cumulative VWAP
        value_sum = trades_df['value'].cumsum()
        volume_sum = trades_df['volume'].cumsum()
        vwap = value_sum / volume_sum
    
    return vwap

def plot_trade_prices(trades_df, title=None, figsize=(14, 7)):
    """
    Plot trade prices over time.

    Parameters:
        trades_df (pd.DataFrame): DataFrame with trade data
        title (str): Plot title
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    plt.scatter(trades_df['timestamp'], trades_df['price'], alpha=0.5, s=10)
    plt.title(title or 'Trade Prices Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_trade_quantities(trades_df, title=None, figsize=(14, 7)):
    """
    Plot trade quantities over time.

    Parameters:
        trades_df (pd.DataFrame): DataFrame with trade data
        title (str): Plot title
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    plt.scatter(trades_df['timestamp'], trades_df['quantity'], alpha=0.5, s=10)
    plt.title(title or 'Trade Quantities Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Quantity')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_trade_volume(volume_series, title=None, figsize=(14, 7)):
    """
    Plot trade volume over time.

    Parameters:
        volume_series (pd.Series): Series with trade volume over time
        title (str): Plot title
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(volume_series.index, volume_series.values)
    plt.title(title or 'Trade Volume Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Volume')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_trade_value(value_series, title=None, figsize=(14, 7)):
    """
    Plot trade value over time.

    Parameters:
        value_series (pd.Series): Series with trade value over time
        title (str): Plot title
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(value_series.index, value_series.values)
    plt.title(title or 'Trade Value Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_vwap(vwap_series, title=None, figsize=(14, 7)):
    """
    Plot VWAP over time.

    Parameters:
        vwap_series (pd.Series): Series with VWAP over time
        title (str): Plot title
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(vwap_series.index, vwap_series.values)
    plt.title(title or 'VWAP Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('VWAP')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def analyze_trade_direction(trades_df):
    """
    Analyze trade direction (buy/sell) distribution.

    Parameters:
        trades_df (pd.DataFrame): DataFrame with trade data

    Returns:
        pd.Series: Series with trade direction counts
    """
    # Create a new column for trade direction
    trades_df['direction'] = np.where(trades_df['quantity'] > 0, 'buy', 'sell')
    
    # Count trades by direction
    direction_counts = trades_df['direction'].value_counts()
    
    # Print summary
    print("Trade Direction Analysis:")
    print(f"Buy trades: {direction_counts.get('buy', 0)} ({direction_counts.get('buy', 0) / len(trades_df) * 100:.2f}%)")
    print(f"Sell trades: {direction_counts.get('sell', 0)} ({direction_counts.get('sell', 0) / len(trades_df) * 100:.2f}%)")
    
    return direction_counts

def analyze_trade_size_distribution(trades_df, bins=10):
    """
    Analyze trade size distribution.

    Parameters:
        trades_df (pd.DataFrame): DataFrame with trade data
        bins (int): Number of bins for histogram

    Returns:
        pd.Series: Series with trade size distribution
    """
    # Calculate absolute quantity
    abs_quantity = trades_df['quantity'].abs()
    
    # Print summary statistics
    print("Trade Size Analysis:")
    print(abs_quantity.describe())
    
    # Plot histogram
    plt.figure(figsize=(12, 6))
    plt.hist(abs_quantity, bins=bins)
    plt.title('Trade Size Distribution')
    plt.xlabel('Trade Size (Absolute Quantity)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return abs_quantity

def analyze_trade_price_distribution(trades_df, bins=10):
    """
    Analyze trade price distribution.

    Parameters:
        trades_df (pd.DataFrame): DataFrame with trade data
        bins (int): Number of bins for histogram

    Returns:
        pd.Series: Series with trade price distribution
    """
    # Print summary statistics
    print("Trade Price Analysis:")
    print(trades_df['price'].describe())
    
    # Plot histogram
    plt.figure(figsize=(12, 6))
    plt.hist(trades_df['price'], bins=bins)
    plt.title('Trade Price Distribution')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return trades_df['price']
