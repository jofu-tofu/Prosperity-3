"""
Utility functions for Squid Ink analysis in Round 2.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directories to path for imports
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('../../..'))

def load_squid_data(round_num=2):
    """
    Load Squid Ink price data for the specified round.
    
    Parameters:
        round_num (int): Round number (default: 2)
        
    Returns:
        pd.DataFrame: DataFrame containing Squid Ink price data
    """
    try:
        # Try to import the backtester package
        from backtester import get_price_data
        print("Using backtester.get_price_data")
        
        # Load data using backtester
        prices = get_price_data('SQUID_INK', round_num)
        print(f"Successfully loaded price data with {len(prices)} rows")
        
        return prices
        
    except ImportError:
        print("Could not import backtester. Trying alternative method...")
        
        # Try to import from util_minimal in data_utils
        try:
            sys.path.append(os.path.abspath('../data_utils'))
            import util_minimal
            print("Using util_minimal")
            
            # Load data using util_minimal
            all_prices = util_minimal.load_all_price_data(round_num)
            prices = all_prices[all_prices['product'] == 'SQUID_INK']
            print(f"Successfully loaded price data with {len(prices)} rows")
            
            return prices
            
        except ImportError:
            print("Could not import util_minimal. Trying direct CSV loading...")
            
            # Define a function to load price data from CSV files
            data_path = '../../../Prosperity 3 Data'
            
            # List all CSV files for the round
            import glob
            file_pattern = os.path.join(data_path, f'Round {round_num}/prices_round_{round_num}_day_*.csv')
            files = glob.glob(file_pattern)
            
            if not files:
                print(f"No files found matching pattern: {file_pattern}")
                return pd.DataFrame()
            
            # Load and concatenate all files
            dfs = []
            for file in files:
                print(f"Loading {file}...")
                df = pd.read_csv(file, sep=';')
                dfs.append(df)
            
            # Concatenate all dataframes
            all_data = pd.concat(dfs, ignore_index=True)
            
            # Filter for SQUID_INK
            squid_data = all_data[all_data['product'] == 'SQUID_INK']
            print(f"Successfully loaded price data with {len(squid_data)} rows")
            
            return squid_data

def calculate_vwap(prices_df):
    """
    Calculate Volume-Weighted Average Price (VWAP) for the given price data.
    
    Parameters:
        prices_df (pd.DataFrame): DataFrame containing price data
        
    Returns:
        pd.Series: Series containing VWAP values
    """
    # Make a copy of the dataframe to avoid modifying the original
    prices_copy = prices_df.copy()
    
    # Set timestamp as index if available
    if 'timestamp' in prices_copy.columns and prices_copy.index.name != 'timestamp':
        prices_copy.set_index('timestamp', inplace=True)
    
    # Check if VWAP is already in the dataframe
    if 'vwap' in prices_copy.columns:
        print("VWAP is already in the dataframe")
        return prices_copy['vwap']
    
    # Calculate VWAP
    order_vol = prices_copy.filter(['ask_volume_1', 'ask_volume_2', 'ask_volume_3',
                                  'bid_volume_1', 'bid_volume_2', 'bid_volume_3'])
    order_vol = order_vol.fillna(0)  # Replace NaN with 0
    total_vol = order_vol.sum(axis=1)
    
    for i in range(1, 4):
        prices_copy.loc[:, f'ask_dolvol_{i}'] = prices_copy[f'ask_price_{i}'].multiply(order_vol[f'ask_volume_{i}'], fill_value=0)
        prices_copy.loc[:, f'bid_dolvol_{i}'] = prices_copy[f'bid_price_{i}'].multiply(order_vol[f'bid_volume_{i}'], fill_value=0)
    
    dolvol = prices_copy.filter([
        'ask_dolvol_1', 'ask_dolvol_2', 'ask_dolvol_3',
        'bid_dolvol_1', 'bid_dolvol_2', 'bid_dolvol_3'
    ]).sum(axis=1)
    
    vwap = dolvol.divide(total_vol)
    
    # Handle any NaN values (e.g., when total_vol is 0)
    vwap = vwap.fillna(method='ffill').fillna(method='bfill')
    
    return vwap

def calculate_mid_price(prices_df):
    """
    Calculate mid price for the given price data.
    
    Parameters:
        prices_df (pd.DataFrame): DataFrame containing price data
        
    Returns:
        pd.Series: Series containing mid price values
    """
    # Make a copy of the dataframe to avoid modifying the original
    prices_copy = prices_df.copy()
    
    # Set timestamp as index if available
    if 'timestamp' in prices_copy.columns and prices_copy.index.name != 'timestamp':
        prices_copy.set_index('timestamp', inplace=True)
    
    # Check if mid_price is already in the dataframe
    if 'mid_price' in prices_copy.columns:
        print("Mid price is already in the dataframe")
        return prices_copy['mid_price']
    
    # Calculate mid price
    mid_price = (prices_copy['ask_price_1'] + prices_copy['bid_price_1']) / 2
    
    return mid_price

def plot_price_series(price_series, title, ylabel='Price', figsize=(14, 7)):
    """
    Plot a price series.
    
    Parameters:
        price_series (pd.Series): Series containing price data
        title (str): Plot title
        ylabel (str): Y-axis label (default: 'Price')
        figsize (tuple): Figure size (default: (14, 7))
    """
    plt.figure(figsize=figsize)
    plt.plot(price_series.index, price_series.values)
    plt.title(title)
    plt.xlabel('Timestamp')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_price_comparison(series1, series2, label1, label2, title, ylabel='Price', figsize=(14, 7)):
    """
    Plot a comparison of two price series.
    
    Parameters:
        series1 (pd.Series): First series
        series2 (pd.Series): Second series
        label1 (str): Label for first series
        label2 (str): Label for second series
        title (str): Plot title
        ylabel (str): Y-axis label (default: 'Price')
        figsize (tuple): Figure size (default: (14, 7))
    """
    # Ensure both series have the same index
    common_index = series1.index.intersection(series2.index)
    series1_aligned = series1.loc[common_index]
    series2_aligned = series2.loc[common_index]
    
    plt.figure(figsize=figsize)
    plt.plot(series1_aligned.index, series1_aligned.values, label=label1, alpha=0.8)
    plt.plot(series2_aligned.index, series2_aligned.values, label=label2, alpha=0.8)
    plt.title(title)
    plt.xlabel('Timestamp')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return series1_aligned, series2_aligned

def calculate_returns(price_series):
    """
    Calculate returns for the given price series.
    
    Parameters:
        price_series (pd.Series): Series containing price data
        
    Returns:
        tuple: (returns, log_returns)
    """
    # Calculate returns
    returns = price_series.pct_change().dropna()
    
    # Calculate log returns
    log_returns = np.log(price_series).diff().dropna()
    
    return returns, log_returns

def plot_returns_distribution(returns, title, bins=50, figsize=(12, 6)):
    """
    Plot the distribution of returns.
    
    Parameters:
        returns (pd.Series): Series containing returns
        title (str): Plot title
        bins (int): Number of bins (default: 50)
        figsize (tuple): Figure size (default: (12, 6))
    """
    plt.figure(figsize=figsize)
    plt.hist(returns, bins=bins, alpha=0.7)
    plt.title(title)
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def print_statistics(series, name):
    """
    Print basic statistics for a series.
    
    Parameters:
        series (pd.Series): Series to analyze
        name (str): Name of the series
    """
    print(f"{name} statistics:")
    print(f"Number of data points: {len(series)}")
    print(f"Min: {series.min()}")
    print(f"Max: {series.max()}")
    print(f"Mean: {series.mean()}")
    print(f"Median: {series.median()}")
    print(f"Standard deviation: {series.std()}")
    print()
