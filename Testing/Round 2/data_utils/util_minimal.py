"""
Minimal version of the util module without dependencies on seaborn.
This module provides essential functions for loading and processing price data for Round 2.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datapath = "../../../Prosperity 3 Data/"

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

def spread_plot(data, product):
    """
    Plot the bid-ask spread for a specific product.

    Parameters:
        data (pd.DataFrame): DataFrame with price data
        product (str): Product name (e.g., 'SQUID_INK')
    """
    # Filter for the specific product
    product_data = data[data['product'] == product].copy()

    if len(product_data) == 0:
        raise ValueError(f"No data found for product {product}")

    # Calculate mid price and spread
    product_data['mid_price'] = (product_data['ask_price_1'] + product_data['bid_price_1']) / 2
    product_data['spread'] = product_data['ask_price_1'] - product_data['bid_price_1']

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot mid price
    # Use index if timestamp causes issues
    try:
        ax1.plot(product_data['timestamp'], product_data['mid_price'])
        ax1.set_xlabel('Timestamp')
    except Exception as e:
        print(f"Using index instead of timestamp due to error: {e}")
        ax1.plot(product_data.index, product_data['mid_price'])
        ax1.set_xlabel('Index')

    ax1.set_title(f'{product} Mid Price')
    ax1.set_ylabel('Price')
    ax1.grid(True)

    # Plot spread
    try:
        ax2.plot(product_data['timestamp'], product_data['spread'])
        ax2.set_xlabel('Timestamp')
    except Exception as e:
        print(f"Using index instead of timestamp due to error: {e}")
        ax2.plot(product_data.index, product_data['spread'])
        ax2.set_xlabel('Index')

    ax2.set_title(f'{product} Bid-Ask Spread')
    ax2.set_ylabel('Spread')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def volume_plot(data, product):
    """
    Plot the volume for a specific product.

    Parameters:
        data (pd.DataFrame): DataFrame with price data
        product (str): Product name (e.g., 'SQUID_INK')
    """
    # Filter for the specific product
    product_data = data[data['product'] == product].copy()

    if len(product_data) == 0:
        raise ValueError(f"No data found for product {product}")

    # Calculate total volume
    volume_cols = ['ask_volume_1', 'ask_volume_2', 'ask_volume_3', 'bid_volume_1', 'bid_volume_2', 'bid_volume_3']
    product_data['total_volume'] = product_data[volume_cols].sum(axis=1)

    # Plot total volume
    fig = plt.figure(figsize=(12, 6))

    # Use index if timestamp causes issues
    try:
        plt.plot(product_data['timestamp'], product_data['total_volume'])
        plt.xlabel('Timestamp')
    except Exception as e:
        print(f"Using index instead of timestamp due to error: {e}")
        plt.plot(product_data.index, product_data['total_volume'])
        plt.xlabel('Index')

    plt.title(f'{product} Total Volume')
    plt.ylabel('Volume')
    plt.grid(True)
    plt.show()

def calculate_returns(data, product):
    """
    Calculate log returns for a specific product.

    Parameters:
        data (pd.DataFrame): DataFrame with price data
        product (str): Product name (e.g., 'SQUID_INK')

    Returns:
        pd.Series: Series with log returns indexed by timestamp
    """
    # Get VWAP
    vwap = get_vwap(data, product)

    # Calculate log returns
    log_returns = np.log(vwap).diff().dropna()

    return log_returns
