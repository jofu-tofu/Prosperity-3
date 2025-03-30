import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

datapath = "../Data/"

def load_data(round, day):
    """
    Load the data for a specific round and day.
    The data is expected to be in CSV files named 'r{round}_d{day}.csv'.
    """
    filename = f"r{round}_d{day}.csv"
    filepath = os.path.join(datapath, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file {filename} not found in {datapath}")
    
    # Load the data into a pandas DataFrame
    data = pd.read_csv(filepath, sep=';')
    
    return data

def spread_plot(raw_data: pd.DataFrame):
    data = raw_data.filter(['ask_price_1', 'ask_price_2', 'ask_price_3', 
                            'bid_price_1', 'bid_price_2', 'bid_price_3'])

    for col in data.columns:
        data[col] = data[col] - raw_data['mid_price']

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.lineplot(data=data, ax=ax[0])
    ax[0].set_title("Spread Plot of Orderbook Normalized by Mid Price")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Price Deviation from Mid Price")

    sns.histplot(data=data.to_numpy().flatten(), kde=True, ax=ax[1], bins=30)
    ax[1].set_title("Distribution of Orderbook Price Deviations from Mid Price")
    plt.tight_layout()
    plt.show()

def volume_plot(raw_data: pd.DataFrame):
    data = raw_data.filter(['ask_volume_1', 'ask_volume_2', 'ask_volume_3', 
                            'bid_volume_1', 'bid_volume_2', 'bid_volume_3'])
    

    fig, ax = plt.subplots(1, 6, figsize=(12, 5))
    for i, col in enumerate(data.columns):
        # Plot each volume column in a separate subplot
        sns.histplot(data=data[col], ax=ax[i % 6])
        ax[i % 6].set_title(f"{col}")
        ax[i % 6].set_xlabel("Volume")
        ax[i % 6].set_ylabel("Freq")
    
    plt.tight_layout()
    plt.show()

def get_vwap(raw_data: pd.DataFrame, min_vol = 0) -> float:
    order_vol = raw_data.filter(['ask_volume_1', 'ask_volume_2', 'ask_volume_3', 'bid_volume_1', 'bid_volume_2', 'bid_volume_3'])
    order_vol = order_vol.map(lambda x: x if x > min_vol else 0)
    total_vol = order_vol.sum(1)
    for i in range(1,4):
        raw_data[f'ask_dolvol_{i}'] = raw_data[f'ask_price_{i}'].multiply(order_vol[f'ask_volume_{i}'], fill_value= 0)
        raw_data[f'bid_dolvol_{i}'] = raw_data[f'bid_price_{i}'].multiply(order_vol[f'bid_volume_{i}'], fill_value= 0)
    dolvol = raw_data.filter(['ask_dolvol_1', 'ask_dolvol_2', 'ask_dolvol_3', 'bid_dolvol_1', 'bid_dolvol_2', 'bid_dolvol_3']).sum(1)
    return dolvol.divide(total_vol)