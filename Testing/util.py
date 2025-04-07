import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

datapath = "../../Prosperity 3 Data/"


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

def spread_plot(raw_data: pd.DataFrame, product: str):
    raw_data = raw_data[raw_data['product'] == product]
    if 'timestamp' in raw_data.columns:
        raw_data.set_index('timestamp', inplace=True)
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

def volume_plot(raw_data: pd.DataFrame, product: str):
    raw_data = raw_data[raw_data['product'] == product]
    if 'timestamp' in raw_data.columns:
        raw_data.set_index('timestamp', inplace=True)
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

def get_vwap(raw_data: pd.DataFrame, product: str, min_vol=0) -> float:
    raw_data = raw_data[raw_data['product'] == product].copy()  # <- add .copy() here
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



def get_adjusted_vwap(raw_data: pd.DataFrame, min_vol = 0) -> float:
    ask_vol = raw_data.filter(['ask_volume_1', 'ask_volume_2', 'ask_volume_3'])
    bid_vol = raw_data.filter(['bid_volume_1', 'bid_volume_2', 'bid_volume_3'])
    ask_vol = ask_vol.map(lambda x: x if x > min_vol else 0)
    bid_vol = bid_vol.map(lambda x: x if x > min_vol else 0)
    tot_ask_vol = ask_vol.sum(1)
    tot_bid_vol = bid_vol.sum(1)
    for i in range(1,4):
        raw_data[f'ask_dolvol_{i}'] = raw_data[f'ask_price_{i}'].multiply(ask_vol[f'ask_volume_{i}'], fill_value= 0)
        raw_data[f'bid_dolvol_{i}'] = raw_data[f'bid_price_{i}'].multiply(bid_vol[f'bid_volume_{i}'], fill_value= 0)
    ask_dolvol = raw_data.filter(['ask_dolvol_1', 'ask_dolvol_2', 'ask_dolvol_3']).sum(1)
    bid_dolvol = raw_data.filter(['bid_dolvol_1', 'bid_dolvol_2', 'bid_dolvol_3']).sum(1)
    return ask_dolvol.divide(tot_ask_vol).add(bid_dolvol.divide(tot_bid_vol)).divide(2)