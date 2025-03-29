import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

datapath = "../Data/"

def load_data(round, day):
    """
    Load the data for a specific round and day.
    The data is expected to be in CSV files named 'round{round}_day{day}.csv'.
    """
    filename = f"r{round}_d{day}.csv"
    filepath = os.path.join(datapath, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file {filename} not found in {datapath}")
    
    # Load the data into a pandas DataFrame
    data = pd.read_csv(filepath, sep=';')
    
    return data

def spread_plot(raw_data: pd.DataFrame):
    data = raw_data.filter(['ask_price_1','ask_price_2','ask_price_3', 'bid_price_1', 'bid_price_2', 'bid_price_3',
                            # 'ask_volume_1', 'ask_volume_2', 'ask_volume_3', 'bid_volume_1', 'bid_volume_2', 'bid_volume_3',
                            ])
    for col in ['ask_price_1','ask_price_2','ask_price_3', 'bid_price_1', 'bid_price_2', 'bid_price_3']:
        data[col] = data[col] - raw_data['mid_price']
    
    sns.lineplot(data=data)
    plt.title("Spread Plot of Orderbook Normalized by Mid Price")
    plt.xlabel("Time")
    plt.ylabel("Price Deviation from Mid Price")
    plt.show()

def get_vwap(raw_data: pd.DataFrame) -> float:
    order_vol = raw_data.filter(['ask_volume_1', 'ask_volume_2', 'ask_volume_3', 'bid_volume_1', 'bid_volume_2', 'bid_volume_3']).sum(1)
    for i in range(1,4):
        raw_data[f'ask_dolvol_{i}'] = raw_data[f'ask_price_{i}'].multiply(raw_data[f'ask_volume_{i}'], fill_value= 0)
        raw_data[f'bid_dolvol_{i}'] = raw_data[f'bid_price_{i}'].multiply(raw_data[f'bid_volume_{i}'], fill_value= 0)
    dolvol = raw_data.filter(['ask_dolvol_1', 'ask_dolvol_2', 'ask_dolvol_3', 'bid_dolvol_1', 'bid_dolvol_2', 'bid_dolvol_3']).sum(1)
    return dolvol.divide(order_vol)