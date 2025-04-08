"""
Utility functions for loading and analyzing trade data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to import seaborn, but don't fail if it's not available
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available. Some plotting functions may not work correctly.")

# Path to data directory
datapath = "../../../Prosperity 3 Data/"

def load_trade_data(round, day):
    """
    Load trade data for a specific round and day.

    Parameters:
        round (int): Round number
        day (int): Day number

    Returns:
        pd.DataFrame: DataFrame with trade data
    """
    filename = f"Round {round}/trades_round_{round}_day_{day}.csv"
    filepath = os.path.join(datapath, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Trade data file {filename} not found in {datapath}")

    # Load the data into a pandas DataFrame
    data = pd.read_csv(filepath, sep=';')

    return data

def load_all_trade_data(round):
    """
    Load all trade data for a specific round.

    Parameters:
        round (int): Round number

    Returns:
        pd.DataFrame: DataFrame with all trade data
    """
    all_data = pd.DataFrame()

    for day in range(-2, 1):
        try:
            data = load_trade_data(round, day)
            # Add day offset to timestamp for continuity
            data['timestamp'] += np.power(10, 6) * (day+2)
            all_data = pd.concat([all_data, data])
        except FileNotFoundError:
            print(f"Trade data for Round {round}, Day {day} not found. Skipping.")

    return all_data

def get_product_trades(trades_df, product):
    """
    Filter trades for a specific product.

    Parameters:
        trades_df (pd.DataFrame): DataFrame with trade data
        product (str): Product name (e.g., 'SQUID_INK')

    Returns:
        pd.DataFrame: DataFrame with trades for the specified product
    """
    return trades_df[trades_df['symbol'] == product].copy()

def calculate_trade_metrics(trades_df):
    """
    Calculate various metrics from trade data.

    Parameters:
        trades_df (pd.DataFrame): DataFrame with trade data

    Returns:
        pd.DataFrame: DataFrame with trade metrics
    """
    # Group by timestamp
    grouped = trades_df.groupby('timestamp')

    # Calculate metrics
    metrics = pd.DataFrame({
        'volume': grouped['quantity'].sum(),
        'num_trades': grouped.size(),
        'avg_price': grouped.apply(lambda x: np.average(x['price'], weights=x['quantity'])),
        'min_price': grouped['price'].min(),
        'max_price': grouped['price'].max(),
        'price_range': grouped['price'].max() - grouped['price'].min(),
        'total_value': grouped.apply(lambda x: (x['price'] * x['quantity']).sum())
    })

    return metrics

def calculate_rolling_metrics(metrics_df, window=10):
    """
    Calculate rolling metrics.

    Parameters:
        metrics_df (pd.DataFrame): DataFrame with trade metrics
        window (int): Rolling window size

    Returns:
        pd.DataFrame: DataFrame with rolling metrics
    """
    rolling_metrics = pd.DataFrame({
        'rolling_volume': metrics_df['volume'].rolling(window).mean(),
        'rolling_num_trades': metrics_df['num_trades'].rolling(window).mean(),
        'rolling_avg_price': metrics_df['avg_price'].rolling(window).mean(),
        'rolling_price_volatility': metrics_df['avg_price'].rolling(window).std(),
        'rolling_volume_volatility': metrics_df['volume'].rolling(window).std(),
        'rolling_price_range': metrics_df['price_range'].rolling(window).mean()
    })

    return rolling_metrics

def load_price_data(round, day):
    """
    Load price data for a specific round and day.

    Parameters:
        round (int): Round number
        day (int): Day number

    Returns:
        pd.DataFrame: DataFrame with price data
    """
    filename = f"Round {round}/prices_round_{round}_day_{day}.csv"
    filepath = os.path.join(datapath, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Price data file {filename} not found in {datapath}")

    # Load the data into a pandas DataFrame
    data = pd.read_csv(filepath, sep=';')

    return data

def load_all_price_data(round):
    """
    Load all price data for a specific round.

    Parameters:
        round (int): Round number

    Returns:
        pd.DataFrame: DataFrame with all price data
    """
    all_data = pd.DataFrame()

    # First try to load from backtester package if available
    try:
        import sys
        sys.path.append('../../')
        from backtester import get_price_data

        print(f"Trying to load price data from backtester package...")
        product_data = get_price_data('SQUID_INK', round)
        if len(product_data) > 0:
            print(f"Successfully loaded price data from backtester package with {len(product_data)} rows")
            return product_data
    except Exception as e:
        print(f"Could not load price data from backtester package: {e}")

    # Fall back to loading from CSV files
    for day in range(-2, 1):
        try:
            data = load_price_data(round, day)
            # Add day offset to timestamp for continuity
            data['timestamp'] += np.power(10, 6) * (day+2)
            all_data = pd.concat([all_data, data])
        except FileNotFoundError:
            print(f"Price data for Round {round}, Day {day} not found. Skipping.")

    return all_data

def get_product_prices(prices_df, product):
    """
    Filter prices for a specific product.

    Parameters:
        prices_df (pd.DataFrame): DataFrame with price data
        product (str): Product name (e.g., 'SQUID_INK')

    Returns:
        pd.DataFrame: DataFrame with prices for the specified product
    """
    # Check if 'product' column exists
    if 'product' in prices_df.columns:
        return prices_df[prices_df['product'] == product].copy()
    # Check if 'symbol' column exists (used in trade data)
    elif 'symbol' in prices_df.columns:
        return prices_df[prices_df['symbol'] == product].copy()
    else:
        raise ValueError(f"Could not find product or symbol column in price data. Available columns: {prices_df.columns.tolist()}")

def calculate_vwap(prices_df):
    """
    Calculate Volume-Weighted Average Price (VWAP).

    Parameters:
        prices_df (pd.DataFrame): DataFrame with price data

    Returns:
        pd.Series: Series with VWAP
    """
    prices_df['value'] = prices_df['price'] * prices_df['quantity']
    grouped = prices_df.groupby('timestamp')
    vwap = grouped['value'].sum() / grouped['quantity'].sum()

    return vwap

def merge_trades_and_prices(trades_metrics, prices_df, product):
    """
    Merge trade metrics with price data.

    Parameters:
        trades_metrics (pd.DataFrame): DataFrame with trade metrics
        prices_df (pd.DataFrame): DataFrame with price data
        product (str): Product name (e.g., 'SQUID_INK')

    Returns:
        pd.DataFrame: Merged DataFrame
    """
    # Get product prices
    product_prices = get_product_prices(prices_df, product)

    # Check if we have the expected columns
    if 'price' not in product_prices.columns and 'mid_price' not in product_prices.columns:
        # Try to use VWAP if available
        if 'vwap' in product_prices.columns:
            product_prices['price'] = product_prices['vwap']
        # If we have ask and bid, calculate mid price
        elif 'ask' in product_prices.columns and 'bid' in product_prices.columns:
            product_prices['price'] = (product_prices['ask'] + product_prices['bid']) / 2
        else:
            raise ValueError(f"Could not find price information in price data. Available columns: {product_prices.columns.tolist()}")

    # Calculate VWAP
    product_prices_grouped = product_prices.groupby('timestamp')

    # Determine which price column to use
    price_col = 'price' if 'price' in product_prices.columns else 'mid_price'

    # Create price metrics
    price_metrics = pd.DataFrame({
        'mid_price': product_prices_grouped.apply(lambda x: (x[price_col].max() + x[price_col].min()) / 2),
        'ask_price': product_prices_grouped[price_col].max(),
        'bid_price': product_prices_grouped[price_col].min(),
        'spread': product_prices_grouped[price_col].max() - product_prices_grouped[price_col].min(),
        'book_volume': product_prices_grouped['quantity'].sum() if 'quantity' in product_prices.columns else 0
    })

    # Merge on timestamp
    merged = pd.merge(trades_metrics, price_metrics, left_index=True, right_index=True, how='outer')

    # Calculate price difference between trades and order book
    if 'avg_price' in merged.columns and 'mid_price' in merged.columns:
        merged['trade_vs_mid'] = merged['avg_price'] - merged['mid_price']
    if 'avg_price' in merged.columns and 'ask_price' in merged.columns:
        merged['trade_vs_ask'] = merged['avg_price'] - merged['ask_price']
    if 'avg_price' in merged.columns and 'bid_price' in merged.columns:
        merged['trade_vs_bid'] = merged['avg_price'] - merged['bid_price']

    return merged

def calculate_future_price_changes(merged_df, horizons=[1, 5, 10, 20]):
    """
    Calculate future price changes at different horizons.

    Parameters:
        merged_df (pd.DataFrame): Merged DataFrame with trade and price data
        horizons (list): List of horizons for future price changes

    Returns:
        pd.DataFrame: DataFrame with future price changes
    """
    result = merged_df.copy()

    # Check if we have the required price column
    if 'mid_price' not in result.columns:
        # Try to use avg_price if available
        if 'avg_price' in result.columns:
            result['mid_price'] = result['avg_price']
        else:
            raise ValueError(f"Could not find price information in merged data. Available columns: {result.columns.tolist()}")

    for horizon in horizons:
        # Future mid price
        result[f'future_mid_{horizon}'] = result['mid_price'].shift(-horizon)
        # Price change
        result[f'price_change_{horizon}'] = result[f'future_mid_{horizon}'] - result['mid_price']
        # Percentage price change
        result[f'pct_change_{horizon}'] = result[f'price_change_{horizon}'] / result['mid_price']
        # Direction (1 for up, -1 for down, 0 for no change)
        result[f'direction_{horizon}'] = np.sign(result[f'price_change_{horizon}'])

    return result

def plot_trade_metrics(metrics_df, title='Trade Metrics'):
    """
    Plot trade metrics.

    Parameters:
        metrics_df (pd.DataFrame): DataFrame with trade metrics
        title (str): Plot title
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Plot volume
    axes[0].plot(metrics_df.index, metrics_df['volume'], label='Volume')
    axes[0].set_title('Trade Volume')
    axes[0].set_ylabel('Volume')
    axes[0].grid(True)

    # Plot number of trades
    axes[1].plot(metrics_df.index, metrics_df['num_trades'], label='Number of Trades', color='orange')
    axes[1].set_title('Number of Trades')
    axes[1].set_ylabel('Count')
    axes[1].grid(True)

    # Plot average price
    axes[2].plot(metrics_df.index, metrics_df['avg_price'], label='Average Price', color='green')
    axes[2].fill_between(metrics_df.index, metrics_df['min_price'], metrics_df['max_price'], alpha=0.2, color='green')
    axes[2].set_title('Trade Price')
    axes[2].set_ylabel('Price')
    axes[2].grid(True)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    return fig

def plot_correlation_matrix(df, title='Correlation Matrix'):
    """
    Plot correlation matrix.

    Parameters:
        df (pd.DataFrame): DataFrame with variables to correlate
        title (str): Plot title
    """
    corr = df.corr()

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))

    if HAS_SEABORN:
        # Use seaborn for a nicer heatmap if available
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    square=True, linewidths=.5, annot=True, fmt='.2f')
    else:
        # Fall back to matplotlib if seaborn is not available
        plt.imshow(np.ma.array(corr, mask=mask), cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')

        # Add correlation values as text
        for i in range(len(corr)):
            for j in range(len(corr)):
                if i < j:  # Only show the lower triangle
                    continue
                plt.text(j, i, f'{corr.iloc[i, j]:.2f}',
                         ha='center', va='center',
                         color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black')

        # Add labels
        plt.xticks(range(len(corr)), corr.columns, rotation=45, ha='right')
        plt.yticks(range(len(corr)), corr.columns)

    plt.title(title, fontsize=16)
    plt.tight_layout()

    return plt.gcf()
