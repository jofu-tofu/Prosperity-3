"""
Momentum Tests for Squid_Ink

This file contains tests for various momentum strategies applied to the Squid_Ink asset.
It implements and compares different momentum indicators and evaluates their performance.
"""

import sys
import os
sys.path.append(os.path.abspath('../../'))

# Import backtester package
from backtester import get_price_data, get_vwap, relative_entropy_binned

# Try to import required packages, use alternatives if not available
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib not available, plotting functionality will be limited")
    plt = None

try:
    import seaborn as sns
except ImportError:
    print("Warning: seaborn not available, using basic plotting instead")
    sns = None

try:
    from scipy.stats import norm
except ImportError:
    print("Warning: scipy.stats.norm not available, using numpy alternatives")
    norm = None

# Try to import util module, fall back to minimal version if not available
try:
    import util
    print("Using standard util module")
except ImportError:
    try:
        sys.path.append(os.path.abspath('../data_utils'))
        import util_minimal as util
        print("Using minimal util module")
    except ImportError:
        print("Warning: Neither util nor util_minimal module could be imported, using custom implementations")

    # Define minimal utility functions needed for testing
    class UtilReplacement:
        @staticmethod
        def load_all_price_data(round):
            # Create sample data
            n_samples = 60000
            index = pd.date_range(start='2023-01-01', periods=n_samples, freq='1min')
            return pd.DataFrame({
                'product': ['SQUID_INK'] * n_samples,
                'timestamp': index,
                'mid_price': np.random.normal(100, 5, n_samples),
                'ask_price_1': np.random.normal(101, 5, n_samples),
                'ask_volume_1': np.random.randint(1, 100, n_samples),
                'ask_price_2': np.random.normal(102, 5, n_samples),
                'ask_volume_2': np.random.randint(1, 100, n_samples),
                'ask_price_3': np.random.normal(103, 5, n_samples),
                'ask_volume_3': np.random.randint(1, 100, n_samples),
                'bid_price_1': np.random.normal(99, 5, n_samples),
                'bid_volume_1': np.random.randint(1, 100, n_samples),
                'bid_price_2': np.random.normal(98, 5, n_samples),
                'bid_volume_2': np.random.randint(1, 100, n_samples),
                'bid_price_3': np.random.normal(97, 5, n_samples),
                'bid_volume_3': np.random.randint(1, 100, n_samples)
            })

        @staticmethod
        def get_vwap(raw_data, product, min_vol=0):
            # Simple implementation that returns a random price series
            filtered_data = raw_data[raw_data['product'] == product]
            index = filtered_data['timestamp'].unique()
            return pd.Series(np.cumsum(np.random.normal(0, 1, len(index))) + 100, index=index)

        @staticmethod
        def relative_entropy_binned(data, num_bins):
            # Simple implementation
            counts, _ = np.histogram(data, bins=num_bins)
            total = counts.sum()
            if total == 0:
                return 0
            p = counts / total
            entropy = -np.sum(p * np.log(p + 1e-10))
            return entropy / np.log(num_bins)

    util = UtilReplacement()

# Load data
def load_data(n_samples=60000):
    """Load price data for Squid_Ink and calculate VWAP and log returns."""
    # Use the backtester package
    print("Loading price data...")
    # Get the price data for Squid_Ink
    prices = get_price_data('SQUID_INK', 1)
    print(f"Loaded {len(prices)} price data points")

    # Get VWAP
    print("Getting VWAP for SQUID_INK...")
    squid_vwap = prices['vwap']
    print(f"Got VWAP with {len(squid_vwap)} data points")
    print(f"VWAP range: {squid_vwap.min()} to {squid_vwap.max()}")

    # Calculate log returns
    log_ret = np.log(squid_vwap).diff().dropna()
    print(f"Calculated log returns with {len(log_ret)} data points")

    return prices, squid_vwap, log_ret

# Momentum Strategies

def simple_momentum(price_series, lookback=10):
    """
    Calculate simple momentum: price change over lookback period.
    Normalized with sigmoid function.

    Parameters:
        price_series (pd.Series): Series of prices
        lookback (int): Lookback period

    Returns:
        pd.Series: Momentum indicator (0-1 range)
    """
    momentum = price_series.pct_change(lookback)
    # Normalize to 0-1 range using sigmoid
    def safe_sigmoid(x):
        # Clip values to avoid overflow
        clipped = np.clip(x, -5, 5)  # More conservative clipping for the 10x multiplier
        return 1 / (1 + np.exp(-10 * clipped))  # Multiplier 10 for better sensitivity

    return safe_sigmoid(momentum)

def rate_of_change(price_series, lookback=10):
    """
    Calculate Rate of Change (ROC) momentum indicator.
    ROC = (Current Price / Price n periods ago) - 1

    Parameters:
        price_series (pd.Series): Series of prices
        lookback (int): Lookback period

    Returns:
        pd.Series: ROC indicator normalized to 0-1 range
    """
    roc = (price_series / price_series.shift(lookback) - 1) * 100
    # Normalize to 0-1 range
    def safe_sigmoid(x):
        # Clip values to avoid overflow
        clipped = np.clip(x, -100, 100)  # ROC can have larger values
        return 1 / (1 + np.exp(-0.5 * clipped))

    return safe_sigmoid(roc)

def rsi(price_series, lookback=14):
    """
    Calculate Relative Strength Index (RSI).

    Parameters:
        price_series (pd.Series): Series of prices
        lookback (int): Lookback period

    Returns:
        pd.Series: RSI indicator (0-100 range, normalized to 0-1)
    """
    # Calculate price changes
    delta = price_series.diff()

    # Separate gains and losses
    gains = delta.copy()
    losses = delta.copy()
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = abs(losses)

    # Calculate average gains and losses
    avg_gain = gains.rolling(window=lookback).mean()
    avg_loss = losses.rolling(window=lookback).mean()

    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Normalize to 0-1 range
    return rsi / 100

def cmma(price_series, lookback=10):
    """
    Compute the cumulative moving average of the price series.

    Parameters:
        price_series (pd.Series): Series of prices
        lookback (int): Lookback period

    Returns:
        pd.Series: CMMA indicator (0-1 range)
    """
    raw_cmma = (price_series - price_series.rolling(lookback).mean().shift(1)).divide(np.sqrt(lookback+1)).dropna()

    # Safe sigmoid function to avoid overflow
    def safe_sigmoid(x):
        # Clip values to avoid overflow
        clipped = np.clip(x, -50, 50)
        return 1 / (1 + np.exp(-clipped))

    return safe_sigmoid(raw_cmma)

def macd(price_series, lookback=10, long_factor=2):
    """
    Compute the Moving Average Convergence Divergence (MACD).

    Parameters:
        price_series (pd.Series): Series of prices
        lookback (int): Lookback period for short-term EMA
        long_factor (int): Factor to multiply lookback for long-term EMA

    Returns:
        pd.Series: MACD indicator (0-1 range)
    """
    short_lookback = lookback
    long_lookback = lookback * long_factor
    raw_macd = price_series.ewm(span=short_lookback, adjust=False).mean() - price_series.ewm(span=long_lookback, adjust=False).mean()
    distance = (long_lookback-1)/2
    distance -= (short_lookback-1)/2
    norm = 3*np.sqrt(distance)

    # Safe sigmoid function to avoid overflow
    def safe_sigmoid(x):
        # Clip values to avoid overflow
        clipped = np.clip(x, -50, 50)
        return 1/(1 + np.exp(-clipped))

    # Normalize MACD and apply sigmoid
    norm_macd = safe_sigmoid(1.5*raw_macd/norm)
    return norm_macd

# Portfolio and Performance Functions

def get_portfolio(signal_series, long_threshold, short_threshold):
    """
    Get portfolio based on signal series.

    Parameters:
        signal_series (pd.Series): Series of signals (0-1 range)
        long_threshold (float): Threshold for long positions
        short_threshold (float): Threshold for short positions

    Returns:
        pd.Series: Portfolio positions (1 for long, -1 for short, 0 for no position)
    """
    portfolio = pd.Series(index=signal_series.index, dtype=float)
    portfolio[signal_series > long_threshold] = 1  # Long position
    portfolio[signal_series < short_threshold] = -1  # Short position
    portfolio.fillna(0, inplace=True)  # No position
    return portfolio

def get_returns(returns, portfolio, tc=1.5/2000):
    """
    Calculate portfolio returns with transaction costs.

    Parameters:
        returns (pd.Series): Series of returns (log returns or pct_change)
        portfolio (pd.Series): Portfolio positions
        tc (float): Transaction cost

    Returns:
        pd.Series: Portfolio returns
    """
    # Handle NaN values in portfolio and returns
    portfolio = portfolio.fillna(0)
    returns = returns.fillna(0)

    # Align indices
    common_idx = portfolio.index.intersection(returns.index)
    portfolio = portfolio.loc[common_idx]
    returns = returns.loc[common_idx]

    # Calculate transaction costs
    to = abs(portfolio.diff().fillna(0))

    # Calculate portfolio returns
    portfolio_returns = portfolio.shift(1).fillna(0) * returns - to * tc

    return portfolio_returns.dropna()

def get_performance_metrics(returns):
    """
    Calculate performance metrics for a returns series.

    Parameters:
        returns (pd.Series): Series of returns

    Returns:
        dict: Dictionary of performance metrics
    """
    total_return = returns.sum()
    annualized_return = total_return * 252 / len(returns)
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
    max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()

    return {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }

def get_performance_table(signal, returns, thresholds=None):
    """
    Generate performance table for different thresholds.

    Parameters:
        signal (pd.Series): Series of signals (0-1 range)
        returns (pd.Series): Series of log returns
        thresholds (list): List of thresholds to test

    Returns:
        pd.DataFrame: Performance table
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 98)

    performance_data = []

    for threshold in thresholds:
        threshold_frac = (signal > threshold).mean()

        # Long when signal > threshold
        buy_portfolio = pd.Series(0, index=signal.index)
        buy_portfolio[signal > threshold] = 1
        buy_returns = get_returns(returns, buy_portfolio)
        buy_cum_returns = buy_returns.sum()

        # Short when signal > threshold
        sell_portfolio = pd.Series(0, index=signal.index)
        sell_portfolio[signal > threshold] = -1
        sell_returns = get_returns(returns, sell_portfolio)
        sell_cum_returns = sell_returns.sum()

        # Long when signal < threshold
        buyless_portfolio = pd.Series(0, index=signal.index)
        buyless_portfolio[signal < threshold] = 1
        buyless_returns = get_returns(returns, buyless_portfolio)
        buyless_cum_returns = buyless_returns.sum()

        # Short when signal < threshold
        sellless_portfolio = pd.Series(0, index=signal.index)
        sellless_portfolio[signal < threshold] = -1
        sellless_returns = get_returns(returns, sellless_portfolio)
        sellless_cum_returns = sellless_returns.sum()

        performance_data.append({
            'Threshold': threshold,
            'FracGreater': 1-threshold_frac,
            'LongTotRet': buy_cum_returns,
            'ShortTotRet': sell_cum_returns,
            'FracLess': threshold_frac,
            'Long2TotRet': buyless_cum_returns,
            'Short2TotRet': sellless_cum_returns
        })

    performance_df = pd.DataFrame(performance_data)
    return performance_df

# Monte Carlo Permutation Testing

def permuted_prices(price_series, block_size, num_permutations):
    """
    Permutes a series of prices by shuffling blocks of returns.

    Parameters:
        price_series (pd.Series): Series of prices
        block_size (int): Size of the blocks for permutation
        num_permutations (int): Number of permutations to perform

    Returns:
        pd.DataFrame: DataFrame of permuted price series
    """
    # Calculate returns (instead of log returns to avoid numerical issues)
    returns = price_series.pct_change().dropna()

    # Split returns into blocks
    num_blocks = len(returns) // block_size
    blocks = [returns.iloc[i*block_size:(i+1)*block_size] for i in range(num_blocks)]

    # Prepare a DataFrame to hold permuted price series
    permuted_df = pd.DataFrame(index=price_series.index)

    # For each permutation
    for p in range(num_permutations):
        # Shuffle the blocks randomly
        shuffled_blocks = blocks.copy()
        np.random.shuffle(shuffled_blocks)

        # Concatenate the shuffled blocks to create a new return series
        permuted_returns = pd.concat(shuffled_blocks).reset_index(drop=True)

        # Reconstruct the permuted price series
        # Use the first price from the original series as the starting point
        start_value = price_series.iloc[0]
        permuted_prices = pd.Series(index=price_series.index[:len(permuted_returns) + 1])
        permuted_prices.iloc[0] = start_value

        # Calculate cumulative product of (1 + return) to get price series
        for i in range(len(permuted_returns)):
            permuted_prices.iloc[i+1] = permuted_prices.iloc[i] * (1 + permuted_returns.iloc[i])

        # Add to DataFrame
        permuted_df[f'perm_{p}'] = permuted_prices

    return permuted_df

def momentum_mcpt(prices, strategy_func, block_size, num_permutations, **strategy_params):
    """
    Perform Monte Carlo permutation testing for a momentum strategy.

    Parameters:
        prices (pd.Series): Series of prices
        strategy_func (function): Momentum strategy function
        block_size (int): Size of the blocks for permutation
        num_permutations (int): Number of permutations to perform
        **strategy_params: Parameters for the strategy function

    Returns:
        dict: Dictionary of test results
    """
    # Calculate returns (using pct_change instead of log returns to avoid numerical issues)
    returns = prices.pct_change().dropna()

    # Calculate strategy signal
    base_signal = strategy_func(prices, **strategy_params)

    # Create portfolio
    long_threshold = 0.7  # These thresholds can be optimized
    short_threshold = 0.3
    base_portfolio = get_portfolio(base_signal, long_threshold, short_threshold)

    # Calculate returns
    base_returns = get_returns(returns, base_portfolio)
    base_cum_returns = base_returns.sum()

    # Generate permuted price series
    permuted_price_series = permuted_prices(prices, block_size, num_permutations)

    # Calculate permuted signals, portfolios, and returns
    permuted_cum_returns = []

    for col in permuted_price_series.columns:
        perm_prices = permuted_price_series[col].dropna()

        # Skip if too short
        if len(perm_prices) < 50:
            continue

        # Calculate returns using pct_change
        perm_returns = perm_prices.pct_change().dropna()

        try:
            # Calculate strategy signal
            perm_signal = strategy_func(perm_prices, **strategy_params)

            # Create portfolio
            perm_portfolio = get_portfolio(perm_signal, long_threshold, short_threshold)

            # Calculate returns
            perm_strategy_returns = get_returns(perm_returns, perm_portfolio)
            perm_cum_returns = perm_strategy_returns.sum()

            permuted_cum_returns.append(perm_cum_returns)
        except Exception as e:
            print(f"Error in permutation {col}: {e}")
            continue

    # Calculate p-value (handle case where no permutations were successful)
    if len(permuted_cum_returns) > 0:
        p_value = sum(r > base_cum_returns for r in permuted_cum_returns) / len(permuted_cum_returns)
        permuted_mean = np.mean(permuted_cum_returns)
        permuted_std = np.std(permuted_cum_returns)
    else:
        p_value = np.nan
        permuted_mean = np.nan
        permuted_std = np.nan

    return {
        'Strategy': strategy_func.__name__,
        'Base Returns': base_cum_returns,
        'Permuted Returns Mean': permuted_mean,
        'Permuted Returns Std': permuted_std,
        'p-value': p_value,
        'Significant': p_value < 0.05 if not np.isnan(p_value) else False
    }

# Main test function
def run_momentum_tests():
    """Run tests for different momentum strategies."""
    # Load data
    prices, squid_vwap, log_ret = load_data()

    # Calculate momentum indicators
    lookback = 10
    print(f"Calculating momentum indicators with lookback={lookback}...")

    simple_mom = simple_momentum(squid_vwap, lookback)
    roc_indicator = rate_of_change(squid_vwap, lookback)
    rsi_indicator = rsi(squid_vwap, lookback)
    cmma_indicator = cmma(squid_vwap, lookback)
    macd_indicator = macd(squid_vwap, lookback=lookback)

    # Plot indicators if matplotlib is available
    if plt is not None:
        try:
            plt.figure(figsize=(15, 10))

            plt.subplot(3, 1, 1)
            plt.plot(squid_vwap.index, squid_vwap.values)
            plt.title('Squid_Ink VWAP')
            plt.grid(True)

            plt.subplot(3, 1, 2)
            plt.plot(simple_mom.index, simple_mom.values, label='Simple Momentum')
            plt.plot(roc_indicator.index, roc_indicator.values, label='ROC')
            plt.plot(rsi_indicator.index, rsi_indicator.values, label='RSI')
            plt.title('Momentum Indicators')
            plt.legend()
            plt.grid(True)

            plt.subplot(3, 1, 3)
            plt.plot(cmma_indicator.index, cmma_indicator.values, label='CMMA')
            plt.plot(macd_indicator.index, macd_indicator.values, label='MACD')
            plt.title('Existing Indicators')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig('momentum_indicators.png')
            plt.close()
            print("Saved momentum indicators plot to momentum_indicators.png")
        except Exception as e:
            print(f"Error creating plots: {e}")
    else:
        print("Plotting skipped: matplotlib not available")

    # Calculate performance
    print("Calculating performance metrics...")

    # Define thresholds for testing
    thresholds = [0.3, 0.7]  # Short below 0.3, long above 0.7

    strategies = {
        'Simple Momentum': simple_mom,
        'ROC': roc_indicator,
        'RSI': rsi_indicator,
        'CMMA': cmma_indicator,
        'MACD': macd_indicator
    }

    results = {}

    for name, signal in strategies.items():
        portfolio = get_portfolio(signal, thresholds[1], thresholds[0])
        returns = get_returns(log_ret, portfolio)
        metrics = get_performance_metrics(returns)
        results[name] = metrics

    # Create performance comparison DataFrame
    performance_df = pd.DataFrame(results).T
    print("\nPerformance Metrics:")
    print(performance_df.round(4))

    # Plot cumulative returns if matplotlib is available
    if plt is not None:
        try:
            plt.figure(figsize=(15, 7))

            for name, signal in strategies.items():
                portfolio = get_portfolio(signal, thresholds[1], thresholds[0])
                returns = get_returns(log_ret, portfolio)
                plt.plot(returns.cumsum(), label=name)

            plt.title('Cumulative Returns of Momentum Strategies')
            plt.legend()
            plt.grid(True)
            plt.savefig('momentum_returns.png')
            plt.close()
            print("Saved cumulative returns plot to momentum_returns.png")
        except Exception as e:
            print(f"Error creating cumulative returns plot: {e}")
    else:
        print("Cumulative returns plotting skipped: matplotlib not available")

    # Monte Carlo Permutation Testing
    print("\nPerforming Monte Carlo Permutation Testing...")

    strategy_funcs = {
        'simple_momentum': simple_momentum,
        'rate_of_change': rate_of_change,
        'rsi': rsi,
        'cmma': cmma,
        'macd': macd
    }

    mcpt_results = []

    try:
        for name, func in strategy_funcs.items():
            print(f"Testing {name}...")
            result = momentum_mcpt(
                squid_vwap,
                func,
                block_size=5000,
                num_permutations=10,
                lookback=lookback
            )
            mcpt_results.append(result)

        mcpt_df = pd.DataFrame(mcpt_results)
        print("\nMonte Carlo Permutation Test Results:")
        print(mcpt_df.round(4))
    except Exception as e:
        print(f"Error during Monte Carlo Permutation Testing: {e}")
        mcpt_df = pd.DataFrame()

    print("\nMomentum testing completed successfully!")
    print("\nSummary:")
    print(f"- Tested {len(strategies)} momentum strategies")
    if not performance_df.empty:
        best_strategy = performance_df['Sharpe Ratio'].idxmax()
        best_sharpe = performance_df.loc[best_strategy, 'Sharpe Ratio']
        print(f"- Best strategy: {best_strategy} (Sharpe Ratio: {best_sharpe:.4f})")

    return {
        'indicators': strategies,
        'performance': performance_df,
        'mcpt_results': mcpt_df
    }

if __name__ == "__main__":
    print("Starting momentum tests...")
    try:
        run_momentum_tests()
    except Exception as e:
        import traceback
        print(f"Error running momentum tests: {e}")
        traceback.print_exc()
