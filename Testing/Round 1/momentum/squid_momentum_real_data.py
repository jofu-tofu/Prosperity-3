"""
Squid_Ink Momentum Analysis with Real Data

This script analyzes momentum strategies for Squid_Ink using real price data.
It uses the util module to load the actual price data and VWAP for Squid_Ink.
"""

import sys
import os
sys.path.append(os.path.abspath('../'))

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
        print("Error: Neither util nor util_minimal module could be imported")
        raise
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def load_squid_data(n_samples=60000):
    """
    Load real Squid_Ink price data and calculate VWAP.

    Parameters:
        n_samples (int): Maximum number of samples to use

    Returns:
        tuple: (prices DataFrame, squid_vwap Series, log_returns Series)
    """
    print("Loading real Squid_Ink price data...")
    prices = util.load_all_price_data(1)

    if n_samples and len(prices) > n_samples:
        prices = prices.iloc[:n_samples]

    print(f"Loaded {len(prices)} price data points")

    # Calculate VWAP for Squid_Ink
    squid_vwap = util.get_vwap(prices, 'SQUID_INK')
    print(f"Calculated VWAP with {len(squid_vwap)} data points")

    # Calculate log returns
    log_ret = np.log(squid_vwap).diff().dropna()

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
    def sigmoid(x):
        return 1 / (1 + np.exp(-10 * x))  # Multiplier 10 for better sensitivity

    return sigmoid(momentum)

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
    def sigmoid(x):
        return 1 / (1 + np.exp(-0.5 * x))

    return sigmoid(roc)

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
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    return sigmoid(raw_cmma)

def macd(price_series, short_lookback=10, long_lookback=20):
    """
    Compute the Moving Average Convergence Divergence (MACD).

    Parameters:
        price_series (pd.Series): Series of prices
        short_lookback (int): Short-term lookback period
        long_lookback (int): Long-term lookback period

    Returns:
        pd.Series: MACD indicator (0-1 range)
    """
    raw_macd = price_series.ewm(span=short_lookback, adjust=False).mean() - price_series.ewm(span=long_lookback, adjust=False).mean()
    distance = (long_lookback-1)/2
    distance -= (short_lookback-1)/2
    norm = 3*np.sqrt(distance)
    def sigmoid(x):
        return 1/(1 + np.exp(-x))
    norm_macd = sigmoid(1.5*raw_macd/norm)
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
        returns (pd.Series): Series of log returns
        portfolio (pd.Series): Portfolio positions
        tc (float): Transaction cost

    Returns:
        pd.Series: Portfolio returns
    """
    to = abs(portfolio.diff())
    portfolio_returns = portfolio.shift(1) * returns - to * tc
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

    # Calculate win rate
    win_rate = (returns > 0).mean()

    # Calculate profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')

    return {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'Profit Factor': profit_factor
    }

def evaluate_momentum_strategies(squid_vwap, log_ret, lookback=10, long_threshold=0.7, short_threshold=0.3):
    """
    Evaluate different momentum strategies on Squid_Ink data.

    Parameters:
        squid_vwap (pd.Series): VWAP price series for Squid_Ink
        log_ret (pd.Series): Log returns series
        lookback (int): Lookback period for momentum calculations
        long_threshold (float): Threshold for long positions
        short_threshold (float): Threshold for short positions

    Returns:
        tuple: (indicators dict, performance DataFrame)
    """
    print(f"Evaluating momentum strategies with lookback={lookback}...")

    # Calculate momentum indicators
    indicators = {
        'Simple Momentum': simple_momentum(squid_vwap, lookback),
        'ROC': rate_of_change(squid_vwap, lookback),
        'RSI': rsi(squid_vwap, lookback),
        'CMMA': cmma(squid_vwap, lookback),
        'MACD': macd(squid_vwap, short_lookback=lookback, long_lookback=lookback*2)
    }

    # Calculate performance for each indicator
    performance = {}

    for name, indicator in indicators.items():
        print(f"Evaluating {name}...")
        portfolio = get_portfolio(indicator, long_threshold, short_threshold)
        returns = get_returns(log_ret, portfolio)
        metrics = get_performance_metrics(returns)
        performance[name] = metrics

    # Create performance DataFrame
    performance_df = pd.DataFrame(performance).T

    return indicators, performance_df

def plot_indicators(squid_vwap, indicators, save_path=None):
    """
    Plot price and momentum indicators.

    Parameters:
        squid_vwap (pd.Series): VWAP price series
        indicators (dict): Dictionary of indicator series
        save_path (str): Path to save the plot (if None, just displays)
    """
    try:
        plt.figure(figsize=(15, 10))

        plt.subplot(3, 1, 1)
        plt.plot(squid_vwap.index, squid_vwap.values)
        plt.title('Squid_Ink VWAP')
        plt.grid(True)

        plt.subplot(3, 1, 2)
        for name in ['Simple Momentum', 'ROC', 'RSI']:
            if name in indicators:
                plt.plot(indicators[name].index, indicators[name].values, label=name)
        plt.title('Momentum Indicators')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 3)
        for name in ['CMMA', 'MACD']:
            if name in indicators:
                plt.plot(indicators[name].index, indicators[name].values, label=name)
        plt.title('Technical Indicators')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Saved indicators plot to {save_path}")
        else:
            plt.show()

        plt.close()
    except Exception as e:
        print(f"Error creating plot: {e}")

def plot_returns(log_ret, indicators, long_threshold=0.7, short_threshold=0.3, save_path=None):
    """
    Plot cumulative returns for different strategies.

    Parameters:
        log_ret (pd.Series): Log returns series
        indicators (dict): Dictionary of indicator series
        long_threshold (float): Threshold for long positions
        short_threshold (float): Threshold for short positions
        save_path (str): Path to save the plot (if None, just displays)
    """
    try:
        plt.figure(figsize=(15, 7))

        # Plot buy and hold strategy
        plt.plot(log_ret.cumsum(), label='Buy & Hold', linestyle='--')

        # Plot strategy returns
        for name, indicator in indicators.items():
            portfolio = get_portfolio(indicator, long_threshold, short_threshold)
            returns = get_returns(log_ret, portfolio)
            plt.plot(returns.cumsum(), label=name)

        plt.title('Cumulative Returns of Momentum Strategies')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            print(f"Saved returns plot to {save_path}")
        else:
            plt.show()

        plt.close()
    except Exception as e:
        print(f"Error creating plot: {e}")

def optimize_lookback(squid_vwap, log_ret, strategy_func, min_lookback=5, max_lookback=30, step=5):
    """
    Optimize the lookback period for a momentum strategy.

    Parameters:
        squid_vwap (pd.Series): VWAP price series
        log_ret (pd.Series): Log returns series
        strategy_func (function): Momentum strategy function
        min_lookback (int): Minimum lookback period
        max_lookback (int): Maximum lookback period
        step (int): Step size for lookback periods

    Returns:
        tuple: (optimal_lookback, performance_df)
    """
    lookback_periods = range(min_lookback, max_lookback + 1, step)
    results = []

    for lookback in lookback_periods:
        print(f"Testing lookback period: {lookback}")

        # Calculate indicator
        indicator = strategy_func(squid_vwap, lookback)

        # Get portfolio and returns
        portfolio = get_portfolio(indicator, 0.7, 0.3)
        returns = get_returns(log_ret, portfolio)

        # Calculate metrics
        metrics = get_performance_metrics(returns)

        results.append({
            'Lookback': lookback,
            **metrics
        })

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Find optimal lookback
    if len(results_df) > 0:
        optimal_idx = results_df['Sharpe Ratio'].idxmax()
        optimal_lookback = results_df.loc[optimal_idx, 'Lookback']
    else:
        optimal_lookback = 10

    return optimal_lookback, results_df

def optimize_thresholds(squid_vwap, log_ret, indicator, threshold_range=None):
    """
    Optimize thresholds for a momentum indicator.

    Parameters:
        squid_vwap (pd.Series): VWAP price series
        log_ret (pd.Series): Log returns series
        indicator (pd.Series): Momentum indicator series
        threshold_range (list): List of thresholds to test

    Returns:
        tuple: (optimal_long_threshold, optimal_short_threshold, performance_df)
    """
    if threshold_range is None:
        threshold_range = np.linspace(0.1, 0.9, 9)

    results = []

    for long_threshold in threshold_range:
        for short_threshold in threshold_range:
            if long_threshold <= short_threshold:
                continue

            # Get portfolio and returns
            portfolio = get_portfolio(indicator, long_threshold, short_threshold)
            returns = get_returns(log_ret, portfolio)

            # Calculate metrics
            metrics = get_performance_metrics(returns)

            results.append({
                'Long Threshold': long_threshold,
                'Short Threshold': short_threshold,
                **metrics
            })

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Find optimal thresholds
    if len(results_df) > 0:
        optimal_idx = results_df['Sharpe Ratio'].idxmax()
        optimal_long = results_df.loc[optimal_idx, 'Long Threshold']
        optimal_short = results_df.loc[optimal_idx, 'Short Threshold']
    else:
        optimal_long = 0.7
        optimal_short = 0.3

    return optimal_long, optimal_short, results_df

def run_momentum_analysis():
    """Run a comprehensive momentum analysis on Squid_Ink data."""
    # Load data
    prices, squid_vwap, log_ret = load_squid_data()

    # Evaluate momentum strategies
    indicators, performance = evaluate_momentum_strategies(squid_vwap, log_ret)

    # Plot indicators
    plot_indicators(squid_vwap, indicators, 'squid_momentum_indicators_real.png')

    # Plot returns
    plot_returns(log_ret, indicators, save_path='squid_momentum_returns_real.png')

    # Print performance metrics
    print("\nPerformance Metrics:")
    print(performance[['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Profit Factor']].round(4))

    # Find the best strategy
    best_strategy = performance['Sharpe Ratio'].idxmax()
    best_sharpe = performance.loc[best_strategy, 'Sharpe Ratio']

    print(f"\nBest strategy: {best_strategy} (Sharpe Ratio: {best_sharpe:.4f})")

    # Optimize lookback for the best strategy
    print(f"\nOptimizing lookback period for {best_strategy}...")

    strategy_funcs = {
        'Simple Momentum': simple_momentum,
        'ROC': rate_of_change,
        'RSI': rsi,
        'CMMA': cmma,
        'MACD': macd
    }

    optimal_lookback, lookback_results = optimize_lookback(
        squid_vwap,
        log_ret,
        strategy_funcs[best_strategy]
    )

    print(f"Optimal lookback period for {best_strategy}: {optimal_lookback}")

    # Calculate indicator with optimal lookback
    optimal_indicator = strategy_funcs[best_strategy](squid_vwap, optimal_lookback)

    # Optimize thresholds
    print(f"\nOptimizing thresholds for {best_strategy} with lookback={optimal_lookback}...")

    optimal_long, optimal_short, threshold_results = optimize_thresholds(
        squid_vwap,
        log_ret,
        optimal_indicator
    )

    print(f"Optimal thresholds for {best_strategy}:")
    print(f"- Long threshold: {optimal_long:.2f}")
    print(f"- Short threshold: {optimal_short:.2f}")

    # Calculate final performance with optimal parameters
    optimal_portfolio = get_portfolio(optimal_indicator, optimal_long, optimal_short)
    optimal_returns = get_returns(log_ret, optimal_portfolio)
    optimal_metrics = get_performance_metrics(optimal_returns)

    print(f"\nOptimal {best_strategy} Performance:")
    for metric, value in optimal_metrics.items():
        print(f"- {metric}: {value:.4f}")

    # Plot optimal strategy returns
    try:
        plt.figure(figsize=(15, 7))
        plt.plot(optimal_returns.cumsum(), label=f'Optimal {best_strategy}')
        plt.plot(log_ret.cumsum(), label='Buy & Hold', linestyle='--')
        plt.title(f'Cumulative Returns of Optimal {best_strategy} Strategy')
        plt.legend()
        plt.grid(True)
        plt.savefig('squid_optimal_strategy_returns.png')
        plt.close()
        print(f"Saved optimal strategy returns plot to squid_optimal_strategy_returns.png")
    except Exception as e:
        print(f"Error creating optimal strategy plot: {e}")

    print("\nMomentum analysis completed!")

    return {
        'indicators': indicators,
        'performance': performance,
        'best_strategy': best_strategy,
        'optimal_lookback': optimal_lookback,
        'optimal_thresholds': (optimal_long, optimal_short),
        'optimal_metrics': optimal_metrics
    }

if __name__ == "__main__":
    print("Starting Squid_Ink Momentum Analysis with Real Data...")
    try:
        print("Checking if data directory exists...")
        import os
        data_path = os.path.abspath("../../Prosperity 3 Data/")
        print(f"Data path: {data_path}")
        print(f"Data path exists: {os.path.exists(data_path)}")

        if os.path.exists(data_path):
            round_path = os.path.join(data_path, "Round 1")
            print(f"Round path: {round_path}")
            print(f"Round path exists: {os.path.exists(round_path)}")

            if os.path.exists(round_path):
                files = os.listdir(round_path)
                print(f"Files in Round 1 directory: {files}")

        print("Running momentum analysis...")
        results = run_momentum_analysis()
        print("Analysis completed successfully!")
    except Exception as e:
        import traceback
        print(f"Error running momentum analysis: {e}")
        traceback.print_exc()
