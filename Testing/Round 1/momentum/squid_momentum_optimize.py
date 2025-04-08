"""
Optimized Momentum Strategy for Squid_Ink using real data
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import backtester package
import sys
import os
sys.path.append(os.path.abspath('../../'))
from backtester import get_price_data, get_vwap

print("Starting Squid_Ink Momentum Optimization...")

# Load data
print("Loading price data...")
prices = get_price_data('SQUID_INK', 1)
print(f"Loaded {len(prices)} price data points")

# Get VWAP for Squid_Ink
print("Getting VWAP for SQUID_INK...")
squid_vwap = prices['vwap']
print(f"Got VWAP with {len(squid_vwap)} data points")
print(f"VWAP range: {squid_vwap.min()} to {squid_vwap.max()}")

# Calculate log returns
log_ret = np.log(squid_vwap).diff().dropna()
print(f"Calculated log returns with {len(log_ret)} data points")

# Define momentum strategies
def simple_momentum(price_series, lookback=10):
    """Calculate simple momentum: price change over lookback period."""
    return price_series.pct_change(lookback)

def rate_of_change(price_series, lookback=10):
    """Calculate Rate of Change (ROC)."""
    return (price_series / price_series.shift(lookback) - 1) * 100

def rsi(price_series, lookback=14):
    """Calculate Relative Strength Index (RSI)."""
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

    return rsi

def cmma(price_series, lookback=10):
    """Compute the cumulative moving average."""
    raw_cmma = (price_series - price_series.rolling(lookback).mean().shift(1)).divide(np.sqrt(lookback+1)).dropna()
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    return sigmoid(raw_cmma)

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
    def sigmoid(x):
        return 1/(1 + np.exp(-x))
    norm_macd = sigmoid(1.5*raw_macd/norm)
    return norm_macd

# Portfolio and performance functions
def get_portfolio(signal, long_threshold, short_threshold):
    """Create a portfolio based on a signal."""
    portfolio = pd.Series(0, index=signal.index)
    portfolio[signal > long_threshold] = 1  # Long position
    portfolio[signal < short_threshold] = -1  # Short position
    return portfolio

def get_returns(returns, portfolio, tc=0.0005):
    """Calculate portfolio returns with transaction costs."""
    to = abs(portfolio.diff().fillna(0))
    portfolio_returns = portfolio.shift(1) * returns - to * tc
    return portfolio_returns.dropna()

def get_performance_metrics(returns):
    """Calculate performance metrics."""
    total_return = returns.sum()
    annualized_return = total_return * 252 / len(returns)
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
    max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()
    win_rate = (returns > 0).mean()

    return {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Win Rate': win_rate
    }

# Optimize lookback period
def optimize_lookback(price_series, log_returns, strategy_func, min_lookback=5, max_lookback=50, step=5):
    """Optimize the lookback period for a momentum strategy."""
    print(f"\nOptimizing lookback period from {min_lookback} to {max_lookback}...")

    lookback_periods = range(min_lookback, max_lookback + 1, step)
    results = []

    for lookback in lookback_periods:
        print(f"Testing lookback period: {lookback}")

        # Calculate indicator
        indicator = strategy_func(price_series, lookback)

        # Get portfolio and returns
        portfolio = get_portfolio(indicator, 0.01, -0.01)  # Simple thresholds for testing
        returns = get_returns(log_returns, portfolio)

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
        print(f"Optimal lookback period: {optimal_lookback}")
    else:
        optimal_lookback = 10
        print("Could not determine optimal lookback, using default: 10")

    return optimal_lookback, results_df

# Optimize thresholds
def optimize_thresholds(indicator, log_returns, threshold_range=None):
    """Optimize thresholds for a momentum indicator."""
    if threshold_range is None:
        threshold_range = np.linspace(0.001, 0.1, 10)  # For simple momentum

    print(f"\nOptimizing thresholds...")

    results = []

    for long_threshold in threshold_range:
        for short_threshold in -threshold_range:
            print(f"Testing thresholds: Long={long_threshold:.4f}, Short={short_threshold:.4f}")

            # Get portfolio and returns
            portfolio = get_portfolio(indicator, long_threshold, short_threshold)
            returns = get_returns(log_returns, portfolio)

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
        print(f"Optimal thresholds: Long={optimal_long:.4f}, Short={optimal_short:.4f}")
    else:
        optimal_long = 0.01
        optimal_short = -0.01
        print("Could not determine optimal thresholds, using defaults: 0.01, -0.01")

    return optimal_long, optimal_short, results_df

# Run optimization for Simple Momentum
print("\nOptimizing Simple Momentum strategy...")

# Step 1: Optimize lookback period
optimal_lookback, lookback_results = optimize_lookback(squid_vwap, log_ret, simple_momentum)

# Step 2: Calculate indicator with optimal lookback
optimal_indicator = simple_momentum(squid_vwap, optimal_lookback)

# Step 3: Optimize thresholds
optimal_long, optimal_short, threshold_results = optimize_thresholds(optimal_indicator, log_ret)

# Step 4: Calculate final performance with optimal parameters
optimal_portfolio = get_portfolio(optimal_indicator, optimal_long, optimal_short)
optimal_returns = get_returns(log_ret, optimal_portfolio)
optimal_metrics = get_performance_metrics(optimal_returns)

print("\nOptimal Simple Momentum Strategy Performance:")
for metric, value in optimal_metrics.items():
    print(f"- {metric}: {value:.4f}")

# Plot optimal strategy returns
try:
    plt.figure(figsize=(12, 6))
    plt.plot(optimal_returns.cumsum(), label='Optimal Simple Momentum')
    plt.plot(log_ret.cumsum(), label='Buy & Hold', linestyle='--')
    plt.title('Cumulative Returns of Optimal Simple Momentum Strategy')
    plt.legend()
    plt.grid(True)
    plt.savefig('../results/squid_optimal_momentum_returns.png')
    plt.close()
    print("\nSaved optimal strategy returns plot to squid_optimal_momentum_returns.png")
except Exception as e:
    print(f"\nError creating plot: {e}")

# Plot lookback optimization results
try:
    plt.figure(figsize=(10, 6))
    plt.plot(lookback_results['Lookback'], lookback_results['Sharpe Ratio'], marker='o')
    plt.axvline(x=optimal_lookback, color='r', linestyle='--')
    plt.title('Sharpe Ratio vs. Lookback Period')
    plt.xlabel('Lookback Period')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True)
    plt.savefig('../results/squid_lookback_optimization.png')
    plt.close()
    print("Saved lookback optimization plot to squid_lookback_optimization.png")
except Exception as e:
    print(f"Error creating lookback plot: {e}")

# Plot threshold optimization results
try:
    plt.figure(figsize=(10, 6))
    plt.scatter(threshold_results['Long Threshold'], threshold_results['Short Threshold'],
               c=threshold_results['Sharpe Ratio'], cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.plot(optimal_long, optimal_short, 'ro', markersize=10)
    plt.title('Sharpe Ratio by Threshold Combinations')
    plt.xlabel('Long Threshold')
    plt.ylabel('Short Threshold')
    plt.grid(True)
    plt.savefig('../results/squid_threshold_optimization.png')
    plt.close()
    print("Saved threshold optimization plot to squid_threshold_optimization.png")
except Exception as e:
    print(f"Error creating threshold plot: {e}")

print("\nMomentum optimization completed!")
print("\nSummary:")
print(f"- Optimized Simple Momentum strategy for Squid_Ink")
print(f"- Optimal parameters:")
print(f"  - Lookback period: {optimal_lookback}")
print(f"  - Long threshold: {optimal_long:.4f}")
print(f"  - Short threshold: {optimal_short:.4f}")
print(f"- Performance with optimal parameters:")
print(f"  - Sharpe Ratio: {optimal_metrics['Sharpe Ratio']:.4f}")
print(f"  - Total Return: {optimal_metrics['Total Return']:.4f}")
print(f"  - Win Rate: {optimal_metrics['Win Rate']:.4f}")
