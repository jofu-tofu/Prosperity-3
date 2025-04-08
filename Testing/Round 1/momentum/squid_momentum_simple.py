"""
Simple Momentum Test for Squid_Ink using real data
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

print("Starting Squid_Ink Momentum Test...")
print("Debug: Python version:", sys.version)

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

# Calculate momentum indicators
lookback = 10
print(f"\nCalculating momentum indicators with lookback={lookback}...")

simple_mom = simple_momentum(squid_vwap, lookback)
roc_indicator = rate_of_change(squid_vwap, lookback)
rsi_indicator = rsi(squid_vwap, lookback)
cmma_indicator = cmma(squid_vwap, lookback)
macd_indicator = macd(squid_vwap, lookback=lookback)

# Create portfolios based on indicators
def get_portfolio(signal, threshold=0.01):
    """Create a portfolio based on a signal."""
    portfolio = pd.Series(0, index=signal.index)
    portfolio[signal > threshold] = 1  # Long position
    portfolio[signal < -threshold] = -1  # Short position
    return portfolio

print("\nCreating portfolios...")
simple_portfolio = get_portfolio(simple_mom, 0.01)
roc_portfolio = get_portfolio(roc_indicator, 1.0)
rsi_portfolio = get_portfolio(rsi_indicator - 50, 10)
cmma_portfolio = get_portfolio(cmma_indicator - 0.5, 0.1)
macd_portfolio = get_portfolio(macd_indicator - 0.5, 0.1)

# Calculate returns
def get_returns(returns, portfolio, tc=0.0005):
    """Calculate portfolio returns with transaction costs."""
    to = abs(portfolio.diff().fillna(0))
    portfolio_returns = portfolio.shift(1) * returns - to * tc
    return portfolio_returns.dropna()

print("\nCalculating returns...")
simple_returns = get_returns(log_ret, simple_portfolio)
roc_returns = get_returns(log_ret, roc_portfolio)
rsi_returns = get_returns(log_ret, rsi_portfolio)
cmma_returns = get_returns(log_ret, cmma_portfolio)
macd_returns = get_returns(log_ret, macd_portfolio)

# Calculate performance metrics
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

print("\nCalculating performance metrics...")
simple_metrics = get_performance_metrics(simple_returns)
roc_metrics = get_performance_metrics(roc_returns)
rsi_metrics = get_performance_metrics(rsi_returns)
cmma_metrics = get_performance_metrics(cmma_returns)
macd_metrics = get_performance_metrics(macd_returns)

# Create performance comparison DataFrame
performance = {
    'Simple Momentum': simple_metrics,
    'ROC': roc_metrics,
    'RSI': rsi_metrics,
    'CMMA': cmma_metrics,
    'MACD': macd_metrics
}

performance_df = pd.DataFrame(performance).T
print("\nPerformance Metrics:")
print(performance_df.round(4))

# Plot cumulative returns
try:
    plt.figure(figsize=(12, 6))
    plt.plot(simple_returns.cumsum(), label='Simple Momentum')
    plt.plot(roc_returns.cumsum(), label='ROC')
    plt.plot(rsi_returns.cumsum(), label='RSI')
    plt.plot(cmma_returns.cumsum(), label='CMMA')
    plt.plot(macd_returns.cumsum(), label='MACD')
    plt.plot(log_ret.cumsum(), label='Buy & Hold', linestyle='--')
    plt.title('Cumulative Returns of Momentum Strategies')
    plt.legend()
    plt.grid(True)
    plt.savefig('../results/squid_momentum_returns_real.png')
    plt.close()
    print("\nSaved cumulative returns plot to squid_momentum_returns_real.png")
except Exception as e:
    print(f"\nError creating plot: {e}")

# Find the best strategy
best_strategy = performance_df['Sharpe Ratio'].idxmax()
best_sharpe = performance_df.loc[best_strategy, 'Sharpe Ratio']
best_return = performance_df.loc[best_strategy, 'Total Return']

print("\nMomentum test completed!")
print("\nSummary:")
print(f"- Tested 4 momentum strategies on real Squid_Ink data")
print(f"- Best strategy: {best_strategy}")
print(f"  - Sharpe Ratio: {best_sharpe:.4f}")
print(f"  - Total Return: {best_return:.4f}")
print(f"  - Win Rate: {performance_df.loc[best_strategy, 'Win Rate']:.4f}")
