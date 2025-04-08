"""
Simple Momentum Test for Squid_Ink

This script implements a simple momentum strategy for Squid_Ink and tests its performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Create sample data for testing
def create_sample_data(n_samples=10000):
    """Create sample price data for testing."""
    print("Creating sample data...")
    
    # Create a price series with some momentum characteristics
    np.random.seed(42)  # For reproducibility
    
    # Create timestamps
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1min')
    
    # Create a price series with momentum (trending) and mean-reversion components
    momentum_component = np.cumsum(np.random.normal(0.0001, 0.001, n_samples))
    noise = np.random.normal(0, 0.005, n_samples)
    mean_reversion = -0.1 * np.cumsum(momentum_component) / np.arange(1, n_samples + 1)
    
    # Combine components
    log_prices = 4.6 + momentum_component + mean_reversion + noise
    prices = np.exp(log_prices)
    
    # Create a price Series
    price_series = pd.Series(prices, index=dates)
    
    return price_series

# Momentum Strategies
def simple_momentum(price_series, lookback=10):
    """
    Calculate simple momentum: price change over lookback period.
    
    Parameters:
        price_series (pd.Series): Series of prices
        lookback (int): Lookback period
        
    Returns:
        pd.Series: Momentum indicator (-1 to 1 range)
    """
    return price_series.pct_change(lookback)

def rate_of_change(price_series, lookback=10):
    """
    Calculate Rate of Change (ROC) momentum indicator.
    ROC = (Current Price / Price n periods ago) - 1
    
    Parameters:
        price_series (pd.Series): Series of prices
        lookback (int): Lookback period
        
    Returns:
        pd.Series: ROC indicator
    """
    return (price_series / price_series.shift(lookback) - 1) * 100

def rsi(price_series, lookback=14):
    """
    Calculate Relative Strength Index (RSI).
    
    Parameters:
        price_series (pd.Series): Series of prices
        lookback (int): Lookback period
        
    Returns:
        pd.Series: RSI indicator (0-100 range)
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
    
    return rsi

# Portfolio and Performance Functions
def get_portfolio(signal_series, threshold=0.01):
    """
    Get portfolio based on signal series.
    
    Parameters:
        signal_series (pd.Series): Series of signals
        threshold (float): Threshold for positions
        
    Returns:
        pd.Series: Portfolio positions (1 for long, -1 for short, 0 for no position)
    """
    portfolio = pd.Series(0, index=signal_series.index)
    portfolio[signal_series > threshold] = 1  # Long position
    portfolio[signal_series < -threshold] = -1  # Short position
    return portfolio

def get_returns(price_series, portfolio):
    """
    Calculate portfolio returns.
    
    Parameters:
        price_series (pd.Series): Series of prices
        portfolio (pd.Series): Portfolio positions
        
    Returns:
        pd.Series: Portfolio returns
    """
    # Calculate log returns of the price series
    log_returns = np.log(price_series).diff()
    
    # Calculate portfolio returns (position * next period return)
    portfolio_returns = portfolio.shift(1) * log_returns
    
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
    annualized_return = total_return * 252 * 24 * 60 / len(returns)  # Assuming 1-minute data
    volatility = returns.std() * np.sqrt(252 * 24 * 60)
    sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
    max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()
    
    return {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }

# Main test function
def run_simple_momentum_test():
    """Run a simple momentum test."""
    # Create sample data
    price_series = create_sample_data()
    
    print(f"Sample data created with {len(price_series)} data points")
    print(f"Price range: {price_series.min():.2f} to {price_series.max():.2f}")
    
    # Calculate momentum indicators
    lookback = 10
    print(f"\nCalculating momentum indicators with lookback={lookback}...")
    
    simple_mom = simple_momentum(price_series, lookback)
    roc_indicator = rate_of_change(price_series, lookback)
    rsi_indicator = rsi(price_series, lookback)
    
    # Create portfolios
    print("\nCreating portfolios...")
    
    simple_portfolio = get_portfolio(simple_mom, 0.01)
    roc_portfolio = get_portfolio(roc_indicator, 1.0)  # 1% threshold for ROC
    rsi_portfolio = get_portfolio(rsi_indicator - 50, 10)  # RSI > 60 for long, < 40 for short
    
    # Calculate returns
    print("\nCalculating returns...")
    
    simple_returns = get_returns(price_series, simple_portfolio)
    roc_returns = get_returns(price_series, roc_portfolio)
    rsi_returns = get_returns(price_series, rsi_portfolio)
    
    # Calculate performance metrics
    print("\nCalculating performance metrics...")
    
    simple_metrics = get_performance_metrics(simple_returns)
    roc_metrics = get_performance_metrics(roc_returns)
    rsi_metrics = get_performance_metrics(rsi_returns)
    
    # Create performance comparison DataFrame
    performance = {
        'Simple Momentum': simple_metrics,
        'ROC': roc_metrics,
        'RSI': rsi_metrics
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
        plt.title('Cumulative Returns of Momentum Strategies')
        plt.legend()
        plt.grid(True)
        plt.savefig('momentum_returns.png')
        plt.close()
        print("\nSaved cumulative returns plot to momentum_returns.png")
    except Exception as e:
        print(f"\nError creating plot: {e}")
    
    # Summary
    print("\nMomentum testing completed successfully!")
    print("\nSummary:")
    print(f"- Tested 3 momentum strategies")
    
    best_strategy = performance_df['Sharpe Ratio'].idxmax()
    best_sharpe = performance_df.loc[best_strategy, 'Sharpe Ratio']
    print(f"- Best strategy: {best_strategy} (Sharpe Ratio: {best_sharpe:.4f})")
    
    return {
        'price_series': price_series,
        'performance': performance_df
    }

if __name__ == "__main__":
    print("Starting simple momentum test...")
    try:
        results = run_simple_momentum_test()
        print("Test completed successfully!")
    except Exception as e:
        import traceback
        print(f"Error running momentum test: {e}")
        traceback.print_exc()
