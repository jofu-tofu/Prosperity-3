import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Add the root directory to the path so we can import the backtester package
sys.path.append(os.path.abspath('../../..'))
from backtester.traders.cmma_rolling_dev_original_trader import Trader
from backtester import get_price_data

def main():
    """
    Test the CMMA Rolling Deviation Original Trader
    """
    print("Testing CMMA Rolling Deviation Original Trader")
    
    # Create trader instance
    trader = Trader()
    print(f"Trader initialized with lookback={trader.lookback}, dev_lookback={trader.dev_lookback}")
    
    # Load price data
    print("Loading price data...")
    prices = get_price_data('SQUID_INK', 1)
    print(f"Loaded {len(prices)} price data points")
    
    # Limit to first 20,000 timestamps (in-sample data)
    in_sample_prices = prices.iloc[:20000]
    print(f"Limited to {len(in_sample_prices)} in-sample data points")
    
    # Get VWAP
    print("Getting VWAP for SQUID_INK...")
    squid_vwap = in_sample_prices['vwap']
    print(f"Got VWAP with {len(squid_vwap)} data points")
    
    # Calculate log prices
    log_prices = np.log(squid_vwap)
    
    # Initialize deques for price history
    max_lookback = max(trader.lookback, trader.dev_lookback)
    price_history = deque(maxlen=max_lookback + 1)
    log_price_history = deque(maxlen=max_lookback + 1)
    
    # Calculate CMMA values
    cmma_values = []
    positions = []
    
    # Process each price point
    current_position = 0
    for i, (_, price) in enumerate(squid_vwap.items()):
        # Update price history
        price_history.append(price)
        log_price_history.append(log_prices[i])
        
        # Calculate CMMA if we have enough data
        if len(price_history) >= max_lookback:
            cmma = trader.calculate_cmma(price_history, log_price_history)
            target_position = trader.calculate_position_size(cmma, price)
            
            # Update current position directly (no position increment constraints)
            current_position = target_position
            
            cmma_values.append(cmma)
            positions.append(current_position)
        else:
            cmma_values.append(0.5)  # Default neutral value
            positions.append(0)
    
    # Convert to Series for easier plotting
    cmma_series = pd.Series(cmma_values, index=squid_vwap.index)
    position_series = pd.Series(positions, index=squid_vwap.index)
    
    # Calculate returns
    price_returns = squid_vwap.pct_change().dropna()
    
    # Calculate strategy returns (without transaction costs for simplicity)
    strategy_returns = position_series.shift(1) * price_returns
    strategy_returns = strategy_returns.dropna()
    
    # Calculate cumulative returns
    cumulative_returns = (1 + strategy_returns).cumprod() - 1
    
    # Calculate performance metrics
    total_return = cumulative_returns.iloc[-1]
    annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
    annualized_volatility = strategy_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
    max_drawdown = (cumulative_returns - cumulative_returns.cummax()).min()
    win_rate = (strategy_returns > 0).mean()
    
    # Calculate position statistics
    position_changes = position_series.diff().dropna()
    num_position_changes = (position_changes != 0).sum()
    avg_position_size = position_series.abs().mean()
    max_position_reached = position_series.abs().max()
    
    # Print performance metrics
    print("\nPerformance Metrics:")
    print(f"Total Return: {total_return:.4f}")
    print(f"Annualized Return: {annualized_return:.4f}")
    print(f"Annualized Volatility: {annualized_volatility:.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Max Drawdown: {max_drawdown:.4f}")
    print(f"Win Rate: {win_rate:.4f}")
    
    print("\nPosition Statistics:")
    print(f"Number of Position Changes: {num_position_changes}")
    print(f"Average Position Size: {avg_position_size:.2f}")
    print(f"Maximum Position Reached: {max_position_reached} (Max allowed: {trader.max_position})")
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    # Plot CMMA values
    plt.subplot(4, 1, 1)
    plt.plot(cmma_series, label='CMMA with Rolling Deviation')
    plt.axhline(y=trader.upper_threshold, color='r', linestyle='--', label=f'Upper Threshold ({trader.upper_threshold})')
    plt.axhline(y=trader.lower_threshold, color='g', linestyle='--', label=f'Lower Threshold ({trader.lower_threshold})')
    plt.title(f'CMMA with Rolling Deviation (Lookback={trader.lookback}, Dev Lookback={trader.dev_lookback})')
    plt.legend()
    plt.grid(True)
    
    # Plot positions
    plt.subplot(4, 1, 2)
    plt.plot(position_series, label='Position', drawstyle='steps-post')
    plt.title('Trading Positions (Original Strategy - No Position Increment Constraints)')
    plt.ylabel(f'Position (Max: {trader.max_position})')
    plt.legend()
    plt.grid(True)
    
    # Plot cumulative returns
    plt.subplot(4, 1, 3)
    plt.plot(cumulative_returns, label='Cumulative Returns')
    plt.title('Strategy Cumulative Returns')
    plt.legend()
    plt.grid(True)
    
    # Plot price
    plt.subplot(4, 1, 4)
    plt.plot(squid_vwap, label='VWAP')
    plt.axhline(y=trader.fair_price, color='r', linestyle='--', label=f'Fair Price ({trader.fair_price})')
    plt.title('Squid_Ink VWAP')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
