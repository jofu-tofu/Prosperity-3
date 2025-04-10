import pandas as pd
import numpy as np

def calculate_cmma_log(log_prices, lookback=10, dev_lookback=None):
    """
    Compute the Cumulative Moving Average Momentum (CMMA) using log prices.
    
    Parameters:
        log_prices (pd.Series): Series of log prices
        lookback (int): Lookback period for CMMA calculation
        dev_lookback (int, optional): Lookback period for rolling deviation calculation.
                                     If provided, raw CMMA will be divided by this rolling deviation.
        
    Returns:
        pd.Series: CMMA indicator (0-1 range)
    """
    # Calculate raw CMMA using log prices
    raw_cmma = (log_prices - log_prices.ewm(span=lookback).mean().shift(1)).divide(np.sqrt(lookback+1)).dropna()
    
    # If dev_lookback is provided, divide by rolling deviation
    if dev_lookback is not None and dev_lookback > 0:
        # Calculate rolling standard deviation
        rolling_dev = log_prices.rolling(window=dev_lookback).std().dropna()
        # Align indices and divide raw CMMA by rolling deviation
        # Add a small constant to avoid division by zero
        aligned_dev = rolling_dev.reindex(raw_cmma.index)
        raw_cmma = raw_cmma / (aligned_dev + 1e-8)
    
    # Normalize using sigmoid function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    cmma = sigmoid(raw_cmma)
    return cmma

def cmma_trading_strategy(prices, cmma, fair_price, upper_threshold=0.7, lower_threshold=0.3, max_position=1, position_increment=0.2):
    """
    Implement a CMMA-based mean reversion strategy with position increment constraints.
    Positions can only move in increments of position_increment (default: 0.2) and
    position changes are limited to at most position_increment in either direction per update.
    
    Parameters:
        prices (pd.Series): Series of prices
        cmma (pd.Series): CMMA indicator
        fair_price (float): Fair price to revert to
        upper_threshold (float): Upper threshold for CMMA
        lower_threshold (float): Lower threshold for CMMA
        max_position (float): Maximum allowed position size (default: 1)
        position_increment (float): Position increment size (default: 0.2)
        
    Returns:
        pd.Series: Trading positions (1 for long, -1 for short, 0 for no position)
    """
    # Initialize raw target positions
    raw_positions = pd.Series(0, index=cmma.index)
    
    # Generate raw trading signals based on CMMA thresholds
    raw_positions[cmma > upper_threshold] = -1  # Sell signal when CMMA is high
    raw_positions[cmma < lower_threshold] = 1   # Buy signal when CMMA is low
    
    # Filter signals based on fair price
    price_at_signal = prices.reindex(raw_positions.index)
    raw_positions[(raw_positions == 1) & (price_at_signal > fair_price)] = 0  # Don't buy above fair price
    raw_positions[(raw_positions == -1) & (price_at_signal < fair_price)] = 0  # Don't sell below fair price
    
    # Apply position increment constraints
    # First, scale raw positions by max_position
    scaled_positions = raw_positions * max_position
    
    # Then, quantize positions to increments of position_increment
    # This ensures positions are always at levels like 0, ±0.2, ±0.4, ±0.6, ±0.8, ±1.0 (if max_position=1)
    num_increments = max_position / position_increment
    positions = (scaled_positions / max_position * num_increments).round() * position_increment
    
    # Ensure positions don't exceed max_position
    positions = positions.clip(lower=-max_position, upper=max_position)
    
    # Apply position change constraints (limit changes to at most position_increment)
    # This requires iterating through the positions
    final_positions = pd.Series(0, index=positions.index)
    current_position = 0
    
    for idx in positions.index:
        target_position = positions[idx]
        
        # Calculate position difference
        position_diff = target_position - current_position
        
        # Limit position change to at most position_increment in either direction
        if position_diff > position_increment:
            position_diff = position_increment
        elif position_diff < -position_increment:
            position_diff = -position_increment
        
        # Update current position
        current_position += position_diff
        final_positions[idx] = current_position
    
    return final_positions

def calculate_returns_with_costs(positions, price_returns, cost_per_dollar=0.00075):
    """
    Calculate strategy returns with transaction costs.
    
    Parameters:
        positions (pd.Series): Series of positions
        price_returns (pd.Series): Series of price returns
        cost_per_dollar (float): Transaction cost per dollar traded (default: 0.075%)
        
    Returns:
        pd.Series: Strategy returns with transaction costs
    """
    # Calculate position changes (absolute value)
    position_changes = positions.diff().abs().fillna(0)
    
    # Calculate transaction costs
    transaction_costs = position_changes * cost_per_dollar
    
    # Calculate raw strategy returns
    raw_returns = positions.shift(1) * price_returns
    
    # Subtract transaction costs from raw returns
    net_returns = raw_returns - transaction_costs
    
    return net_returns.dropna()

def calculate_performance_metrics(returns):
    """
    Calculate performance metrics for a strategy.
    
    Parameters:
        returns (pd.Series): Series of strategy returns
        
    Returns:
        dict: Dictionary of performance metrics
    """
    # Calculate cumulative returns
    cumulative_returns = (1 + returns).cumprod() - 1
    
    # Calculate performance metrics
    total_return = cumulative_returns.iloc[-1]
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    annualized_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
    max_drawdown = (cumulative_returns - cumulative_returns.cummax()).min()
    win_rate = (returns > 0).mean()
    
    return {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'Cumulative Returns': cumulative_returns
    }
