import numpy as np
import pandas as pd

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
