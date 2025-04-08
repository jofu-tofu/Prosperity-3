# Momentum Strategies for Squid_Ink

This directory contains implementations and tests of various momentum strategies for the Squid_Ink asset.

## Files

- `squid_momentum_tests.py`: Comprehensive momentum strategy tests with multiple indicators
- `squid_momentum_real_data.py`: Momentum analysis using real Squid_Ink data
- `squid_momentum_simple.py`: Simplified momentum strategy implementation for quick testing
- `squid_momentum_optimize.py`: Parameter optimization for momentum strategies
- `squid_momentum_strategy.py`: Object-oriented implementation of momentum strategies
- `squid_momentum_backtest.py`: Backtesting framework for momentum strategies
- `simple_momentum_test.py`: Basic momentum strategy test with synthetic data

## Implemented Strategies

1. **Simple Momentum**: Measures price change over a lookback period
   ```python
   def simple_momentum(price_series, lookback=10):
       return price_series.pct_change(lookback)
   ```

2. **Rate of Change (ROC)**: Calculates percentage price change over a lookback period
   ```python
   def rate_of_change(price_series, lookback=10):
       return (price_series / price_series.shift(lookback) - 1) * 100
   ```

3. **Relative Strength Index (RSI)**: Measures the speed and change of price movements
   ```python
   def rsi(price_series, lookback=14):
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
   ```

4. **Cumulative Moving Average Momentum (CMMA)**: Compares current price to the moving average
   ```python
   def cmma(price_series, lookback=10):
       raw_cmma = (price_series - price_series.rolling(lookback).mean().shift(1)).divide(np.sqrt(lookback+1)).dropna()
       def sigmoid(x):
           return 1 / (1 + np.exp(-x))
       
       return sigmoid(raw_cmma)
   ```

5. **Moving Average Convergence Divergence (MACD)**: Uses the difference between two exponential moving averages
   ```python
   def macd(price_series, short_lookback=10, long_lookback=20):
       raw_macd = price_series.ewm(span=short_lookback, adjust=False).mean() - price_series.ewm(span=long_lookback, adjust=False).mean()
       distance = (long_lookback-1)/2 - (short_lookback-1)/2
       norm = 3*np.sqrt(distance)
       def sigmoid(x):
           return 1/(1 + np.exp(-x))
       norm_macd = sigmoid(1.5*raw_macd/norm)
       return norm_macd
   ```

## Optimal Parameters

Based on our optimization tests, the following parameters work best for Squid_Ink:

- **Strategy**: Simple Momentum
- **Lookback Period**: 15
- **Long Threshold**: 0.0450
- **Short Threshold**: -0.0560

## Usage

To run the simple momentum test:

```powershell
python squid_momentum_simple.py
```

To optimize the momentum strategy parameters:

```powershell
python squid_momentum_optimize.py
```
