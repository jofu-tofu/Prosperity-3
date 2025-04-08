# Squid_Ink Analysis - Round 1

This directory contains analysis and testing of various trading strategies for the Squid_Ink asset in Round 1.

## Directory Structure

- **momentum/**: Contains momentum strategy implementations and tests
  - `squid_momentum_tests.py`: Comprehensive momentum strategy tests
  - `squid_momentum_real_data.py`: Momentum analysis using real data
  - `squid_momentum_simple.py`: Simplified momentum strategy implementation
  - `squid_momentum_optimize.py`: Parameter optimization for momentum strategies
  - `squid_momentum_strategy.py`: Object-oriented implementation of momentum strategies
  - `squid_momentum_backtest.py`: Backtesting framework for momentum strategies
  - `simple_momentum_test.py`: Basic momentum strategy test

- **data_utils/**: Contains utility functions for data loading and processing
  - `util_minimal.py`: Minimal version of util module without external dependencies
  - `test_load_data.py`: Test script for loading Squid_Ink data

- **results/**: Contains output files and visualizations
  - Various PNG files with strategy performance visualizations

- **notebooks/**: Contains Jupyter notebooks for interactive analysis
  - `squid_cmma.ipynb`: Analysis of Cumulative Moving Average Momentum
  - `squid_macd.ipynb`: Analysis of Moving Average Convergence Divergence
  - `squid_pco.ipynb`: Analysis of Price Cycle Oscillator
  - `squid_kelp_eda.ipynb`: Exploratory Data Analysis of Squid_Ink and Kelp
  - `squid_momentum_analysis.ipynb`: Comprehensive momentum strategy analysis

## Momentum Strategies

The following momentum strategies have been implemented and tested:

1. **Simple Momentum**: Measures price change over a lookback period
2. **Rate of Change (ROC)**: Calculates percentage price change over a lookback period
3. **Relative Strength Index (RSI)**: Measures the speed and change of price movements
4. **Moving Average Convergence Divergence (MACD)**: Uses the difference between two exponential moving averages
5. **Cumulative Moving Average Momentum (CMMA)**: Compares current price to the moving average

## Key Findings

- Momentum strategies on Squid_Ink require careful parameter optimization to be profitable
- The optimal Simple Momentum strategy showed a slight positive return
- Recommended parameters:
  - Lookback period: 15
  - Long threshold: 0.0450
  - Short threshold: -0.0560

## Usage

### Running Momentum Tests

To run the momentum tests with real data:

```powershell
cd momentum
python squid_momentum_simple.py
```

To optimize the momentum strategy parameters:

```powershell
cd momentum
python squid_momentum_optimize.py
```

### Accessing Data Utilities

To use the data utilities in your scripts:

```python
import sys
import os
sys.path.append(os.path.abspath('../data_utils'))
import util_minimal

# Load data
prices = util_minimal.load_all_price_data(1)

# Calculate VWAP
squid_vwap = util_minimal.get_vwap(prices, 'SQUID_INK')
```

### Viewing Results

The results directory contains visualizations from the momentum strategy analysis. You can open these PNG files to see the performance of different strategies and the results of parameter optimization.

### Running Notebooks

To run the Jupyter notebooks:

```powershell
cd notebooks
jupyter notebook
```

Or to run a specific notebook:

```powershell
jupyter notebook notebooks/squid_momentum_analysis.ipynb
```
