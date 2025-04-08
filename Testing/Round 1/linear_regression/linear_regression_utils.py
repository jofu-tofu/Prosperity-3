"""
Utility functions for linear regression analysis of price data.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def stochastic_oscillator(prices, k_period=14, d_period=3):
    """
    Calculate the Stochastic Oscillator (%K and %D).
    
    Parameters:
        prices (pd.Series): Series of prices
        k_period (int): Period for %K calculation
        d_period (int): Period for %D calculation (moving average of %K)
        
    Returns:
        tuple: (%K, %D) - both are pandas Series
    """
    # Calculate %K
    low_min = prices.rolling(window=k_period).min()
    high_max = prices.rolling(window=k_period).max()
    
    # Avoid division by zero
    denominator = high_max - low_min
    denominator = denominator.replace(0, np.nan)
    
    k = 100 * ((prices - low_min) / denominator)
    
    # Calculate %D (moving average of %K)
    d = k.rolling(window=d_period).mean()
    
    return k, d

def cmma(price_series, lookback=10):
    """
    Compute the Cumulative Moving Average Momentum (CMMA).
    
    Parameters:
        price_series (pd.Series): Series of prices
        lookback (int): Lookback period
        
    Returns:
        pd.Series: CMMA indicator (0-1 range)
    """
    # Calculate raw CMMA
    raw_cmma = (price_series - price_series.rolling(lookback).mean().shift(1)).divide(np.sqrt(lookback+1)).dropna()
    
    # Normalize using sigmoid function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    return sigmoid(raw_cmma)

def macd(price_series, short_lookback=12, long_lookback=26, signal_lookback=9):
    """
    Compute the Moving Average Convergence Divergence (MACD).
    
    Parameters:
        price_series (pd.Series): Series of prices
        short_lookback (int): Lookback period for short-term EMA
        long_lookback (int): Lookback period for long-term EMA
        signal_lookback (int): Lookback period for signal line
        
    Returns:
        tuple: (MACD line, Signal line, Histogram)
    """
    # Calculate EMAs
    short_ema = price_series.ewm(span=short_lookback, adjust=False).mean()
    long_ema = price_series.ewm(span=long_lookback, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = short_ema - long_ema
    
    # Calculate Signal line
    signal_line = macd_line.ewm(span=signal_lookback, adjust=False).mean()
    
    # Calculate Histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def prepare_features(prices, lookback_periods=[1, 5, 10, 20], target_horizon=1):
    """
    Prepare features for linear regression.
    
    Parameters:
        prices (pd.Series): Series of prices
        lookback_periods (list): List of lookback periods for features
        target_horizon (int): Horizon for target variable (future price change)
        
    Returns:
        tuple: (X, y) - features and target
    """
    # Calculate log prices
    log_prices = np.log(prices)
    
    # Calculate returns
    returns = log_prices.diff().dropna()
    
    # Calculate target (future return)
    target = returns.shift(-target_horizon).dropna()
    
    # Calculate features
    features = pd.DataFrame(index=returns.index)
    
    # Add lagged returns
    for lag in lookback_periods:
        features[f'return_lag_{lag}'] = returns.shift(lag)
    
    # Add CMMA for different lookback periods
    for lookback in lookback_periods:
        features[f'cmma_{lookback}'] = cmma(prices, lookback)
    
    # Add MACD
    macd_line, signal_line, histogram = macd(prices)
    features['macd_line'] = macd_line
    features['macd_signal'] = signal_line
    features['macd_hist'] = histogram
    
    # Add Stochastic Oscillator
    k, d = stochastic_oscillator(prices)
    features['stoch_k'] = k
    features['stoch_d'] = d
    
    # Align features and target
    common_idx = features.index.intersection(target.index)
    X = features.loc[common_idx].dropna()
    y = target.loc[X.index]
    
    return X, y

def train_linear_regression(X, y, test_size=0.2, random_state=42):
    """
    Train a linear regression model.
    
    Parameters:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (model, X_train, X_test, y_train, y_test)
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate a linear regression model.
    
    Parameters:
        model (LinearRegression): Trained model
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        y_train (pd.Series): Training target
        y_test (pd.Series): Testing target
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Calculate feature importances
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.coef_
    }).sort_values('Importance', ascending=False)
    
    return {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'feature_importance': feature_importance
    }

def get_trading_signals(model, X, threshold=0.0):
    """
    Generate trading signals from model predictions.
    
    Parameters:
        model (LinearRegression): Trained model
        X (pd.DataFrame): Features
        threshold (float): Threshold for generating signals
        
    Returns:
        pd.Series: Trading signals (1 for buy, -1 for sell, 0 for hold)
    """
    # Make predictions
    predictions = model.predict(X)
    
    # Generate signals
    signals = pd.Series(0, index=X.index)
    signals[predictions > threshold] = 1  # Buy signal
    signals[predictions < -threshold] = -1  # Sell signal
    
    return signals

def calculate_returns(signals, actual_returns, transaction_cost=0.00075):
    """
    Calculate strategy returns.
    
    Parameters:
        signals (pd.Series): Trading signals
        actual_returns (pd.Series): Actual returns
        transaction_cost (float): Transaction cost
        
    Returns:
        pd.Series: Strategy returns
    """
    # Calculate position changes
    position_changes = signals.diff().fillna(0)
    
    # Calculate transaction costs
    transaction_costs = pd.Series(0, index=signals.index)
    transaction_costs[position_changes != 0] = transaction_cost
    
    # Calculate strategy returns
    strategy_returns = signals.shift(1) * actual_returns - transaction_costs
    
    return strategy_returns.dropna()

def calculate_performance_metrics(returns):
    """
    Calculate performance metrics for a returns series.
    
    Parameters:
        returns (pd.Series): Series of returns
        
    Returns:
        dict: Dictionary of performance metrics
    """
    # Calculate cumulative returns
    cumulative_returns = (1 + returns).cumprod() - 1
    
    # Calculate performance metrics
    total_return = cumulative_returns.iloc[-1]
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
    max_drawdown = (cumulative_returns - cumulative_returns.cummax()).min()
    win_rate = (returns > 0).mean()
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate
    }

def plot_cumulative_returns(strategy_returns, benchmark_returns=None, title='Cumulative Returns'):
    """
    Plot cumulative returns.
    
    Parameters:
        strategy_returns (pd.Series): Strategy returns
        benchmark_returns (pd.Series): Benchmark returns
        title (str): Plot title
    """
    # Calculate cumulative returns
    strategy_cum_returns = (1 + strategy_returns).cumprod() - 1
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(strategy_cum_returns, label='Strategy')
    
    if benchmark_returns is not None:
        benchmark_cum_returns = (1 + benchmark_returns).cumprod() - 1
        plt.plot(benchmark_cum_returns, label='Benchmark')
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_feature_importance(feature_importance, title='Feature Importance'):
    """
    Plot feature importance.
    
    Parameters:
        feature_importance (pd.DataFrame): Feature importance DataFrame
        title (str): Plot title
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.grid(True)
    plt.show()

def plot_correlation_matrix(X, title='Feature Correlation Matrix'):
    """
    Plot correlation matrix.
    
    Parameters:
        X (pd.DataFrame): Features
        title (str): Plot title
    """
    plt.figure(figsize=(12, 10))
    corr = X.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', square=True, linewidths=0.5)
    plt.title(title)
    plt.tight_layout()
    plt.show()
