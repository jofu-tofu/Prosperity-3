"""
Squid_Ink Momentum Strategy

This script implements and tests various momentum strategies for the Squid_Ink asset.
It can be used to evaluate and compare different momentum indicators.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class SquidMomentumStrategy:
    """
    A class that implements various momentum strategies for Squid_Ink.
    """
    
    def __init__(self, price_data=None, lookback=10):
        """
        Initialize the strategy with price data.
        
        Parameters:
            price_data (pd.Series): Series of prices
            lookback (int): Default lookback period for momentum calculations
        """
        self.lookback = lookback
        
        if price_data is None:
            # Create sample data if none provided
            self.price_data = self._create_sample_data()
        else:
            self.price_data = price_data
        
        # Calculate log returns
        self.log_returns = np.log(self.price_data).diff().dropna()
        
        # Initialize indicators dictionary
        self.indicators = {}
        self.portfolios = {}
        self.returns = {}
        self.performance = {}
    
    def _create_sample_data(self, n_samples=10000):
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
        return pd.Series(prices, index=dates)
    
    def calculate_simple_momentum(self, lookback=None):
        """
        Calculate simple momentum: price change over lookback period.
        
        Parameters:
            lookback (int): Lookback period (uses default if None)
            
        Returns:
            pd.Series: Momentum indicator
        """
        if lookback is None:
            lookback = self.lookback
        
        momentum = self.price_data.pct_change(lookback)
        self.indicators['Simple Momentum'] = momentum
        return momentum
    
    def calculate_roc(self, lookback=None):
        """
        Calculate Rate of Change (ROC) momentum indicator.
        ROC = (Current Price / Price n periods ago) - 1
        
        Parameters:
            lookback (int): Lookback period (uses default if None)
            
        Returns:
            pd.Series: ROC indicator
        """
        if lookback is None:
            lookback = self.lookback
        
        roc = (self.price_data / self.price_data.shift(lookback) - 1) * 100
        self.indicators['ROC'] = roc
        return roc
    
    def calculate_rsi(self, lookback=14):
        """
        Calculate Relative Strength Index (RSI).
        
        Parameters:
            lookback (int): Lookback period
            
        Returns:
            pd.Series: RSI indicator (0-100 range)
        """
        # Calculate price changes
        delta = self.price_data.diff()
        
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
        
        self.indicators['RSI'] = rsi
        return rsi
    
    def calculate_macd(self, short_lookback=12, long_lookback=26, signal_lookback=9):
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        Parameters:
            short_lookback (int): Short-term lookback period
            long_lookback (int): Long-term lookback period
            signal_lookback (int): Signal line lookback period
            
        Returns:
            tuple: (MACD line, Signal line, Histogram)
        """
        # Calculate EMAs
        short_ema = self.price_data.ewm(span=short_lookback, adjust=False).mean()
        long_ema = self.price_data.ewm(span=long_lookback, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = short_ema - long_ema
        
        # Calculate Signal line
        signal_line = macd_line.ewm(span=signal_lookback, adjust=False).mean()
        
        # Calculate Histogram
        histogram = macd_line - signal_line
        
        # Normalize to 0-1 range
        normalized_histogram = (histogram - histogram.min()) / (histogram.max() - histogram.min())
        
        self.indicators['MACD'] = normalized_histogram
        return normalized_histogram
    
    def calculate_cmma(self, lookback=None):
        """
        Compute the Cumulative Moving Average Momentum (CMMA).
        
        Parameters:
            lookback (int): Lookback period (uses default if None)
            
        Returns:
            pd.Series: CMMA indicator (0-1 range)
        """
        if lookback is None:
            lookback = self.lookback
        
        # Calculate raw CMMA
        raw_cmma = (self.price_data - self.price_data.rolling(lookback).mean().shift(1)).divide(np.sqrt(lookback+1)).dropna()
        
        # Normalize using sigmoid function
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        cmma = sigmoid(raw_cmma)
        self.indicators['CMMA'] = cmma
        return cmma
    
    def calculate_all_indicators(self):
        """Calculate all momentum indicators."""
        self.calculate_simple_momentum()
        self.calculate_roc()
        self.calculate_rsi()
        self.calculate_macd()
        self.calculate_cmma()
        return self.indicators
    
    def get_portfolio(self, indicator_name, long_threshold=0.7, short_threshold=0.3):
        """
        Get portfolio based on indicator.
        
        Parameters:
            indicator_name (str): Name of the indicator
            long_threshold (float): Threshold for long positions
            short_threshold (float): Threshold for short positions
            
        Returns:
            pd.Series: Portfolio positions (1 for long, -1 for short, 0 for no position)
        """
        if indicator_name not in self.indicators:
            raise ValueError(f"Indicator {indicator_name} not found. Calculate it first.")
        
        signal = self.indicators[indicator_name]
        
        # Handle RSI differently (scale is 0-100)
        if indicator_name == 'RSI':
            portfolio = pd.Series(0, index=signal.index)
            portfolio[signal > 70] = -1  # Overbought -> Short
            portfolio[signal < 30] = 1   # Oversold -> Long
        else:
            portfolio = pd.Series(0, index=signal.index)
            portfolio[signal > long_threshold] = 1  # Long position
            portfolio[signal < short_threshold] = -1  # Short position
        
        self.portfolios[indicator_name] = portfolio
        return portfolio
    
    def get_returns(self, indicator_name, tc=0.001):
        """
        Calculate portfolio returns for an indicator.
        
        Parameters:
            indicator_name (str): Name of the indicator
            tc (float): Transaction cost
            
        Returns:
            pd.Series: Portfolio returns
        """
        if indicator_name not in self.portfolios:
            self.get_portfolio(indicator_name)
        
        portfolio = self.portfolios[indicator_name]
        
        # Calculate transaction costs
        transactions = abs(portfolio.diff().fillna(0))
        transaction_costs = transactions * tc
        
        # Calculate returns (position * next period return - transaction costs)
        portfolio_returns = portfolio.shift(1) * self.log_returns - transaction_costs
        
        self.returns[indicator_name] = portfolio_returns.dropna()
        return self.returns[indicator_name]
    
    def get_performance_metrics(self, indicator_name):
        """
        Calculate performance metrics for an indicator.
        
        Parameters:
            indicator_name (str): Name of the indicator
            
        Returns:
            dict: Dictionary of performance metrics
        """
        if indicator_name not in self.returns:
            self.get_returns(indicator_name)
        
        returns = self.returns[indicator_name]
        
        # Calculate metrics
        total_return = returns.sum()
        annualized_return = total_return * 252 / len(returns)  # Assuming daily data
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
        max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()
        
        metrics = {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        }
        
        self.performance[indicator_name] = metrics
        return metrics
    
    def evaluate_all_strategies(self, long_threshold=0.7, short_threshold=0.3):
        """
        Evaluate all momentum strategies.
        
        Parameters:
            long_threshold (float): Threshold for long positions
            short_threshold (float): Threshold for short positions
            
        Returns:
            pd.DataFrame: Performance comparison DataFrame
        """
        # Calculate all indicators if not already calculated
        if not self.indicators:
            self.calculate_all_indicators()
        
        # Get portfolios and returns for all indicators
        for indicator_name in self.indicators:
            self.get_portfolio(indicator_name, long_threshold, short_threshold)
            self.get_returns(indicator_name)
            self.get_performance_metrics(indicator_name)
        
        # Create performance comparison DataFrame
        performance_df = pd.DataFrame(self.performance).T
        return performance_df
    
    def plot_indicators(self, save_path=None):
        """
        Plot all indicators.
        
        Parameters:
            save_path (str): Path to save the plot (if None, just displays)
        """
        if not self.indicators:
            self.calculate_all_indicators()
        
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(len(self.indicators) + 1, 1, figsize=(12, 3 * (len(self.indicators) + 1)))
            
            # Plot price data
            axes[0].plot(self.price_data)
            axes[0].set_title('Squid_Ink Price')
            axes[0].grid(True)
            
            # Plot indicators
            for i, (name, indicator) in enumerate(self.indicators.items(), 1):
                axes[i].plot(indicator)
                axes[i].set_title(name)
                axes[i].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                print(f"Saved indicators plot to {save_path}")
            else:
                plt.show()
            
            plt.close()
        except Exception as e:
            print(f"Error creating indicators plot: {e}")
    
    def plot_returns(self, save_path=None):
        """
        Plot cumulative returns for all strategies.
        
        Parameters:
            save_path (str): Path to save the plot (if None, just displays)
        """
        if not self.returns:
            self.evaluate_all_strategies()
        
        try:
            plt.figure(figsize=(12, 6))
            
            for name, returns in self.returns.items():
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
            print(f"Error creating returns plot: {e}")

# Main function to run the strategy
def run_squid_momentum_strategy():
    """Run the Squid Momentum Strategy."""
    # Create the strategy
    strategy = SquidMomentumStrategy()
    
    print(f"Price data range: {strategy.price_data.min():.2f} to {strategy.price_data.max():.2f}")
    
    # Calculate all indicators
    print("\nCalculating momentum indicators...")
    strategy.calculate_all_indicators()
    
    # Plot indicators
    print("\nPlotting indicators...")
    strategy.plot_indicators('squid_momentum_indicators.png')
    
    # Evaluate all strategies
    print("\nEvaluating momentum strategies...")
    performance = strategy.evaluate_all_strategies()
    
    print("\nPerformance Metrics:")
    print(performance.round(4))
    
    # Plot returns
    print("\nPlotting returns...")
    strategy.plot_returns('squid_momentum_returns.png')
    
    # Find the best strategy
    best_strategy = performance['Sharpe Ratio'].idxmax()
    best_sharpe = performance.loc[best_strategy, 'Sharpe Ratio']
    
    print("\nMomentum strategy evaluation completed!")
    print("\nSummary:")
    print(f"- Tested {len(strategy.indicators)} momentum strategies")
    print(f"- Best strategy: {best_strategy} (Sharpe Ratio: {best_sharpe:.4f})")
    
    return {
        'strategy': strategy,
        'performance': performance
    }

if __name__ == "__main__":
    print("Starting Squid Momentum Strategy evaluation...")
    try:
        results = run_squid_momentum_strategy()
        print("Evaluation completed successfully!")
    except Exception as e:
        import traceback
        print(f"Error running Squid Momentum Strategy: {e}")
        traceback.print_exc()
