"""
Squid_Ink Momentum Backtest

This script backtests various momentum strategies on Squid_Ink data.
It attempts to load real data if available, otherwise uses synthetic data.
"""

import sys
import os
sys.path.append(os.path.abspath('../'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import backtester package
import sys
import os
sys.path.append(os.path.abspath('../../'))
from backtester import get_price_data, get_vwap
print("Using backtester package")
UTIL_AVAILABLE = True

class SquidMomentumBacktest:
    """
    A class for backtesting momentum strategies on Squid_Ink data.
    """

    def __init__(self, lookback=10, use_real_data=True):
        """
        Initialize the backtest with price data.

        Parameters:
            lookback (int): Default lookback period for momentum calculations
            use_real_data (bool): Whether to try loading real data
        """
        self.lookback = lookback

        # Load real data
        print("Loading real Squid_Ink data...")
        prices = get_price_data('SQUID_INK', 1)
        self.squid_vwap = prices['vwap']
        print(f"Loaded {len(self.squid_vwap)} data points")
        print(f"VWAP range: {self.squid_vwap.min()} to {self.squid_vwap.max()}")

        # Calculate log returns
        self.log_returns = np.log(self.squid_vwap).diff().dropna()

        # Initialize containers
        self.indicators = {}
        self.portfolios = {}
        self.returns = {}
        self.performance = {}

    def _create_synthetic_data(self, n_samples=10000):
        """Create synthetic price data for testing."""
        print("Creating synthetic Squid_Ink data...")

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

        momentum = self.squid_vwap.pct_change(lookback)

        # Normalize to 0-1 range using sigmoid
        def sigmoid(x):
            return 1 / (1 + np.exp(-10 * x))

        normalized = sigmoid(momentum)
        self.indicators['Simple Momentum'] = normalized
        return normalized

    def calculate_roc(self, lookback=None):
        """
        Calculate Rate of Change (ROC) momentum indicator.

        Parameters:
            lookback (int): Lookback period (uses default if None)

        Returns:
            pd.Series: ROC indicator (0-1 range)
        """
        if lookback is None:
            lookback = self.lookback

        roc = (self.squid_vwap / self.squid_vwap.shift(lookback) - 1) * 100

        # Normalize to 0-1 range using sigmoid
        def sigmoid(x):
            return 1 / (1 + np.exp(-0.5 * x))

        normalized = sigmoid(roc)
        self.indicators['ROC'] = normalized
        return normalized

    def calculate_rsi(self, lookback=14):
        """
        Calculate Relative Strength Index (RSI).

        Parameters:
            lookback (int): Lookback period

        Returns:
            pd.Series: RSI indicator (0-1 range)
        """
        # Calculate price changes
        delta = self.squid_vwap.diff()

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

        # Normalize to 0-1 range
        normalized = rsi / 100
        self.indicators['RSI'] = normalized
        return normalized

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
        raw_cmma = (self.squid_vwap - self.squid_vwap.rolling(lookback).mean().shift(1)).divide(np.sqrt(lookback+1)).dropna()

        # Normalize using sigmoid function
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        cmma = sigmoid(raw_cmma)
        self.indicators['CMMA'] = cmma
        return cmma

    def calculate_macd(self, short_lookback=12, long_lookback=26):
        """
        Calculate Moving Average Convergence Divergence (MACD).

        Parameters:
            short_lookback (int): Short-term lookback period
            long_lookback (int): Long-term lookback period

        Returns:
            pd.Series: MACD indicator (0-1 range)
        """
        # Calculate EMAs
        short_ema = self.squid_vwap.ewm(span=short_lookback, adjust=False).mean()
        long_ema = self.squid_vwap.ewm(span=long_lookback, adjust=False).mean()

        # Calculate MACD line
        raw_macd = short_ema - long_ema

        # Normalize
        distance = (long_lookback-1)/2 - (short_lookback-1)/2
        norm = 3 * np.sqrt(distance)

        def sigmoid(x):
            return 1 / (1 + np.exp(-1.5 * x / norm))

        normalized = sigmoid(raw_macd)
        self.indicators['MACD'] = normalized
        return normalized

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

        portfolio = pd.Series(0, index=signal.index)
        portfolio[signal > long_threshold] = 1  # Long position
        portfolio[signal < short_threshold] = -1  # Short position

        self.portfolios[indicator_name] = portfolio
        return portfolio

    def get_returns(self, indicator_name, tc=0.00075):
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

        # Calculate win rate
        win_rate = (returns > 0).mean()

        # Calculate average win/loss
        avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
        avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0

        # Calculate profit factor
        profit_factor = abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if returns[returns < 0].sum() != 0 else float('inf')

        metrics = {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Win Rate': win_rate,
            'Avg Win': avg_win,
            'Avg Loss': avg_loss,
            'Profit Factor': profit_factor
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

    def optimize_thresholds(self, indicator_name, threshold_range=None):
        """
        Optimize thresholds for a specific indicator.

        Parameters:
            indicator_name (str): Name of the indicator
            threshold_range (list): List of thresholds to test

        Returns:
            tuple: (optimal_long_threshold, optimal_short_threshold, performance_df)
        """
        if indicator_name not in self.indicators:
            raise ValueError(f"Indicator {indicator_name} not found. Calculate it first.")

        if threshold_range is None:
            threshold_range = np.linspace(0.1, 0.9, 9)

        results = []

        for long_threshold in threshold_range:
            for short_threshold in threshold_range:
                if long_threshold <= short_threshold:
                    continue

                # Get portfolio and returns
                portfolio = self.get_portfolio(indicator_name, long_threshold, short_threshold)
                returns = self.get_returns(indicator_name)

                # Calculate metrics
                total_return = returns.sum()
                annualized_return = total_return * 252 / len(returns)
                volatility = returns.std() * np.sqrt(252)
                sharpe_ratio = annualized_return / volatility if volatility != 0 else 0

                results.append({
                    'Long Threshold': long_threshold,
                    'Short Threshold': short_threshold,
                    'Total Return': total_return,
                    'Annualized Return': annualized_return,
                    'Volatility': volatility,
                    'Sharpe Ratio': sharpe_ratio
                })

        # Create DataFrame
        results_df = pd.DataFrame(results)

        # Find optimal thresholds
        if len(results_df) > 0:
            optimal_idx = results_df['Sharpe Ratio'].idxmax()
            optimal_long = results_df.loc[optimal_idx, 'Long Threshold']
            optimal_short = results_df.loc[optimal_idx, 'Short Threshold']
        else:
            optimal_long = 0.7
            optimal_short = 0.3

        return optimal_long, optimal_short, results_df

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
            axes[0].plot(self.squid_vwap)
            axes[0].set_title('Squid_Ink Price')
            axes[0].grid(True)

            # Plot indicators
            for i, (name, indicator) in enumerate(self.indicators.items(), 1):
                axes[i].plot(indicator)
                axes[i].set_title(name)
                axes[i].grid(True)

                # Add horizontal lines for thresholds
                axes[i].axhline(y=0.7, color='g', linestyle='--', alpha=0.5)
                axes[i].axhline(y=0.3, color='r', linestyle='--', alpha=0.5)

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

            # Add buy-and-hold for comparison
            plt.plot(self.log_returns.cumsum(), label='Buy & Hold', linestyle='--')

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

    def plot_drawdowns(self, save_path=None):
        """
        Plot drawdowns for all strategies.

        Parameters:
            save_path (str): Path to save the plot (if None, just displays)
        """
        if not self.returns:
            self.evaluate_all_strategies()

        try:
            plt.figure(figsize=(12, 6))

            for name, returns in self.returns.items():
                # Calculate drawdowns
                cum_returns = returns.cumsum()
                running_max = cum_returns.cummax()
                drawdown = cum_returns - running_max

                plt.plot(drawdown, label=name)

            plt.title('Drawdowns of Momentum Strategies')
            plt.legend()
            plt.grid(True)

            if save_path:
                plt.savefig(save_path)
                print(f"Saved drawdowns plot to {save_path}")
            else:
                plt.show()

            plt.close()
        except Exception as e:
            print(f"Error creating drawdowns plot: {e}")

# Main function to run the backtest
def run_squid_momentum_backtest():
    """Run the Squid Momentum Backtest."""
    # Create the backtest
    backtest = SquidMomentumBacktest(lookback=10)

    # Calculate all indicators
    print("\nCalculating momentum indicators...")
    backtest.calculate_all_indicators()

    # Plot indicators
    print("\nPlotting indicators...")
    backtest.plot_indicators('squid_momentum_indicators.png')

    # Evaluate all strategies
    print("\nEvaluating momentum strategies...")
    performance = backtest.evaluate_all_strategies()

    print("\nPerformance Metrics:")
    print(performance[['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Profit Factor']].round(4))

    # Plot returns
    print("\nPlotting returns...")
    backtest.plot_returns('squid_momentum_returns.png')

    # Plot drawdowns
    print("\nPlotting drawdowns...")
    backtest.plot_drawdowns('squid_momentum_drawdowns.png')

    # Find the best strategy
    best_strategy = performance['Sharpe Ratio'].idxmax()
    best_sharpe = performance.loc[best_strategy, 'Sharpe Ratio']

    # Optimize thresholds for the best strategy
    print(f"\nOptimizing thresholds for {best_strategy}...")
    optimal_long, optimal_short, threshold_results = backtest.optimize_thresholds(best_strategy)

    print(f"Optimal thresholds for {best_strategy}:")
    print(f"- Long threshold: {optimal_long:.2f}")
    print(f"- Short threshold: {optimal_short:.2f}")

    # Re-evaluate with optimal thresholds
    print("\nRe-evaluating with optimal thresholds...")
    backtest.get_portfolio(best_strategy, optimal_long, optimal_short)
    backtest.get_returns(best_strategy)
    optimal_metrics = backtest.get_performance_metrics(best_strategy)

    print(f"\nOptimal {best_strategy} Performance:")
    for metric, value in optimal_metrics.items():
        print(f"- {metric}: {value:.4f}")

    print("\nMomentum backtest completed!")
    print("\nSummary:")
    print(f"- Tested {len(backtest.indicators)} momentum strategies")
    print(f"- Best strategy: {best_strategy} (Sharpe Ratio: {best_sharpe:.4f})")
    print(f"- Optimal {best_strategy} with thresholds ({optimal_long:.2f}, {optimal_short:.2f}):")
    print(f"  - Total Return: {optimal_metrics['Total Return']:.4f}")
    print(f"  - Sharpe Ratio: {optimal_metrics['Sharpe Ratio']:.4f}")
    print(f"  - Win Rate: {optimal_metrics['Win Rate']:.4f}")
    print(f"  - Profit Factor: {optimal_metrics['Profit Factor']:.4f}")

    return {
        'backtest': backtest,
        'performance': performance,
        'best_strategy': best_strategy,
        'optimal_thresholds': (optimal_long, optimal_short),
        'optimal_metrics': optimal_metrics
    }

if __name__ == "__main__":
    print("Starting Squid Momentum Backtest...")
    try:
        results = run_squid_momentum_backtest()
        print("Backtest completed successfully!")
    except Exception as e:
        import traceback
        print(f"Error running Squid Momentum Backtest: {e}")
        traceback.print_exc()
