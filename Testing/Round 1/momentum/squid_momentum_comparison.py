"""
Squid_Ink Momentum Strategy Comparison

This script compares different momentum strategies for Squid_Ink and evaluates their performance.
It can be run directly to generate performance metrics and visualizations.
"""

import sys
import os
sys.path.append(os.path.abspath('../'))
import util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import squid_momentum_tests as smt

def compare_momentum_strategies(lookback=10, threshold_high=0.7, threshold_low=0.3, 
                               n_samples=60000, save_plots=True):
    """
    Compare different momentum strategies for Squid_Ink.
    
    Parameters:
        lookback (int): Lookback period for momentum calculations
        threshold_high (float): Threshold for long positions
        threshold_low (float): Threshold for short positions
        n_samples (int): Number of price samples to use
        save_plots (bool): Whether to save plots to files
        
    Returns:
        dict: Dictionary containing performance metrics and other results
    """
    # Load data
    prices, squid_vwap, log_ret = smt.load_data(n_samples)
    
    # Calculate momentum indicators
    print(f"Calculating momentum indicators with lookback={lookback}...")
    
    simple_mom = smt.simple_momentum(squid_vwap, lookback)
    roc_indicator = smt.rate_of_change(squid_vwap, lookback)
    rsi_indicator = smt.rsi(squid_vwap, lookback)
    cmma_indicator = smt.cmma(squid_vwap, lookback)
    macd_indicator = smt.macd(squid_vwap, short_lookback=lookback, long_lookback=lookback*2)
    
    # Create a DataFrame with all indicators
    indicators_df = pd.DataFrame({
        'Simple Momentum': simple_mom,
        'ROC': roc_indicator,
        'RSI': rsi_indicator,
        'CMMA': cmma_indicator,
        'MACD': macd_indicator
    })
    
    # Calculate correlation matrix
    corr_matrix = indicators_df.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Momentum Indicators')
    if save_plots:
        plt.savefig('momentum_correlation.png')
    plt.close()
    
    # Calculate performance
    print("Calculating performance metrics...")
    
    strategies = {
        'Simple Momentum': simple_mom,
        'ROC': roc_indicator,
        'RSI': rsi_indicator,
        'CMMA': cmma_indicator,
        'MACD': macd_indicator
    }
    
    results = {}
    
    for name, signal in strategies.items():
        portfolio = smt.get_portfolio(signal, threshold_high, threshold_low)
        returns = smt.get_returns(log_ret, portfolio)
        metrics = smt.get_performance_metrics(returns)
        results[name] = metrics
    
    # Create performance comparison DataFrame
    performance_df = pd.DataFrame(results).T
    print("\nPerformance Metrics:")
    print(performance_df.round(4))
    
    # Plot cumulative returns
    plt.figure(figsize=(15, 7))
    
    for name, signal in strategies.items():
        portfolio = smt.get_portfolio(signal, threshold_high, threshold_low)
        returns = smt.get_returns(log_ret, portfolio)
        plt.plot(returns.cumsum(), label=name)
    
    plt.title('Cumulative Returns of Momentum Strategies')
    plt.legend()
    plt.grid(True)
    if save_plots:
        plt.savefig('momentum_returns.png')
    plt.close()
    
    # Create combined signals
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min())
    
    # Normalize all signals
    normalized_signals = {}
    for name, signal in strategies.items():
        normalized_signals[name] = normalize(signal)
    
    # Create combined signals
    # 1. Simple average
    combined_avg = pd.DataFrame(normalized_signals).mean(axis=1)
    
    # 2. Weighted average based on performance
    weights = performance_df['Sharpe Ratio']
    weights = weights / weights.sum()
    combined_weighted = pd.DataFrame({
        name: signal * weights[name] for name, signal in normalized_signals.items()
    }).sum(axis=1)
    
    # 3. Voting system (majority rule)
    signals_df = pd.DataFrame(normalized_signals)
    long_votes = (signals_df > threshold_high).sum(axis=1)
    short_votes = (signals_df < threshold_low).sum(axis=1)
    combined_voting = pd.Series(0.5, index=signals_df.index)
    combined_voting[long_votes >= 3] = 0.9  # Long if majority vote for long
    combined_voting[short_votes >= 3] = 0.1  # Short if majority vote for short
    
    # Add combined strategies to our list
    combined_strategies = {
        'Combined (Avg)': combined_avg,
        'Combined (Weighted)': combined_weighted,
        'Combined (Voting)': combined_voting
    }
    
    # Calculate performance for combined strategies
    combined_results = {}
    
    for name, signal in combined_strategies.items():
        portfolio = smt.get_portfolio(signal, threshold_high, threshold_low)
        returns = smt.get_returns(log_ret, portfolio)
        metrics = smt.get_performance_metrics(returns)
        combined_results[name] = metrics
    
    # Create performance comparison DataFrame
    combined_performance_df = pd.DataFrame(combined_results).T
    print("\nCombined Strategies Performance:")
    print(combined_performance_df.round(4))
    
    # Compare combined strategies with individual strategies
    all_performance_df = pd.concat([performance_df, combined_performance_df])
    all_performance_df = all_performance_df.sort_values('Sharpe Ratio', ascending=False)
    print("\nAll Strategies Ranked by Sharpe Ratio:")
    print(all_performance_df.round(4))
    
    # Plot cumulative returns for all strategies
    plt.figure(figsize=(15, 7))
    
    # Plot individual strategies
    for name, signal in strategies.items():
        portfolio = smt.get_portfolio(signal, threshold_high, threshold_low)
        returns = smt.get_returns(log_ret, portfolio)
        plt.plot(returns.cumsum(), label=name, alpha=0.5)
    
    # Plot combined strategies with thicker lines
    for name, signal in combined_strategies.items():
        portfolio = smt.get_portfolio(signal, threshold_high, threshold_low)
        returns = smt.get_returns(log_ret, portfolio)
        plt.plot(returns.cumsum(), label=name, linewidth=2)
    
    plt.title('Cumulative Returns of All Momentum Strategies')
    plt.legend()
    plt.grid(True)
    if save_plots:
        plt.savefig('all_momentum_returns.png')
    plt.close()
    
    # Perform Monte Carlo permutation testing on the best strategy
    best_strategy = all_performance_df.index[0]
    print(f"\nBest strategy: {best_strategy}")
    
    if best_strategy in strategies:
        strategy_func = getattr(smt, best_strategy.lower().replace(' ', '_'))
        signal = strategies[best_strategy]
    else:
        # It's a combined strategy
        strategy_func = None
        signal = combined_strategies[best_strategy]
    
    # Create a portfolio using the best strategy
    best_portfolio = smt.get_portfolio(signal, threshold_high, threshold_low)
    best_returns = smt.get_returns(log_ret, best_portfolio)
    
    # Plot the best strategy's returns
    plt.figure(figsize=(15, 7))
    plt.plot(best_returns.cumsum(), label=f'{best_strategy} Returns')
    plt.plot(log_ret.cumsum(), label='Buy and Hold', alpha=0.5)
    plt.title(f'Returns of Best Strategy: {best_strategy}')
    plt.legend()
    plt.grid(True)
    if save_plots:
        plt.savefig('best_strategy_returns.png')
    plt.close()
    
    # Return results
    return {
        'indicators': indicators_df,
        'correlation': corr_matrix,
        'performance': performance_df,
        'combined_performance': combined_performance_df,
        'all_performance': all_performance_df,
        'best_strategy': best_strategy,
        'best_returns': best_returns
    }

def optimize_lookback_period(strategy_name, min_lookback=5, max_lookback=30, step=5, 
                            threshold_high=0.7, threshold_low=0.3, n_samples=60000, 
                            save_plots=True):
    """
    Optimize the lookback period for a specific momentum strategy.
    
    Parameters:
        strategy_name (str): Name of the strategy to optimize
        min_lookback (int): Minimum lookback period to test
        max_lookback (int): Maximum lookback period to test
        step (int): Step size for lookback periods
        threshold_high (float): Threshold for long positions
        threshold_low (float): Threshold for short positions
        n_samples (int): Number of price samples to use
        save_plots (bool): Whether to save plots to files
        
    Returns:
        dict: Dictionary containing optimization results
    """
    # Load data
    prices, squid_vwap, log_ret = smt.load_data(n_samples)
    
    # Get the strategy function
    if strategy_name == 'Simple Momentum':
        strategy_func = smt.simple_momentum
    elif strategy_name == 'ROC':
        strategy_func = smt.rate_of_change
    elif strategy_name == 'RSI':
        strategy_func = smt.rsi
    elif strategy_name == 'CMMA':
        strategy_func = smt.cmma
    elif strategy_name == 'MACD':
        strategy_func = smt.macd
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    # Test different lookback periods
    lookback_periods = range(min_lookback, max_lookback + 1, step)
    lookback_results = []
    
    for lookback in lookback_periods:
        print(f"Testing lookback period: {lookback}")
        
        if strategy_name == 'MACD':
            signal = strategy_func(squid_vwap, short_lookback=lookback, long_lookback=lookback*2)
        else:
            signal = strategy_func(squid_vwap, lookback=lookback)
        
        portfolio = smt.get_portfolio(signal, threshold_high, threshold_low)
        returns = smt.get_returns(log_ret, portfolio)
        metrics = smt.get_performance_metrics(returns)
        
        lookback_results.append({
            'Lookback': lookback,
            **metrics
        })
    
    lookback_df = pd.DataFrame(lookback_results)
    print("\nLookback Period Optimization Results:")
    print(lookback_df.round(4))
    
    # Find the optimal lookback period
    optimal_lookback = lookback_df.loc[lookback_df['Sharpe Ratio'].idxmax(), 'Lookback']
    print(f"\nOptimal lookback period for {strategy_name}: {optimal_lookback}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(lookback_df['Lookback'], lookback_df['Total Return'], marker='o')
    plt.title('Total Return vs. Lookback Period')
    plt.xlabel('Lookback Period')
    plt.ylabel('Total Return')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(lookback_df['Lookback'], lookback_df['Sharpe Ratio'], marker='o')
    plt.title('Sharpe Ratio vs. Lookback Period')
    plt.xlabel('Lookback Period')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(lookback_df['Lookback'], lookback_df['Volatility'], marker='o')
    plt.title('Volatility vs. Lookback Period')
    plt.xlabel('Lookback Period')
    plt.ylabel('Volatility')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(lookback_df['Lookback'], lookback_df['Max Drawdown'], marker='o')
    plt.title('Max Drawdown vs. Lookback Period')
    plt.xlabel('Lookback Period')
    plt.ylabel('Max Drawdown')
    plt.grid(True)
    
    plt.suptitle(f'Lookback Period Optimization for {strategy_name}')
    plt.tight_layout()
    if save_plots:
        plt.savefig(f'{strategy_name.lower().replace(" ", "_")}_lookback_optimization.png')
    plt.close()
    
    # Calculate the optimal strategy
    if strategy_name == 'MACD':
        optimal_signal = strategy_func(squid_vwap, short_lookback=int(optimal_lookback), 
                                      long_lookback=int(optimal_lookback)*2)
    else:
        optimal_signal = strategy_func(squid_vwap, lookback=int(optimal_lookback))
    
    optimal_portfolio = smt.get_portfolio(optimal_signal, threshold_high, threshold_low)
    optimal_returns = smt.get_returns(log_ret, optimal_portfolio)
    
    # Plot the optimal strategy's returns
    plt.figure(figsize=(15, 7))
    plt.plot(optimal_returns.cumsum(), label=f'Optimal {strategy_name}')
    plt.plot(log_ret.cumsum(), label='Buy and Hold', alpha=0.5)
    plt.title(f'Returns of Optimal {strategy_name} (Lookback={optimal_lookback})')
    plt.legend()
    plt.grid(True)
    if save_plots:
        plt.savefig(f'optimal_{strategy_name.lower().replace(" ", "_")}_returns.png')
    plt.close()
    
    return {
        'lookback_results': lookback_df,
        'optimal_lookback': optimal_lookback,
        'optimal_returns': optimal_returns
    }

if __name__ == "__main__":
    # Compare all momentum strategies
    results = compare_momentum_strategies(lookback=10)
    
    # Optimize the best individual strategy
    best_individual = results['performance']['Sharpe Ratio'].idxmax()
    print(f"\nOptimizing the best individual strategy: {best_individual}")
    optimize_results = optimize_lookback_period(best_individual)
    
    print("\nAnalysis complete. Check the generated plots for visualizations.")
