import os
import subprocess
import re
import itertools
import pandas as pd
from tqdm import tqdm
import argparse
import time

def run_backtest(trader_path, round_day, params):
    """
    Run a backtest with the specified parameters
    
    Args:
        trader_path: Path to the trader file
        round_day: Round and day in format "round--day" or "round-day"
        params: Dictionary of parameters to set
    
    Returns:
        Total profit for the day
    """
    # Create a temporary copy of the trader file with modified parameters
    try:
        with open(trader_path, 'r') as f:
            lines = f.readlines()
        
        # Find and replace the parameter lines
        modified_lines = []
        for line in lines:
            if "self.entry_threshold =" in line and "entry_threshold" in params:
                modified_lines.append(f"        self.entry_threshold = {params['entry_threshold']}\n")
            elif "self.exit_threshold =" in line and "exit_threshold" in params:
                modified_lines.append(f"        self.exit_threshold = {params['exit_threshold']}\n")
            elif "self.entry_price_adjustment =" in line and "entry_price_adjustment" in params:
                modified_lines.append(f"        self.entry_price_adjustment = {params['entry_price_adjustment']}\n")
            elif "self.exit_price_adjustment =" in line and "exit_price_adjustment" in params:
                modified_lines.append(f"        self.exit_price_adjustment = {params['exit_price_adjustment']}\n")
            else:
                modified_lines.append(line)
        
        # Save the modified trader code to a temporary file
        temp_trader_path = trader_path.replace('.py', f'_temp_{int(time.time())}.py')
        with open(temp_trader_path, 'w') as f:
            f.writelines(modified_lines)
    except Exception as e:
        print(f"Error modifying trader file: {e}")
        return 0
    
    # Run the backtest
    cmd = f"prosperity3bt {temp_trader_path} {round_day}"
    print(f"Running command: {cmd}")
    
    # Run the command and capture output
    output_file = f"backtest_output_{round_day.replace('-', '_')}.txt"
    os.system(f"{cmd} > {output_file}")
    
    # Read the output from the file
    try:
        with open(output_file, 'r') as f:
            output = f.read()
        
        # Try to find the profit
        profit = 0
        profit_match = re.search(r"Total profit: ([\d,]+)", output)
        if profit_match:
            profit_str = profit_match.group(1).replace(',', '')
            profit = int(profit_str)
            print(f"Found profit: {profit}")
        else:
            print(f"Warning: Could not find profit in output for {round_day} with params {params}")
            # Look for any numbers that might be the profit
            all_numbers = re.findall(r"\b\d[\d,]*\b", output)
            if all_numbers:
                print(f"Found these numbers in output: {all_numbers[-10:]}")
    except Exception as e:
        print(f"Error processing output: {e}")
        profit = 0
    
    # Clean up
    try:
        os.remove(temp_trader_path)
        os.remove(output_file)
    except Exception as e:
        print(f"Error cleaning up: {e}")
    
    return profit

def grid_search(trader_path, round_days, param_grid):
    """
    Perform a grid search over the parameter space
    
    Args:
        trader_path: Path to the trader file
        round_days: List of round and day combinations to test
        param_grid: Dictionary mapping parameter names to lists of values to try
    
    Returns:
        DataFrame with results
    """
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(itertools.product(*[param_grid[name] for name in param_names]))
    
    # Create a list to store results
    results = []
    
    # Total number of combinations to test
    total_combinations = len(param_values) * len(round_days)
    
    # Run grid search with progress bar
    with tqdm(total=total_combinations, desc="Grid Search Progress") as pbar:
        for params_tuple in param_values:
            params = dict(zip(param_names, params_tuple))
            
            # Test these parameters on each round/day
            day_profits = {}
            total_profit = 0
            
            try:
                for round_day in round_days:
                    profit = run_backtest(trader_path, round_day, params)
                    day_profits[round_day] = profit
                    total_profit += profit
                    pbar.update(1)
            except Exception as e:
                print(f"Error during backtest with params {params}: {e}")
                # Fill in missing days with 0 profit
                for round_day in round_days:
                    if round_day not in day_profits:
                        day_profits[round_day] = 0
                        pbar.update(1)
            
            # Store the results
            result = {
                **params,
                **day_profits,
                'total_profit': total_profit
            }
            results.append(result)
    
    # Convert results to DataFrame and sort by total profit
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('total_profit', ascending=False)
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Grid search for custom spread trader parameters')
    parser.add_argument('--trader', type=str, default='backtester/traders/custom_spread_trader.py',
                        help='Path to the trader file')
    parser.add_argument('--rounds', type=str, nargs='+', default=['2--1', '2-0'],
                        help='Round and day combinations to test')
    parser.add_argument('--entry', type=float, nargs='+', default=[5.0, 10.0, 15.0, 20.0, 25.0],
                        help='Entry threshold values to test')
    parser.add_argument('--exit', type=float, nargs='+', default=[1.0, 2.0, 3.0, 5.0],
                        help='Exit threshold values to test')
    parser.add_argument('--entry-pay', type=float, nargs='+', default=[0.0, 1.0, 2.0, 3.0],
                        help='Willingness to pay values to test for entering positions')
    parser.add_argument('--exit-pay', type=float, nargs='+', default=[0.0, 1.0, 2.0, 3.0],
                        help='Willingness to pay values to test for exiting positions')
    parser.add_argument('--output', type=str, default='grid_search_results.csv',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    # Define parameter grid
    param_grid = {
        'entry_threshold': args.entry,
        'exit_threshold': args.exit,
        'entry_price_adjustment': args.entry_pay,
        'exit_price_adjustment': args.exit_pay
    }
    
    # Run grid search
    results = grid_search(args.trader, args.rounds, param_grid)
    
    # Save results to CSV
    try:
        results.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")
    
    # Print top 10 results
    print("\nTop 10 Parameter Combinations:")
    print(results.head(10).to_string())
    
    # Print best parameters
    best_params = results.iloc[0]
    print("\nBest Parameters:")
    for param in param_grid.keys():
        print(f"{param}: {best_params[param]}")
    
    print(f"\nBest Total Profit: {best_params['total_profit']}")
    for round_day in args.rounds:
        print(f"Profit for {round_day}: {best_params[round_day]}")

if __name__ == "__main__":
    main()
