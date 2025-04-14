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
            # Preserve indentation by detecting the leading whitespace
            leading_whitespace = ''
            for char in line:
                if char in [' ', '\t']:
                    leading_whitespace += char
                else:
                    break

            if "self.threshold =" in line and "threshold" in params:
                modified_lines.append(f"{leading_whitespace}self.threshold = {params['threshold']}\n")
            elif "self.lookback =" in line and "lookback" in params:
                modified_lines.append(f"{leading_whitespace}self.lookback = {params['lookback']}  # Used as span for EWM\n")
            elif "self.atr_lookback =" in line and "lookback" in params:
                # Update atr_lookback based on lookback
                modified_lines.append(f"{leading_whitespace}self.atr_lookback = 5.0*self.lookback/4.0  # Used as span for EWM of absolute differences\n")
            elif "self.cmma_smooth_lookback =" in line and "cmma_smooth_lookback" in params:
                modified_lines.append(f"{leading_whitespace}self.cmma_smooth_lookback = {params['cmma_smooth_lookback']}  # Used as span for smoothing the CMMA difference\n")
            elif "self.max_position =" in line and "max_position" in params:
                modified_lines.append(f"{leading_whitespace}self.max_position = {params['max_position']}  # Maximum number of spread units to hold (not individual instruments)\n")
            elif "self.price_adjustment =" in line and "price_adjustment" in params:
                modified_lines.append(f"{leading_whitespace}self.price_adjustment = {params['price_adjustment']}\n")
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
    exit_code = os.system(f"{cmd} > {output_file}")

    # Check if the command executed successfully
    if exit_code != 0:
        print(f"Warning: Command exited with code {exit_code}")

    # Read the output from the file
    try:
        # Check if the file exists and is not empty
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            with open(output_file, 'r') as f:
                output = f.read()
        else:
            print(f"Warning: Output file {output_file} does not exist or is empty")
            # Try running the command directly and capturing output
            try:
                import subprocess
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                output = result.stdout
                print(f"Captured output directly from subprocess: {len(output)} characters")
            except Exception as e:
                print(f"Error running command directly: {e}")
                output = ""

        # Try to find the profit using multiple patterns
        profit = 0

        # List of regex patterns to try
        profit_patterns = [
            r"Total profit: ([\d,]+)",  # Standard format
            r"Total profit:\s*([\d,]+)",  # With variable whitespace
            r"Total profit[^\d]*(\d[\d,]*)",  # More flexible pattern
            r"profit: ([\d,]+)",  # Lowercase variant
            r"Profit: ([\d,]+)"   # Capital variant
        ]

        # Try each pattern
        profit_found = False
        for pattern in profit_patterns:
            profit_match = re.search(pattern, output)
            if profit_match:
                profit_str = profit_match.group(1).replace(',', '')
                profit = int(profit_str)
                print(f"Found profit: {profit} (using pattern: {pattern})")
                profit_found = True
                break

        # If no profit found with regex patterns, try to extract from the last lines
        if not profit_found:
            print(f"Warning: Could not find profit with standard patterns for {round_day} with params {params}")

            # Get the last few lines of output
            last_lines = output.strip().split('\n')[-5:]

            # Look for lines that might contain the profit
            for line in last_lines:
                # Look for lines with 'profit', 'total', or just numbers
                if 'profit' in line.lower() or 'total' in line.lower() or re.search(r'\d{3,}', line):
                    # Extract all numbers from the line
                    numbers = re.findall(r'\b\d[\d,]*\b', line)
                    if numbers:
                        # Take the largest number as the profit
                        largest_num = max([int(n.replace(',', '')) for n in numbers])
                        profit = largest_num
                        print(f"Extracted profit from line: '{line.strip()}' -> {profit}")
                        profit_found = True
                        break

            # If still no profit found, look for any numbers in the output
            if not profit_found:
                all_numbers = re.findall(r"\b\d[\d,]*\b", output)
                if all_numbers:
                    # Filter out small numbers and timestamps
                    potential_profits = [int(n.replace(',', '')) for n in all_numbers if len(n) >= 3 and int(n.replace(',', '')) > 100]
                    if potential_profits:
                        # Take the largest number as the profit
                        profit = max(potential_profits)
                        print(f"Using largest number as profit: {profit}")
                        print(f"Found these numbers in output: {all_numbers[-15:]}")
                    else:
                        print(f"Found these numbers in output, but none look like profits: {all_numbers[-15:]}")
    except Exception as e:
        print(f"Error processing output: {e}")
        profit = 0

    # Clean up
    try:
        # Only remove files if they exist
        if os.path.exists(temp_trader_path):
            os.remove(temp_trader_path)
        if os.path.exists(output_file):
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
    parser = argparse.ArgumentParser(description='Grid search for CMMA difference trader parameters')
    parser.add_argument('--trader', type=str, default='backtester/traders/cmma_difference_trader.py',
                        help='Path to the trader file')
    parser.add_argument('--rounds', type=str, nargs='+', default=['2--1', '2-0', '2-1'],
                        help='Round and day combinations to test')
    parser.add_argument('--threshold', type=float, nargs='+', default=[0.5, 0.7, 0.9, 1.1],
                        help='Threshold values to test')
    parser.add_argument('--lookback', type=int, nargs='+', default=[400, 600, 800, 1000],
                        help='Lookback period values to test')
    parser.add_argument('--cmma-smooth-lookback', type=int, nargs='+', default=[100, 200, 300, 400],
                        help='CMMA smoothing lookback period values to test')
    parser.add_argument('--max-position', type=int, nargs='+', default=[20, 25, 30, 35],
                        help='Maximum position values to test')
    parser.add_argument('--price-adjustment', type=int, nargs='+', default=[1, 2, 3, 4],
                        help='Price adjustment values to test')
    parser.add_argument('--output', type=str, default='cmma_grid_search_results.csv',
                        help='Output file for results')

    args = parser.parse_args()

    # Define parameter grid
    param_grid = {
        'threshold': args.threshold,
        'lookback': args.lookback,
        'cmma_smooth_lookback': args.cmma_smooth_lookback,
        'max_position': args.max_position,
        'price_adjustment': args.price_adjustment
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
