import os
import subprocess
import re
import time

def run_backtest(trader_path, round_day, entry, exit, entry_pay, exit_pay):
    """Run a single backtest and extract the profit"""
    # Create a temporary copy of the trader file with modified parameters
    with open(trader_path, 'r') as f:
        lines = f.readlines()
    
    # Find and replace the parameter lines
    modified_lines = []
    for line in lines:
        if "self.entry_threshold =" in line:
            modified_lines.append(f"        self.entry_threshold = {entry}\n")
        elif "self.exit_threshold =" in line:
            modified_lines.append(f"        self.exit_threshold = {exit}\n")
        elif "self.entry_price_adjustment =" in line:
            modified_lines.append(f"        self.entry_price_adjustment = {entry_pay}\n")
        elif "self.exit_price_adjustment =" in line:
            modified_lines.append(f"        self.exit_price_adjustment = {exit_pay}\n")
        else:
            modified_lines.append(line)
    
    # Save the modified trader code to a temporary file
    temp_trader_path = trader_path.replace('.py', f'_temp_{int(time.time())}.py')
    with open(temp_trader_path, 'w') as f:
        f.writelines(modified_lines)
    
    # Run the backtest directly with os.system
    cmd = f"prosperity3bt {temp_trader_path} {round_day}"
    print(f"Running command: {cmd}")
    
    # Run the command and capture output
    output_file = f"backtest_output_{round_day.replace('-', '_')}.txt"
    os.system(f"{cmd} > {output_file}")
    
    # Read the output from the file
    try:
        with open(output_file, 'r') as f:
            output = f.read()
        
        print(f"Output preview:")
        print(output[:500])
        
        # Try to find the profit
        profit = 0
        profit_match = re.search(r"Total profit: ([\d,]+)", output)
        if profit_match:
            profit_str = profit_match.group(1).replace(',', '')
            profit = int(profit_str)
            print(f"Found profit: {profit}")
        else:
            print("Could not find profit in output")
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

def main():
    # Test a few parameter combinations
    trader_path = "backtester/traders/custom_spread_trader.py"
    round_day = "2--1"
    
    # Test parameters
    params = [
        (5.0, 1.0, 0.0, 1.0),  # entry, exit, entry_pay, exit_pay
        (10.0, 2.0, 1.0, 0.0),
        (15.0, 3.0, 0.0, 0.0)
    ]
    
    results = []
    for entry, exit, entry_pay, exit_pay in params:
        print(f"\nTesting parameters: entry={entry}, exit={exit}, entry_pay={entry_pay}, exit_pay={exit_pay}")
        profit = run_backtest(trader_path, round_day, entry, exit, entry_pay, exit_pay)
        results.append((entry, exit, entry_pay, exit_pay, profit))
    
    # Print results
    print("\nResults:")
    for entry, exit, entry_pay, exit_pay, profit in results:
        print(f"entry={entry}, exit={exit}, entry_pay={entry_pay}, exit_pay={exit_pay}: profit={profit}")

if __name__ == "__main__":
    main()
