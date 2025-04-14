@echo off
python full_grid_search.py --trader backtester/traders/custom_spread_trader.py --rounds 2--1 2-0 2-1 --entry 5 5.25 5.5 5.75 6 --exit 0.5 1 --entry-pay 4 5  --exit-pay 1 0 --output pb2_grid_search_results.csv
pause
