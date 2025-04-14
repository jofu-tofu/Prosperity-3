@echo off
python cmma_grid_search.py --trader backtester/traders/cmma_difference_trader.py --rounds 2--1 2-0 2-1 --threshold 0.2 0.25 0.3 0.4 0.5 0.6 0.66 0.7 --lookback 2400 2600 3000 3500 4000 4500 --cmma-smooth-lookback 100 200 300 400 --max-position 30 --price-adjustment 1 --output cmma_grid_search_results4.csv
pause
