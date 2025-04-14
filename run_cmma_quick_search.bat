@echo off
python cmma_grid_search.py --trader backtester/traders/cmma_difference_trader.py --rounds 2--1 2-0 2-1 --threshold 0.7 0.9 --lookback 800 --cmma-smooth-lookback 200 400 --max-position 30 --price-adjustment 2 --output cmma_quick_search_results.csv
pause
