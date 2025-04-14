# Orderbook Analysis Before Large Returns - Squid Ink Round 2

This directory contains a series of notebooks that analyze the state of the orderbook right before large changes in returns for Squid Ink in Round 2. The analysis is split into multiple notebooks to make it easier to run and understand each part separately.

## Notebooks

### 1. orderbook_before_large_returns.ipynb

This notebook loads the price data for Squid Ink and calculates returns to identify periods with large price movements. It:
- Loads the price data for Round 2
- Calculates mid price and returns
- Identifies large return events (top 1% of absolute returns)
- Visualizes these events on the price chart

### 2. orderbook_features.ipynb

This notebook extracts and analyzes orderbook features right before large return events. It:
- Defines a function to calculate various orderbook features (spread, volume imbalance, etc.)
- Calculates these features for the entire dataset
- Extracts the orderbook state right before large return events
- Creates a DataFrame of pre-event orderbook states for analysis

### 3. orderbook_analysis.ipynb

This notebook analyzes the relationship between orderbook features and subsequent returns. It:
- Separates positive and negative return events
- Compares orderbook features between positive and negative returns
- Calculates correlations between features and returns
- Performs statistical tests to identify significant differences
- Uses machine learning to determine feature importance
- Summarizes the findings

### 4. orderbook_visualization.ipynb

This notebook creates visualizations to better understand the orderbook patterns before large return events. It:
- Visualizes individual examples of large positive and negative return events
- Shows the orderbook depth at specific timestamps
- Creates visualizations of average orderbook patterns before positive vs negative returns
- Draws conclusions about the patterns observed

## How to Use

1. Run the notebooks in order (1-4)
2. Each notebook builds on the results of the previous ones
3. You can also run individual notebooks if you're only interested in specific aspects of the analysis

## Dependencies

These notebooks require the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn (optional, for better styling)
- scipy (for statistical tests)
- scikit-learn (for machine learning analysis)

## Data

The notebooks use price data for Squid Ink from Round 2, which includes orderbook information (bid/ask prices and volumes at different levels).
