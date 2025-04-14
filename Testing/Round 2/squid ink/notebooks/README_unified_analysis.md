# Unified Orderbook Analysis Before Large Returns - Squid Ink Round 2

This directory contains a series of notebooks that analyze the state of the orderbook right before large changes in returns for Squid Ink in Round 2. The analysis is split into multiple parts to make it easier to run and understand each step separately.

## Notebooks

### 1. unified_orderbook_analysis_part1.ipynb

**Part 1: Load Data and Identify Large Return Events**
- Loads the price data for Squid Ink in Round 2
- Calculates mid prices and returns
- Identifies large return events (top 1% of absolute returns)
- Visualizes these events on the price chart
- Saves the processed data for use in Part 2

### 2. unified_orderbook_analysis_part2.ipynb

**Part 2: Extract Orderbook Features**
- Loads the data from Part 1
- Calculates various orderbook features (spread, volume imbalance, etc.)
- Extracts the orderbook state right before large return events
- Creates a DataFrame of pre-event orderbook states for analysis
- Visualizes the distribution of key features
- Saves the processed data for use in Part 3

### 3. unified_orderbook_analysis_part3.ipynb

**Part 3: Analyze Relationship Between Orderbook Features and Returns**
- Loads the data from Part 2
- Separates positive and negative return events
- Compares orderbook features between positive and negative returns
- Calculates correlations between features and returns
- Performs statistical tests to identify significant differences
- Uses machine learning to determine feature importance
- Saves the processed data for use in Part 4

### 4. unified_orderbook_analysis_part4.ipynb

**Part 4: Visualize Orderbook Patterns**
- Loads the data from Part 3
- Visualizes individual examples of large positive and negative return events
- Shows the orderbook depth at specific timestamps
- Creates visualizations of average orderbook patterns before positive vs negative returns
- Draws conclusions about the patterns observed

## How to Use

1. Run the notebooks in order (Part 1 → Part 2 → Part 3 → Part 4)
2. Each notebook loads the data saved by the previous notebook
3. Make sure to run all cells in each notebook before proceeding to the next one
4. The data directory (`../data`) is created automatically to store intermediate results

## Dependencies

These notebooks require the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn (optional, for better styling)
- scipy (for statistical tests in Part 3)
- scikit-learn (for machine learning analysis in Part 3)

## Data

The notebooks use price data for Squid Ink from Round 2, which includes orderbook information (bid/ask prices and volumes at different levels).
