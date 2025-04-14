# Squid_Ink Analysis - Round 2

This directory contains analysis and testing of various trading strategies for the Squid_Ink asset in Round 2.

## Directory Structure

- **notebooks/**: Contains Jupyter notebooks for interactive analysis
  - `eda.ipynb`: Exploratory Data Analysis of Round 2 data

## Usage

### Running Notebooks

To run the Jupyter notebooks:

```powershell
cd notebooks
jupyter notebook
```

Or to run a specific notebook:

```powershell
jupyter notebook notebooks/eda.ipynb
```

### Accessing Data

Data for Round 2 can be accessed using the backtester package:

```python
import sys
import os
sys.path.append(os.path.abspath('../../'))
from backtester import get_price_data

# Load SQUID_INK data for Round 2
prices = get_price_data('SQUID_INK', 2)
```

Alternatively, you can use the util module:

```python
import sys
import os
sys.path.append(os.path.abspath('../../'))
import util

# Load all price data for Round 2
prices = util.load_all_price_data(2)

# Filter for SQUID_INK
squid_data = prices[prices['product'] == 'SQUID_INK']
```

## Data Location

The data for Round 2 is expected to be in the following location:
`../../../Prosperity 3 Data/Round 2/`

The data files should be named:
- `prices_round_2_day_-2.csv`
- `prices_round_2_day_-1.csv`
- `prices_round_2_day_0.csv`
