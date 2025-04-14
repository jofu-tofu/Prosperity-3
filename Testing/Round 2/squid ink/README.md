# Squid Ink Analysis - Round 2

This directory contains analysis and testing of various trading strategies for the Squid_Ink asset in Round 2.

## Directory Structure

- **notebooks/**: Contains Jupyter notebooks for interactive analysis
  - `squid_vwap_analysis.ipynb`: Analysis of VWAP (Volume-Weighted Average Price) for Squid Ink

## Usage

### Running Notebooks

To run the Jupyter notebooks:

```powershell
cd notebooks
jupyter notebook
```

Or to run a specific notebook:

```powershell
jupyter notebook notebooks/squid_vwap_analysis.ipynb
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

Alternatively, you can use the util_minimal module:

```python
import sys
import os
sys.path.append(os.path.abspath('../../data_utils'))
import util_minimal

# Load all price data for Round 2
prices = util_minimal.load_all_price_data(2)

# Filter for SQUID_INK
squid_data = prices[prices['product'] == 'SQUID_INK']

# Calculate VWAP
squid_vwap = util_minimal.get_vwap(squid_data, 'SQUID_INK')
```
