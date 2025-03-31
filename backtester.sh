#!/bin/bash

# Check if 'uv' is installed
if ! command -v uv &> /dev/null
then
    echo "'uv' not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "'uv' has been installed."
else
    echo "'uv' is already installed."
fi

uv sync
cd backtester
uv run streamlit run app.py