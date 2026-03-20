#!/bin/bash

# Check if Python 3.10 is available
if ! command -v python3.10 &> /dev/null; then
    echo "Error: Python 3.10 is required but not found"
    echo "Please install Python 3.10 before running this script"
    exit 1
fi

# Create and activate virtual environment with Python 3.10
echo "Creating virtual environment with Python 3.10..."
python3.10 -m venv venv
source venv/bin/activate

# Verify Python version
PYTHON_VERSION=$(python --version | cut -d' ' -f2 | cut -d'.' -f1,2)
if [ "$PYTHON_VERSION" != "3.10" ]; then
    echo "Error: Virtual environment is not using Python 3.10"
    echo "Got: Python $PYTHON_VERSION"
    exit 1
fi

echo "Using Python $(python --version)"

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

echo "Installation complete!"
echo "To activate the environment, run: source venv/bin/activate"