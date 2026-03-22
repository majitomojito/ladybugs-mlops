#!/bin/bash
# One-click setup and run script

set -e

echo "=========================================="
echo "MLOps Demo - Setup Script"
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

echo ""
echo "1. Creating virtual environment..."
python3 -m venv venv

echo "2. Activating virtual environment..."
source venv/bin/activate || . venv/Scripts/activate

echo "3. Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "4. Running tests..."
pytest tests/ -v

echo "5. Running ML pipeline..."
python train.py

echo ""
echo "=========================================="
echo "Setup complete! Check models/ and data/ directories for outputs."
echo "=========================================="
