#!/bin/bash

# Setup script for VFM-Guidance repository
# This script automates the environment setup process

echo "========================================="
echo "VFM-Guidance Setup Script"
echo "========================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing required packages..."
pip install -r requirements.txt

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start Jupyter Notebook, run:"
echo "  jupyter notebook"
echo ""
echo "Then navigate to the notebooks/ directory and open any notebook."
echo ""
