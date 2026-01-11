#!/bin/bash

# VTA Tracking Run Script
# This script sets up and runs the tap detection system

echo "Starting VTA Tracking System..."

# Create necessary directories
mkdir -p logs uploads outputs static

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Run the application
echo "Starting FastAPI server..."
python main.py

# Deactivate virtual environment on exit
deactivate
