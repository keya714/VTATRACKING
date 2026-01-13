#!/bin/bash

# VTA Tracking Run Script
# This script sets up and runs the tap detection system

echo "Starting VTA Tracking System..."

# Create necessary directories
mkdir -p logs uploads outputs static

# # Check if virtual environment exists
# if [ ! -d "venv" ]; then
#     echo "Creating virtual environment..."
#     python3 -m venv venv
# fi

# # Activate virtual environment
# echo "Activating virtual environment..."
# source venv/bin/activate

# Install/upgrade dependencies
# echo "Installing dependencies..."
# pip install --upgrade pip
# pip install -r requirements.txt

# Run the application
echo "Starting backend on port 8000 and frontend on port 8501..."

# Start backend in the background
uvicorn main:app --host 0.0.0.0 --port 8501 &
BACKEND_PID=$!

# Start frontend 
python -m http.server 8000
FRONTEND_PID=$!

echo "Backend running on http://localhost:8000 (PID: $BACKEND_PID)"
echo "Frontend running on http://localhost:8501 (PID: $FRONTEND_PID)"
echo "Press Ctrl+C to stop both servers..."

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID

kill -9 $ FRONTEND_PID
