#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

echo "Setup complete. To run the project, use: source venv/bin/activate && python main.py"
