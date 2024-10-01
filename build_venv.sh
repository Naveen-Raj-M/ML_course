#!/bin/bash

echo "Creating virtual environment 'venv-mlcourse'..."
python3 -m virtualenv venv-mlcourse

echo "Activating the virtual environment..."
source venv-mlcourse/bin/activate

which python

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "All done! The virtual environment is ready."