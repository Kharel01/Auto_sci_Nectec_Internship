#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "Installing experiment requirements..."
../../../.venv/bin/python -m pip install -r requirements.txt

echo "Running neural network hyperparameter tuning demo..."
../../../.venv/bin/python train.py
