#!/bin/bash
# SOTA Setup Script
# Run: bash setup.sh

set -e

echo "=============================================="
echo "  SOTA — Setup"
echo "=============================================="

# Check Python
echo ""
echo "[1/5] Checking Python..."
python3 --version
pip3 --version

# Install Stockfish
echo ""
echo "[2/5] Installing Stockfish..."
pkg install -y stockfish 2>/dev/null || echo "  Stockfish already installed or not available"

# Install Python dependencies
echo ""
echo "[3/5] Installing Python dependencies..."
pip3 install python-chess pyyaml tqdm numpy

# Install ML dependencies (may take a while)
echo ""
echo "[4/5] Installing ML dependencies..."
pip3 install transformers peft accelerate bitsandbytes safetensors 2>&1 | tail -5

# Install training dependencies
echo ""
echo "[5/5] Installing training dependencies..."
pip3 install trl datasets 2>&1 | tail -5

# Create directories
echo ""
echo "Creating directories..."
mkdir -p data/{raw,processed,checkpoints}
mkdir -p logs
mkdir -p output

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Generate data:  python main.py generate --samples 100"
echo "  2. Train model:    python main.py train"
echo "  3. Play chess:     python main.py play"
echo "  4. Eval position:  python main.py eval 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1'"
echo ""
