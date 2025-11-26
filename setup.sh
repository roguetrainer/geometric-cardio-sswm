#!/bin/bash

# Setup script for Geometric Cardio SSWM

echo "================================================"
echo "Geometric Cardio SSWM - Installation Script"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo "✓ Python version $python_version is compatible"
else
    echo "✗ Python version $python_version is too old. Please upgrade to Python 3.8 or higher"
    exit 1
fi

echo ""

# Create virtual environment (optional but recommended)
read -p "Do you want to create a virtual environment? (recommended) [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
    echo ""
    echo "To activate the virtual environment, run:"
    echo "  source venv/bin/activate  (on Linux/Mac)"
    echo "  venv\\Scripts\\activate     (on Windows)"
    echo ""
    
    # Activate virtual environment
    source venv/bin/activate 2>/dev/null || . venv/bin/activate 2>/dev/null || true
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ All dependencies installed successfully"
else
    echo "✗ Error installing dependencies. Please check the error messages above."
    exit 1
fi

echo ""

# Create necessary directories
echo "Creating project directories..."
mkdir -p img
mkdir -p notebooks
mkdir -p docs
mkdir -p src
mkdir -p checkpoints
mkdir -p data
echo "✓ Directories created"
echo ""

# Check PyTorch installation
echo "Verifying PyTorch installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

if [ $? -eq 0 ]; then
    echo "✓ PyTorch installed correctly"
else
    echo "✗ Error importing PyTorch"
    exit 1
fi

echo ""

# Create __init__.py files
echo "Creating Python package structure..."
touch src/__init__.py
echo "✓ Package structure created"
echo ""

# Summary
echo "================================================"
echo "Installation Complete!"
echo "================================================"
echo ""
echo "To get started:"
echo "  1. Activate virtual environment (if created): source venv/bin/activate"
echo "  2. Explore Jupyter notebooks: jupyter notebook notebooks/"
echo "  3. Run example: python -c 'from src.sswm_model import SynthCardioSSWM; print(\"Success!\")'"
echo ""
echo "For more information, see README.md"
echo ""
