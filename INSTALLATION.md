# Installation Guide

This project supports Python 3.10 and can be installed using either conda or pip virtual environments.

## Prerequisites

- Python 3.10 (required)
- Git
- Either conda/miniconda or pip/venv

## Option 1: Using Conda (Recommended)

### Create environment from environment.yml

```bash
# Clone the repository (if not already done)
git clone <your-repo-url>
cd Thesis

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate thesis
```

### Manual conda setup

```bash
# Create conda environment with Python 3.10
conda create -n thesis python=3.10

# Activate environment
conda activate thesis

# Install packages using pip (for full compatibility)
pip install -r requirements.txt
```

## Option 2: Using Python venv

```bash
# Clone the repository (if not already done)
git clone <your-repo-url>
cd Thesis

# Create virtual environment
python3.10 -m venv thesis_env

# Activate environment
# On macOS/Linux:
source thesis_env/bin/activate
# On Windows:
thesis_env\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## Verification

After installation, verify the setup works:

```python
import torch
import transformers
import bertopic
import pandas as pd
import numpy as np
import streamlit

print("✅ All packages installed successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"BERTopic version: {bertopic.__version__}")
```

## Common Issues

### CUDA Support (Optional)

If you need CUDA support for GPU acceleration:

```bash
# For conda users
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For pip users
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues

If you encounter memory issues during installation:

```bash
# Increase pip timeout and use no-cache
pip install --no-cache-dir --timeout 1000 -r requirements.txt
```

### Package Conflicts

If you encounter package conflicts:

1. Try creating a fresh environment
2. Install packages in smaller groups
3. Use `pip install --force-reinstall` for specific packages

## Development Setup

For development, install additional tools:

```bash
pip install jupyter lab black flake8 pytest
```

## Notes

- The `requirements.txt` uses flexible version ranges for better compatibility
- The `environment.yml` optimizes package sources (conda vs pip)
- All packages are tested with Python 3.10
- GPU support requires additional CUDA packages (see above)




