# Installation Guide

## Prerequisites

- **Python**: 3.8 or higher
- **pip**: Latest version recommended

## Quick Install

```bash
# Clone the repository
git clone https://github.com/Abdulrahmann-Omar/NLP-Assingment-1.git
cd NLP-Assingment-1

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install tensorflow numpy matplotlib scikit-learn
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `tensorflow` | ≥2.0 | Deep learning framework |
| `numpy` | ≥1.19 | Numerical operations |
| `matplotlib` | ≥3.3 | Visualization |
| `scikit-learn` | ≥0.24 | Train/test splitting |

## GPU Setup (Optional)

For faster training with NVIDIA GPU:

```bash
# Install CUDA-enabled TensorFlow
pip install tensorflow[and-cuda]
```

Verify GPU availability:
```python
import tensorflow as tf
print("GPUs:", tf.config.list_physical_devices('GPU'))
```

## Verify Installation

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(f"TensorFlow: {tf.__version__}")
print(f"NumPy: {np.__version__}")
print("✓ All dependencies installed successfully!")
```

---

[← Back to Index](../index.md) | [Next: Quickstart →](quickstart.md)
