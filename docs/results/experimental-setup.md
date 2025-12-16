# Experimental Setup

Reproducibility details for the attention comparison experiments.

## Hardware

| Component | Specification |
|-----------|---------------|
| **Platform** | Google Colab / Local |
| **GPU** | NVIDIA T4 / RTX 3080 (optional) |
| **RAM** | 16GB+ recommended |
| **Storage** | 2GB for data and checkpoints |

## Software

| Package | Version |
|---------|---------|
| Python | 3.8+ |
| TensorFlow | 2.x |
| NumPy | 1.19+ |
| Matplotlib | 3.3+ |
| scikit-learn | 0.24+ |

```python
# Check your versions
import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
```

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `BATCH_SIZE` | 64 | Reduce to 32 for less memory |
| `EMBEDDING_DIM` | 256 | Word embedding size |
| `UNITS` | 512 | GRU hidden units |
| `NUM_EXAMPLES` | 30,000 | Subset of full dataset |
| `EPOCHS` | 10 | Full training run |
| `OPTIMIZER` | Adam | Default learning rate |
| `LOSS` | Sparse Categorical Crossentropy | With masking |

## Dataset

- **Source**: [TensorFlow Portuguese-English](http://storage.googleapis.com/download.tensorflow.org/data/por-eng.zip)
- **Size**: 30,000 sentence pairs (filtered)
- **Split**: 80% train / 20% validation
- **Preprocessing**: Lowercase, unicode normalized, special tokens

## Reproducibility

```python
# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)

# For fully deterministic results (slower)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
```

## Training Time

| Setup | Time per Epoch | Total (10 epochs) |
|-------|----------------|-------------------|
| GPU (T4) | ~2 min | ~20 min |
| GPU (RTX 3080) | ~1 min | ~10 min |
| CPU only | ~10-15 min | ~2 hours |

---

[← Back to Index](../index.md) | [Next: Performance Analysis →](performance-analysis.md)
