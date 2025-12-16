# Quickstart Guide

Run your first attention comparison in under 5 minutes.

## Option 1: Jupyter Notebook (Recommended)

```bash
jupyter notebook seq2seq_attention_comparison.ipynb
```

Click **"Run All Cells"** and wait for results (~30-60 min).

## Option 2: Python Script

```bash
python seq2seq_attention_comparison.py
```

## What Happens

1. **Downloads** English-Portuguese dataset (30K sentence pairs)
2. **Trains** Bahdanau attention model (10 epochs)
3. **Trains** Luong attention model (10 epochs)
4. **Generates** comparison visualizations and summary

## Expected Output

```
============================================================
ATTENTION MECHANISM COMPARISON RESULTS
============================================================
Bahdanau Attention (Additive):
  - Final Loss: ~1.2xxx

Luong Attention (Multiplicative):
  - Final Loss: ~1.2xxx

Better Performance: [Winner] (by X.XXXX loss)
============================================================
```

## Generated Files

| File | Description |
|------|-------------|
| `training_comparison.png` | Training loss curves |
| `attention_comparison_*.png` | Attention heatmaps |
| `results_summary.txt` | Detailed results report |

## Quick Configuration

Edit these values for faster training:

```python
EPOCHS = 5          # Reduce for quick test
NUM_EXAMPLES = 10000  # Smaller dataset
```

---

[← Installation](installation.md) | [Next: Configuration →](configuration.md)
