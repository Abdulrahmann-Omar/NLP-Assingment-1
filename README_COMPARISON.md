# Seq2Seq Attention Comparison - Quick Start Guide

## What's Been Created ✓

1. **seq2seq_attention_comparison.py** - Complete Python script
2. **seq2seq_attention_comparison.ipynb** - Jupyter notebook (recommended)
3. **Implementation documentation** - Plan and walkthrough

## How to Run 

### Recommended: Use Jupyter Notebook

Since you already have an environment with TensorFlow (from `NLP_10.ipynb`), the easiest way to run the comparison is:

```bash
cd "c:\a Zewail City\4Y\NLP\Assingment-1"
jupyter notebook seq2seq_attention_comparison.ipynb
```

Then simply **"Run All Cells"** from the Jupyter menu.

### What the Notebook Does

1. **Loads data**: Downloads English-Portuguese dataset (30K pairs)
2. **Trains Bahdanau model**: ~10 epochs, displays progress
3. **Trains Luong model**: ~10 epochs, displays progress  
4. **Creates visualizations**: Loss curves and comparison charts
5. **Shows summary**: Final performance comparison

**Expected Runtime**: 30-60 minutes (depending on your hardware)

## Expected Results

After running, you'll see:

### Console Output
```
============================================================
CREATING BAHDANAU ATTENTION MODEL
============================================================
✓ Bahdanau model created

Training Bahdanau Attention Model...
  Epoch 1 Batch 0 Loss 4.2341
  Epoch 1 Batch 100 Loss 3.8912
Epoch 1 Loss 3.9123 Time: 125.43s
...
✓ Bahdanau training complete

============================================================
CREATING LUONG ATTENTION MODEL
============================================================
✓ Luong model created

Training Luong Attention Model...
...
✓ Luong training complete

============================================================
COMPARISON RESULTS
============================================================
Better Performance: [Model Name] (by X.XXXX loss)
```

### Visualization
A plot showing:
- Training loss curves for both models
- Final loss comparison bar chart

### Files Generated
- `training_comparison.png` - Comparison visualization

## Key Differences Between Attention Mechanisms

| Feature | Bahdanau (Additive) | Luong (Multiplicative) |
|---------|---------------------|------------------------|
| Formula | `v^T * tanh(W₁s + W₂h)` | `s^T * W * h` |
| Parameters | More (3 matrices) | Fewer (1 matrix) |
| Complexity | Higher | Lower |
| Speed | Slower | Faster |
| Accuracy | Slightly better | Very close |

## Customization Options

Edit these variables in the notebook to customize training:

```python
BATCH_SIZE = 64        # Reduce if out of memory
EMBEDDING_DIM = 256    # Embedding size
UNITS = 512           # Hidden units
NUM_EXAMPLES = 30000  # Data size (reduce for faster training)
EPOCHS = 10           # Training epochs (reduce for quicker results)
```

## If You Get Errors

**Out of Memory**: Reduce `BATCH_SIZE` to 32 or `UNITS` to 256

**Too Slow**: Reduce `EPOCHS` to 5 or `NUM_EXAMPLES` to 15000

**No GPU Warning**: Normal - will use CPU (just slower)

## What You'll Learn

By running this comparison, you'll see:
1. How different attention mechanisms perform on the same task
2. Training dynamics of both models
3. Computational trade-offs (speed vs accuracy)
4. Which attention type works better for this specific dataset

Ready to run? Just open the notebook and hit "Run All"! 🚀
