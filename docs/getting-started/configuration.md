# Configuration Reference

All configurable parameters for training and evaluation.

## Training Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `BATCH_SIZE` | 64 | 16-128 | Samples per training batch |
| `EMBEDDING_DIM` | 256 | 64-512 | Word embedding dimensions |
| `UNITS` | 512 | 128-1024 | Hidden layer size |
| `NUM_EXAMPLES` | 30000 | 1000-50000 | Dataset size |
| `EPOCHS` | 10 | 1-50 | Training iterations |

## Memory Optimization

If running out of memory:

```python
BATCH_SIZE = 32    # Reduce batch size
UNITS = 256        # Smaller hidden layer
```

## Speed Optimization

For faster training:

```python
EPOCHS = 5         # Fewer epochs
NUM_EXAMPLES = 15000  # Smaller dataset
```

## Luong Attention Variants

```python
# Score types available
luong_attention = LuongAttention(UNITS, score_type='general')  # Default
luong_attention = LuongAttention(UNITS, score_type='dot')      # Fastest
luong_attention = LuongAttention(UNITS, score_type='concat')   # Most parameters
```

| Score Type | Formula | Speed | Parameters |
|------------|---------|-------|------------|
| `dot` | `sᵀh` | Fastest | None |
| `general` | `sᵀWh` | Medium | `units × units` |
| `concat` | `vᵀtanh(W[s;h])` | Slowest | `2×units + 1` |

---

[← Quickstart](quickstart.md) | [Back to Index →](../index.md)
