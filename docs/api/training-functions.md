# Training Functions API

## loss_function

Compute masked cross-entropy loss (ignores padding tokens).

```python
loss = loss_function(real, pred, loss_object)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `real` | Tensor `(batch,)` | Ground truth token IDs |
| `pred` | Tensor `(batch, vocab)` | Predicted logits |
| `loss_object` | Loss | SparseCategoricalCrossentropy |

**Returns**: Scalar mean loss (padding masked out)

### Implementation Logic

```python
def loss_function(real, pred, loss_object):
    # Create mask for non-padding tokens
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    
    # Compute loss
    loss_ = loss_object(real, pred)
    
    # Apply mask
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_mean(loss_)
```

---

## train_step

Single training iteration with gradient update.

```python
@tf.function
def train_step(inp, targ, enc_hidden, encoder, decoder, 
               start_token_id, optimizer, loss_object, max_length_targ)
```

| Parameter | Description |
|-----------|-------------|
| `inp` | Source sequences `(batch, src_len)` |
| `targ` | Target sequences `(batch, tgt_len)` |
| `enc_hidden` | Initial encoder state |
| `encoder` | Encoder model |
| `decoder` | Decoder model |
| `start_token_id` | ID of `<start>` token |
| `optimizer` | Adam optimizer |
| `loss_object` | Loss function |
| `max_length_targ` | Target sequence length |

**Returns**: `batch_loss` - Average loss for this batch

### Flow

1. Encode source → get all hidden states
2. Initialize decoder with encoder final state
3. For each target position:
   - Decode one step (with attention)
   - Compute loss against ground truth
   - Use teacher forcing (feed actual target)
4. Backpropagate gradients

---

## train_model

Full training loop with checkpointing.

```python
history = train_model(
    encoder, decoder, dataset, targ_lang,
    epochs, batch_size, steps_per_epoch, max_length_targ,
    model_name="model"
)
```

| Parameter | Description |
|-----------|-------------|
| `encoder` | Encoder model instance |
| `decoder` | Decoder model instance |
| `dataset` | `tf.data.Dataset` of (input, target) pairs |
| `targ_lang` | Target language tokenizer |
| `epochs` | Number of training epochs |
| `batch_size` | Batch size |
| `steps_per_epoch` | Batches per epoch |
| `max_length_targ` | Max target length |
| `model_name` | Name for checkpoints |

**Returns**: `history` dict with keys:
- `'loss'`: List of epoch losses

### Features

- ✅ Adam optimizer
- ✅ Checkpoint saving every 2 epochs
- ✅ Progress logging every 100 batches
- ✅ Time tracking per epoch

---

[← Encoder & Decoder](encoder-decoder.md) | [Next: Evaluation Functions →](evaluation-functions.md)
