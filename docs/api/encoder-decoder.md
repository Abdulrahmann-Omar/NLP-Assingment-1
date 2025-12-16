# Encoder & Decoder API

## Encoder

GRU-based sequence encoder.

### Constructor

```python
Encoder(vocab_size, embedding_dim, enc_units, batch_sz)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `vocab_size` | int | Size of source vocabulary |
| `embedding_dim` | int | Embedding dimensions |
| `enc_units` | int | GRU hidden units |
| `batch_sz` | int | Default batch size |

### Methods

#### `call(x, hidden)`

Encode a batch of sequences.

```python
output, state = encoder(input_tensor, initial_hidden)
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `x` | `(batch, seq_len)` | Token IDs |
| `hidden` | `(batch, enc_units)` | Initial hidden state |

| Returns | Shape | Description |
|---------|-------|-------------|
| `output` | `(batch, seq_len, enc_units)` | All hidden states |
| `state` | `(batch, enc_units)` | Final hidden state |

#### `initialize_hidden_state(batch_size=None)`

Create zero-initialized hidden state.

```python
hidden = encoder.initialize_hidden_state(batch_size=32)
# Returns: tf.zeros of shape (32, enc_units)
```

### Example

```python
encoder = Encoder(
    vocab_size=10000,
    embedding_dim=256,
    enc_units=512,
    batch_sz=64
)

# Initialize hidden state
hidden = encoder.initialize_hidden_state()

# Encode sequence
inputs = tf.constant([[1, 2, 3, 4, 0, 0]] * 64)  # Batch of 64
output, state = encoder(inputs, hidden)

print(output.shape)  # (64, 6, 512)
print(state.shape)   # (64, 512)
```

---

## Decoder

GRU-based decoder with attention.

### Constructor

```python
Decoder(vocab_size, embedding_dim, dec_units, batch_sz, attention_layer)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `vocab_size` | int | Size of target vocabulary |
| `embedding_dim` | int | Embedding dimensions |
| `dec_units` | int | GRU hidden units |
| `batch_sz` | int | Default batch size |
| `attention_layer` | Layer | BahdanauAttention or LuongAttention |

### Methods

#### `call(x, hidden, enc_output)`

Decode one timestep.

```python
predictions, state, attention_weights = decoder(dec_input, dec_hidden, enc_output)
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `x` | `(batch, 1)` | Current token ID |
| `hidden` | `(batch, dec_units)` | Previous decoder state |
| `enc_output` | `(batch, src_len, enc_units)` | Encoder outputs |

| Returns | Shape | Description |
|---------|-------|-------------|
| `predictions` | `(batch, vocab_size)` | Logits over vocabulary |
| `state` | `(batch, dec_units)` | New decoder state |
| `attention_weights` | `(batch, src_len, 1)` | Where model looked |

### Example

```python
# Create attention and decoder
attention = BahdanauAttention(512)
decoder = Decoder(
    vocab_size=8000,
    embedding_dim=256,
    dec_units=512,
    batch_sz=64,
    attention_layer=attention
)

# Single decode step
dec_input = tf.fill([64, 1], start_token_id)
predictions, dec_hidden, attn_weights = decoder(
    dec_input, 
    enc_hidden,   # From encoder
    enc_output    # From encoder
)

# Get predicted token
predicted_id = tf.argmax(predictions, axis=-1)
```

---

[← Attention Layers](attention-layers.md) | [Next: Training Functions →](training-functions.md)
