# Attention Layers API

## BahdanauAttention

Additive attention mechanism from Bahdanau et al. (2014).

### Constructor

```python
BahdanauAttention(units, name="bahdanau_attention")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `units` | int | Number of attention units |
| `name` | str | Layer name (optional) |

### Methods

#### `call(query, values)`

Compute attention context and weights.

```python
context_vector, attention_weights = attention(query, values)
```

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `query` | `(batch, hidden_dim)` | Decoder hidden state |
| `values` | `(batch, seq_len, hidden_dim)` | Encoder outputs |

| Returns | Shape | Description |
|---------|-------|-------------|
| `context_vector` | `(batch, hidden_dim)` | Weighted sum of values |
| `attention_weights` | `(batch, seq_len, 1)` | Softmax weights |

### Example

```python
attention = BahdanauAttention(units=512)

# Encoder outputs: 10 timesteps, 512 hidden dim
encoder_output = tf.random.normal([64, 10, 512])

# Decoder state
decoder_hidden = tf.random.normal([64, 512])

context, weights = attention(decoder_hidden, encoder_output)
print(context.shape)   # (64, 512)
print(weights.shape)   # (64, 10, 1)
```

---

## LuongAttention

Multiplicative attention mechanism from Luong et al. (2015).

### Constructor

```python
LuongAttention(units, score_type='general', name="luong_attention")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `units` | int | - | Number of attention units |
| `score_type` | str | `'general'` | One of `'dot'`, `'general'`, `'concat'` |
| `name` | str | `"luong_attention"` | Layer name |

### Score Types

| Type | Formula | Use Case |
|------|---------|----------|
| `'dot'` | `sᵀh` | Same dimensions, fastest |
| `'general'` | `sᵀWh` | Different dimensions |
| `'concat'` | `vᵀtanh(W[s;h])` | Maximum expressiveness |

### Methods

#### `call(query, values)`

Same signature as BahdanauAttention.

### Example

```python
# General attention (default)
attention = LuongAttention(units=512, score_type='general')

# Dot product attention
attention_dot = LuongAttention(units=512, score_type='dot')

# Use like Bahdanau
context, weights = attention(decoder_hidden, encoder_output)
```

---

[← API Overview](overview.md) | [Next: Encoder & Decoder →](encoder-decoder.md)
